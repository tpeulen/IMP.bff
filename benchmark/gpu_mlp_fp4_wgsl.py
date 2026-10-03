"""FP4 networks in WGSL: the weights kept on the GPU as 4-bit codes.

The float network already runs through the compute door (`gpu/mlp.wgsl`),
but it converts, transposes and uploads every weight on every call, and a
QuantizedNeuralNet never reaches the GPU at all. This is the harness that
says whether an FP4 kernel is worth writing into the plugin:

* the weights stay resident as packed E2M1 codes, eight to a `u32`, with one
  float32 scale per 16-element sub-block (fp4's row scale, mxfp4's E8M0 and
  nvfp4's E4M3 x tensor scale all decoded on the host into that one array);
* each code is decoded to f32 in the shader and the products and sums are
  f32 -- W4A32; the activations are not quantised, unlike the CPU's W4A8;
* two decoders are timed: the bit trick (E2M1's sign, exponent and mantissa
  shifted into f32's, times 2^126 to move the exponent bias from 1 to 127;
  E2M1's 0.5 lands on an f32 subnormal, so it is exact only where the GPU
  keeps subnormals, which WGSL does not promise) and a 16-entry table.

It is a *benchmark*, not a backend: wgpu-py drives it, as in
`benchmark/gpu_diffusion_wgsl.py`. Run it with a build that has `IMP.bff`
importable and `wgpu` installed:

    python benchmark/gpu_mlp_fp4_wgsl.py
"""
import sys, time
import msgpack, numpy as np, wgpu
import IMP.bff as b

ACT = {"identity": 0, "relu": 1, "tanh": 2, "logistic": 3, "softplus": 4, "silu": 5, "sin": 6}
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])

COMMON = """
struct Params { n_rows: u32, n_in: u32, n_out: u32, act: u32,
                w_off: u32, s_off: u32, b_off: u32, stride: u32,
                p_in: u32, p_out: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read>       xin:  array<f32>;
@group(0) @binding(1) var<storage, read_write> xout: array<f32>;
@group(0) @binding(2) var<storage, read>       wgt:  array<u32>;
@group(0) @binding(3) var<storage, read>       scl:  array<f32>;
@group(0) @binding(4) var<storage, read>       bia:  array<f32>;
@group(0) @binding(5) var<uniform>             p:    Params;

fn activate(v: f32, act: u32) -> f32 {
    if (act == 0u) { return v; }
    if (act == 1u) { return max(v, 0.0); }
    if (act == 2u) { return tanh(v); }
    if (act == 3u) { return 1.0 / (1.0 + exp(-v)); }
    if (act == 4u) { if (v > 20.0) { return v; } return log(1.0 + exp(v)); }
    if (act == 5u) { return v / (1.0 + exp(-v)); }
    if (act == 6u) { return sin(v); }
    return v;
}
"""

DECODE_BITS = """
// E2M1 s.ee.m -> f32: the sign to bit 31, ee.m to bits 24..22 (f32's two
// lowest exponent bits and top mantissa bit), then 2^126 rebias. Code 1
// (0.5) is an f32 subnormal before the multiply.
fn dec(c: u32) -> f32 {
    return bitcast<f32>(((c & 8u) << 28u) | ((c & 7u) << 22u)) * 0x1p126f;
}
"""

# The same without subnormals: four times the magnitude as an integer,
# (2 + m) << e for a normal code and 2m for e == 0, converted and scaled.
DECODE_ARITH = """
fn dec(c: u32) -> f32 {
    let e = (c >> 1u) & 3u;
    let m = c & 1u;
    let v = f32(select((2u + m) << e, m << 1u, e == 0u)) * 0.25;
    return select(v, -v, (c & 8u) != 0u);
}
"""

DECODE_LUT = """
const E2M1 = array<f32, 16>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0);
fn dec(c: u32) -> f32 { var t = E2M1; return t[c]; }
"""

# Codes interleaved word-major: word (k / 8) of output o at k8 * n_out + o, so
# neighbouring threads (neighbouring o) read neighbouring words. Scales the
# same way per 16-element sub-block. Activations are stored with the padded
# row stride (a multiple of 32), the padding zero, so the codes' zero padding
# multiplies zeros.
LAYER_FP4 = """
@compute @workgroup_size(64)
fn layer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n_rows * p.p_out;
    var t = gid.x;
    loop {
        if (t >= total) { break; }
        let r = t / p.p_out;
        let o = t % p.p_out;
        if (o >= p.n_out) { xout[t] = 0.0; t = t + p.stride; continue; }
        let xrow = r * p.p_in;
        var acc = 0.0;
        let nsb = p.p_in / 16u;
        for (var sb = 0u; sb < nsb; sb = sb + 1u) {
            var part = 0.0;
            for (var h = 0u; h < 2u; h = h + 1u) {
                let k8 = 2u * sb + h;
                let w = wgt[p.w_off + k8 * p.n_out + o];
                let xb = xrow + 8u * k8;
                for (var j = 0u; j < 8u; j = j + 1u) {
                    part = part + dec((w >> (4u * j)) & 15u) * xin[xb + j];
                }
            }
            acc = acc + part * scl[p.s_off + sb * p.n_out + o];
        }
        xout[t] = activate(acc + bia[p.b_off + o], p.act);
        t = t + p.stride;
    }
}
"""

# The float kernel of gpu/mlp.wgsl with the same padded-stride activations,
# so the three share all host code. `wgt` holds f32 bits.
LAYER_F32 = """
@compute @workgroup_size(64)
fn layer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.n_rows * p.p_out;
    var t = gid.x;
    loop {
        if (t >= total) { break; }
        let r = t / p.p_out;
        let o = t % p.p_out;
        if (o >= p.n_out) { xout[t] = 0.0; t = t + p.stride; continue; }
        let xrow = r * p.p_in;
        _ = scl[0];  // keep the binding in the auto layout
        var acc = bia[p.b_off + o];
        for (var i = 0u; i < p.n_in; i = i + 1u) {
            acc = acc + bitcast<f32>(wgt[p.w_off + i * p.n_out + o]) * xin[xrow + i];
        }
        xout[t] = activate(acc, p.act);
        t = t + p.stride;
    }
}
"""


# The tiled kernel: a workgroup of 64 threads owns BR rows x 32 outputs and
# walks the contraction 32 at a time. The input tile and the weight tile go
# into workgroup memory -- the FP4 weights decoded and scaled on the way in,
# once per workgroup instead of once per row -- and each thread keeps a
# register block of BR/8 rows x 4 outputs (rows tr + 8i, outputs to + 8j).
# Rows are padded to a multiple of BR and every stride is a padded width, so
# there is no bounds check inside the loop.
LOAD_W_F32 = """
fn load_w(lid: u32, k0: u32, o0: u32) {
    for (var e = lid; e < 1024u; e = e + 64u) {
        ws[e] = bitcast<f32>(wgt[p.w_off + (k0 + e / 32u) * p.p_out + o0 + e % 32u]);
    }
}
"""

LOAD_W_FP4 = """
fn load_w(lid: u32, k0: u32, o0: u32) {
    for (var e = lid; e < 128u; e = e + 64u) {
        let k8l = e / 32u;
        let oo = e % 32u;
        let k8 = k0 / 8u + k8l;
        let w = wgt[p.w_off + k8 * p.p_out + o0 + oo];
        let s = scl[p.s_off + (k8 / 2u) * p.p_out + o0 + oo];
        for (var j = 0u; j < 8u; j = j + 1u) {
            ws[(k8l * 8u + j) * 32u + oo] = dec((w >> (4u * j)) & 15u) * s;
        }
    }
}
"""


def layer_tiled(br):
    rm = br // 8
    inner = "\n".join(
        f"            {{ let a = xs[(tr + {8 * i}u) * 33u + kk];"
        + "".join(f" acc[{4 * i + j}] = acc[{4 * i + j}] + a * b{j};" for j in range(4)) + " }"
        for i in range(rm))
    store = "\n".join(
        f"    {{ let r = r0 + tr + {8 * i}u; let o = o0 + to + {8 * j}u; var v = 0.0;\n"
        f"      if (o < p.n_out) {{ v = activate(acc[{4 * i + j}] + bia[p.b_off + o], p.act); }}\n"
        f"      xout[r * p.p_out + o] = v; }}"
        for i in range(rm) for j in range(4))
    return f"""
var<workgroup> xs: array<f32, {br * 33}>;
var<workgroup> ws: array<f32, 1024>;

@compute @workgroup_size(64)
fn layer(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let rt = wid.y + wid.z * 65535u;
    if (rt * {br}u >= p.n_rows) {{ return; }}
    _ = scl[0];
    let r0 = rt * {br}u;
    let o0 = wid.x * 32u;
    let tr = lid / 8u;
    let to = lid % 8u;
    var acc: array<f32, {4 * rm}>;
    for (var k0 = 0u; k0 < p.p_in; k0 = k0 + 32u) {{
        for (var e = lid; e < {br * 32}u; e = e + 64u) {{
            xs[(e / 32u) * 33u + e % 32u] = xin[(r0 + e / 32u) * p.p_in + k0 + e % 32u];
        }}
        load_w(lid, k0, o0);
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk = kk + 1u) {{
            let b0 = ws[kk * 32u + to];
            let b1 = ws[kk * 32u + to + 8u];
            let b2 = ws[kk * 32u + to + 16u];
            let b3 = ws[kk * 32u + to + 24u];
{inner}
        }}
        workgroupBarrier();
    }}
{store}
}}
"""


CHECK = """
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { out[gid.x] = dec(gid.x); }
"""


def pad32(n):
    return (n + 31) // 32 * 32


def random_net(sizes, hidden="tanh", seed=0):
    r = np.random.default_rng(seed)
    layers = []
    for i, (a, c) in enumerate(zip(sizes[:-1], sizes[1:])):
        w = r.normal(scale=1.0 / np.sqrt(a), size=(c, a))
        layers.append({"n_in": a, "n_out": c,
                       "activation": "identity" if i == len(sizes) - 2 else hidden,
                       "weight": w.ravel().tolist(), "bias": r.normal(scale=0.1, size=c).tolist()})
    doc = {"format": "bff.neural_net", "version": 1, "layers": layers}
    return b.NeuralNet(msgpack.packb(doc, use_bin_type=True))


def e4m3(c):
    c = c.astype(int)
    e, m = (c >> 3) & 0xF, c & 7
    v = np.where(e == 0, np.ldexp(m / 8.0, -6), np.ldexp(1.0 + m / 8.0, e - 7))
    return np.where(c & 0x80, -v, v)


def fp4_layers(q):
    """Per layer: nibble codes (n_out x padded), f32 scale per 16-sub-block
    (n_out x padded/16), bias, activation -- decoded from the document."""
    doc = msgpack.unpackb(bytes(q.to_msgpack()), raw=False)
    assert not doc["x_scaler"] and not doc["y_scaler"], "the prototype takes unscaled nets"
    fmt, out = doc["quantization"], []
    for L in doc["layers"]:
        n_in, n_out, P = L["n_in"], L["n_out"], pad32(L["n_in"])
        raw = np.frombuffer(L["codes"], np.uint8).reshape(n_out, P // 2)
        sc = np.frombuffer(L["scales"], np.uint8)
        if fmt == "fp4":
            s16 = np.repeat(np.frombuffer(L["scales"], "<f4")[:, None], P // 16, 1)
        elif fmt == "mxfp4":
            s16 = np.repeat(np.ldexp(1.0, sc.reshape(n_out, -1).astype(int) - 127), 2, 1)
        else:
            s16 = e4m3(sc.reshape(n_out, -1)) * L["tensor_scale"]
        out.append(dict(n_in=n_in, n_out=n_out, P=P, raw=raw, s16=s16.astype(np.float32),
                        bias=np.asarray(L["bias"], float), act=L["activation"]))
    return out


def dequant(L):
    b8 = L["raw"]
    codes = np.stack([b8 & 15, b8 >> 4], 2).reshape(L["n_out"], -1)
    return (E2M1[codes] * np.repeat(L["s16"].astype(float), 16, 1))[:, :L["n_in"]]


def forward_ref(layers, W, X):
    f = {"identity": lambda v: v, "tanh": np.tanh, "relu": lambda v: np.maximum(v, 0)}
    for L, w in zip(layers, W):
        X = f[L["act"]](X @ w.T + L["bias"])
    return X


class Gpu:
    def __init__(self):
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.dev = self.adapter.request_device_sync(
            required_limits={"max_storage_buffer_binding_size": self.adapter.limits["max-storage-buffer-binding-size"],
                             "max_buffer_size": self.adapter.limits["max-buffer-size"]})
        self.name = self.adapter.info["device"]

    def buf(self, data=None, size=None, usage=wgpu.BufferUsage.STORAGE):
        if data is not None:
            data = np.ascontiguousarray(data)
            return self.dev.create_buffer_with_data(data=data, usage=usage | wgpu.BufferUsage.COPY_SRC)
        return self.dev.create_buffer(size=max(16, size), usage=usage)

    def check_decoder(self, decode):
        out = self.buf(size=64, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        sm = self.dev.create_shader_module(code=decode + CHECK)
        pl = self.dev.create_compute_pipeline(layout="auto", compute={"module": sm, "entry_point": "main"})
        bg = self.dev.create_bind_group(layout=pl.get_bind_group_layout(0),
                                        entries=[{"binding": 0, "resource": {"buffer": out}}])
        enc = self.dev.create_command_encoder()
        cp = enc.begin_compute_pass(); cp.set_pipeline(pl); cp.set_bind_group(0, bg); cp.dispatch_workgroups(1); cp.end()
        self.dev.queue.submit([enc.finish()])
        return np.frombuffer(self.dev.queue.read_buffer(out), np.float32)


class Net:
    """One network resident on the GPU: weights, scales, biases, pipeline."""

    def __init__(self, gpu, layers, kind):
        self.g, self.L, self.kind = gpu, layers, kind
        self.br = int(kind.split("-t")[1]) if "-t" in kind else 0
        wb, sb, bb, self.offs = [], [], [], []
        wo = so = bo = 0
        for i, L in enumerate(layers):
            p_out = layers[i + 1]["P"] if i + 1 < len(layers) else pad32(L["n_out"])
            if self.br and kind.startswith("f32"):
                w = np.zeros((L["P"], p_out), np.float32)
                w[:L["n_in"], :L["n_out"]] = dequant(L).T
                wb.append(w.view(np.uint32).ravel()); sb.append(np.zeros(1, np.float32))
            elif self.br:
                words = np.zeros((L["P"] // 8, p_out), np.uint32)
                words[:, :L["n_out"]] = np.ascontiguousarray(L["raw"]).view("<u4").T
                sc = np.zeros((L["P"] // 16, p_out), np.float32)
                sc[:, :L["n_out"]] = L["s16"].T
                wb.append(words.ravel()); sb.append(sc.ravel())
            elif kind == "f32":
                w = dequant(L).astype(np.float32).T.copy()          # n_in x n_out
                wb.append(w.view(np.uint32).ravel()); sb.append(np.zeros(1, np.float32))
            else:
                words = np.ascontiguousarray(L["raw"]).view("<u4")   # n_out x P/8
                wb.append(words.T.ravel()); sb.append(L["s16"].T.ravel())
            bb.append(L["bias"].astype(np.float32))
            self.offs.append((wo, so, bo))
            wo += wb[-1].size; so += sb[-1].size; bo += bb[-1].size
        self.wgt = gpu.buf(np.concatenate(wb).astype(np.uint32))
        self.scl = gpu.buf(np.concatenate(sb).astype(np.float32))
        self.bia = gpu.buf(np.concatenate(bb))
        tiled = {"f32": LOAD_W_F32, "fp4": DECODE_ARITH + LOAD_W_FP4}
        code = COMMON + tiled[kind.split("-")[0]] + layer_tiled(self.br) if self.br else COMMON + {"f32": LAYER_F32, "bits": DECODE_BITS + LAYER_FP4, "lut": DECODE_LUT + LAYER_FP4,
                         "arith": DECODE_ARITH + LAYER_FP4}[kind]
        sm = gpu.dev.create_shader_module(code=code)
        self.pl = gpu.dev.create_compute_pipeline(layout="auto", compute={"module": sm, "entry_point": "layer"})
        self.rows = 0

    def _alloc(self, n):
        g, S, U = self.g, wgpu.BufferUsage.STORAGE, wgpu.BufferUsage
        self.n = n
        if self.br:
            n = (n + self.br - 1) // self.br * self.br
        width = max(max(L["P"] for L in self.L), pad32(self.L[-1]["n_out"]))
        self.a = [g.buf(size=4 * n * width, usage=S | U.COPY_DST | U.COPY_SRC) for _ in range(2)]
        self.groups, self.ubufs = [], []
        for i, L in enumerate(self.L):
            p_out = self.L[i + 1]["P"] if i + 1 < len(self.L) else pad32(L["n_out"])
            total = n * p_out
            nwg = min(65535, (total + 63) // 64)
            if self.br:
                rt = n // self.br
                nwg = (p_out // 32, min(rt, 65535), (rt + 65534) // 65535)
            wo, so, bo = self.offs[i]
            u = np.array([n, L["n_in"], L["n_out"], ACT[L["act"]], wo, so, bo, 0 if self.br else nwg * 64,
                          L["P"], p_out, 0, 0], np.uint32)
            ub = g.buf(u, usage=U.UNIFORM)
            src, dst = self.a[i % 2], self.a[(i + 1) % 2]
            bg = g.dev.create_bind_group(layout=self.pl.get_bind_group_layout(0), entries=[
                {"binding": 0, "resource": {"buffer": src}}, {"binding": 1, "resource": {"buffer": dst}},
                {"binding": 2, "resource": {"buffer": self.wgt}}, {"binding": 3, "resource": {"buffer": self.scl}},
                {"binding": 4, "resource": {"buffer": self.bia}}, {"binding": 5, "resource": {"buffer": ub}}])
            self.groups.append((bg, nwg)); self.ubufs.append(ub)
        self.out_buf, self.p_last, self.rows, self.n_pad = self.a[len(self.L) % 2], pad32(self.L[-1]["n_out"]), self.n, n

    def predict(self, X):
        n = X.shape[0]
        if n != self.rows:
            self._alloc(n)
        P0 = self.L[0]["P"]
        xp = np.zeros((self.n_pad, P0), np.float32); xp[:n, :X.shape[1]] = X
        q = self.g.dev.queue
        q.write_buffer(self.a[0], 0, xp)
        enc = self.g.dev.create_command_encoder()
        for bg, nwg in self.groups:
            cp = enc.begin_compute_pass(); cp.set_pipeline(self.pl); cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(*(nwg if isinstance(nwg, tuple) else (nwg,))); cp.end()
        q.submit([enc.finish()])
        y = np.frombuffer(q.read_buffer(self.out_buf, 0, 4 * n * self.p_last), np.float32)
        return y.reshape(n, self.p_last)[:, :self.L[-1]["n_out"]].astype(float)


def best(f, reps):
    f()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
    return 1e3 * min(t)


def main():
    g = Gpu()
    print(f"GPU: {g.name}   CPU fp4 kernel: {b.QuantizedNeuralNet(random_net([4, 8, 1]), 'fp4').get_kernel_name()}")
    print("\nDecoders over the 16 codes (want", E2M1.tolist(), ")")
    ok = {}
    for k, d in (("bits", DECODE_BITS), ("arith", DECODE_ARITH), ("lut", DECODE_LUT)):
        v = g.check_decoder(d)
        ok[k] = np.array_equal(v, E2M1.astype(np.float32))
        print(f"  {k:4s} {'exact' if ok[k] else 'WRONG'}  {v.tolist()}")

    kinds = ["f32", "arith", "f32-t32", "fp4-t32", "f32-t64", "fp4-t64"]
    rng = np.random.default_rng(1)
    for sizes in ([4, 64, 64, 2], [4, 128, 128, 128, 1], [32, 256, 256, 256, 8]):
        net = random_net(sizes)
        print(f"\nnet {'-'.join(map(str, sizes))}")
        for fmt in ("fp4", "mxfp4", "nvfp4"):
            q = b.QuantizedNeuralNet(net, fmt)
            layers = fp4_layers(q)
            W = [dequant(L) for L in layers]
            X = rng.normal(size=(5000, sizes[0]))
            ref = forward_ref(layers, W, X)
            cpu_q = np.asarray(q.predict(X.ravel().tolist(), len(X))).reshape(ref.shape)
            errs = {}
            for k in kinds:
                y = Net(g, layers, k).predict(X)
                errs[k] = np.abs(y - ref).max() / np.abs(ref).max()
            errs["cpu W4A8"] = np.abs(cpu_q - ref).max() / np.abs(ref).max()
            print(f"  {fmt:5s} rel. error vs f64 on the decoded weights: " +
                  "  ".join(f"{k} {v:.1e}" for k, v in errs.items()))
        q = b.QuantizedNeuralNet(net, "nvfp4")
        layers = fp4_layers(q)
        nets = {k: Net(g, layers, k) for k in kinds}
        hdr = "".join(f"{k + ' GPU':>11s}" for k in kinds)
        print(f"  {'rows':>7s}{'MACs':>8s}{hdr}{'f64 CPU':>10s}{'fp4 CPU':>10s}   ms, nvfp4, weights resident")
        macs1 = sum(a * c for a, c in zip(sizes[:-1], sizes[1:]))
        for n in (1000, 10000, 100000, 400000):
            X = rng.normal(size=(n, sizes[0]))
            xl = X.ravel().tolist()
            reps = 3 if n >= 100000 else 10
            gt = [best(lambda: nets[k].predict(X), reps) for k in kinds]
            tc = best(lambda: net.predict(xl, n), max(2, reps // 3))
            tq = best(lambda: q.predict(xl, n), max(2, reps // 3))
            print(f"  {n:7d}{macs1 * n / 1e6:7.0f}M" + "".join(f"{t:11.2f}" for t in gt) + f"{tc:10.2f}{tq:10.2f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
