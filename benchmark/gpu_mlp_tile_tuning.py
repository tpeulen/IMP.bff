"""Tile shapes for gpu/mlp.wgsl: rows x outputs a workgroup, threads, and
the register block a thread keeps.

The shipped kernel is 32 rows x 32 outputs, 64 threads, 4 x 4 a thread, and
reaches about 180 GFLOP/s on an M1 Pro against an f32 peak near 5 TFLOP/s.
This sweeps the shapes around it for f32 and FP4 weights, with the same
layouts as the plugin (W^T or word-interleaved codes, the batch in and out
unpadded), and checks each against the dequantised network in double.

    python benchmark/gpu_mlp_tile_tuning.py
"""
import itertools, sys, time
import numpy as np, wgpu
import IMP.bff as b

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gpu_mlp_fp4_wgsl import (ACT, COMMON, DECODE_ARITH, Gpu, dequant, forward_ref,
                              fp4_layers, random_net)

COMMON_T = COMMON.replace("p_in: u32, p_out: u32, pad0: u32, pad1: u32",
                          "p_in: u32, p_out: u32, x_stride: u32, y_stride: u32")


def shader(BR, BO, TR, TO, fp4, vec, KC=32, xt=False):
    NT, RM, OM = TR * TO, BR // TR, BO // TO
    assert BR % TR == 0 and BO % TO == 0 and NT <= 256
    if fp4:
        load_w = f"""
fn load_w(lid: u32, k0: u32, o0: u32) {{
    for (var e = lid; e < {KC // 8 * BO}u; e = e + {NT}u) {{
        let k8l = e / {BO}u;
        let oo = e % {BO}u;
        let k8 = k0 / 8u + k8l;
        let w = wgt[p.w_off + k8 * p.p_out + o0 + oo];
        let s = scl[p.s_off + (k8 / 2u) * p.p_out + o0 + oo];
        for (var j = 0u; j < 8u; j = j + 1u) {{
            ws[(k8l * 8u + j) * {BO}u + oo] = dec((w >> (4u * j)) & 15u) * s;
        }}
    }}
}}"""
    else:
        load_w = f"""
fn load_w(lid: u32, k0: u32, o0: u32) {{
    for (var e = lid; e < {KC * BO}u; e = e + {NT}u) {{
        ws[e] = bitcast<f32>(wgt[p.w_off + (k0 + e / {BO}u) * p.p_out + o0 + e % {BO}u]);
    }}
}}"""
    # the register block: outputs to + TO j are strided, so the ws reads of
    # neighbouring threads are neighbouring; with `vec`, a thread's outputs
    # are 4 consecutive ones per vec4 (to * 4 + 4 TO j + q) and read as one.
    lines = []
    if vec:
        nv = OM // 4
        for j in range(nv):
            lines.append(f"            let b{j} = wv[kk * {BO // 4}u + to + {TO}u * {j}u];")
        for i in range(RM):
            lines.append(f"            {{ let a = xs[(tr + {TR * i}u) * 33u + kk];")
            for j in range(nv):
                lines.append(f"              acc{i}_{j} = acc{i}_{j} + a * b{j};")
            lines.append("            }")
        decl = "\n".join(f"    var acc{i}_{j} = vec4<f32>(0.0);" for i in range(RM) for j in range(nv))
        store = []
        for i in range(RM):
            for j in range(nv):
                for q in range(4):
                    store.append(f"    put(r0 + tr + {TR * i}u, o0 + (to + {TO}u * {j}u) * 4u + {q}u, acc{i}_{j}[{q}]);")
        wdecl = f"var<workgroup> ws: array<f32, {32 * BO}>;\n"
        inner_pre = ""
        # ws viewed as vec4: a separate array filled after the f32 load
        wdecl += f"var<workgroup> wv: array<vec4<f32>, {8 * BO}>;\n"
        inner_pre = f"""
        for (var e = lid; e < {8 * BO}u; e = e + {NT}u) {{
            wv[e] = vec4<f32>(ws[4u * e], ws[4u * e + 1u], ws[4u * e + 2u], ws[4u * e + 3u]);
        }}
        workgroupBarrier();"""
    else:
        for j in range(OM):
            lines.append(f"            let b{j} = ws[kk * {BO}u + to + {TO * j}u];")
        for i in range(RM):
            lines.append(f"            {{ let a = xs[kk * {BR}u + tr + {TR * i}u];" if xt else f"            {{ let a = xs[(tr + {TR * i}u) * {KC + 1}u + kk];")
            lines.append("              " + " ".join(f"acc{i}_{j} = acc{i}_{j} + a * b{j};" for j in range(OM)))
            lines.append("            }")
        decl = "\n".join(f"    var acc{i}_{j} = 0.0;" for i in range(RM) for j in range(OM))
        store = [f"    put(r0 + tr + {TR * i}u, o0 + to + {TO * j}u, acc{i}_{j});"
                 for i in range(RM) for j in range(OM)]
        wdecl = f"var<workgroup> ws: array<f32, {KC * BO}>;\n"
        inner_pre = ""
    inner = "\n".join(lines)
    return COMMON_T + (DECODE_ARITH if fp4 else "") + f"""
var<workgroup> xs: array<f32, {BR * KC if xt else BR * (KC + 1)}>;
{wdecl}
{load_w}

fn put(r: u32, o: u32, v: f32) {{
    if (o < p.n_out) {{
        xout[r * p.y_stride + o] = activate(v + bia[p.b_off + o], p.act);
    }} else if (p.y_stride == p.p_out) {{
        xout[r * p.y_stride + o] = 0.0;
    }}
}}

@compute @workgroup_size({NT})
fn layer(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let rt = wid.y + wid.z * 65535u;
    if (rt * {BR}u >= p.n_rows) {{ return; }}
    _ = scl[0];
    let r0 = rt * {BR}u;
    let o0 = wid.x * {BO}u;
    let tr = lid / {TO}u;
    let to = lid % {TO}u;
{decl}
    for (var k0 = 0u; k0 < p.p_in; k0 = k0 + {KC}u) {{
        for (var e = lid; e < {BR * KC}u; e = e + {NT}u) {{
            let k = k0 + e % {KC}u;
            var v = 0.0;
            if (k < p.n_in) {{ v = xin[(r0 + e / {KC}u) * p.x_stride + k]; }}
            {f"xs[(e % {KC}u) * {BR}u + e / {KC}u] = v;" if xt else f"xs[(e / {KC}u) * {KC + 1}u + e % {KC}u] = v;"}
        }}
        load_w(lid, k0, o0);
        workgroupBarrier();{inner_pre}
        for (var kk = 0u; kk < {KC}u; kk = kk + 1u) {{
{inner}
        }}
        workgroupBarrier();
    }}
{chr(10).join(store)}
}}
"""


def padto(n, m):
    return (n + m - 1) // m * m


class TiledNet:
    def __init__(self, g, layers, fp4, BR, BO, TR, TO, vec=False, KC=32, xt=False):
        self.g, self.L, self.BR, self.BO = g, layers, BR, BO
        m = max(BO, KC, 32)
        self.pin = [padto(layers[0]["n_in"], max(KC, 32))] + [padto(L["n_out"], m) for L in layers[:-1]]
        self.pout = [padto(L["n_out"], m) for L in layers]
        wb, sb, bb, self.offs = [], [], [], []
        wo = so = bo = 0
        for i, L in enumerate(layers):
            P, Q = self.pin[i], self.pout[i]
            if fp4:
                words = np.zeros((P // 8, Q), np.uint32)
                raw = np.zeros((L["n_out"], P // 2), np.uint8)
                raw[:, :L["raw"].shape[1]] = L["raw"]
                words[:, :L["n_out"]] = np.ascontiguousarray(raw).view("<u4").T
                sc = np.zeros((P // 16, Q), np.float32)
                sc[:L["s16"].shape[1], :L["n_out"]] = L["s16"].T
                wb.append(words.ravel()); sb.append(sc.ravel())
            else:
                w = np.zeros((P, Q), np.float32)
                w[:L["n_in"], :L["n_out"]] = dequant(L).T
                wb.append(w.view(np.uint32).ravel()); sb.append(np.zeros(1, np.float32))
            bb.append(L["bias"].astype(np.float32))
            self.offs.append((wo, so, bo))
            wo += wb[-1].size; so += sb[-1].size; bo += bb[-1].size
        self.wgt = g.buf(np.concatenate(wb).astype(np.uint32))
        self.scl = g.buf(np.concatenate(sb).astype(np.float32))
        self.bia = g.buf(np.concatenate(bb))
        sm = g.dev.create_shader_module(code=shader(BR, BO, TR, TO, fp4, vec, KC, xt))
        self.pl = g.dev.create_compute_pipeline(layout="auto", compute={"module": sm, "entry_point": "layer"})
        self.rows = -1

    def _alloc(self, n):
        g, U = self.g, wgpu.BufferUsage
        npad = padto(n, self.BR)
        width = max(max(self.pin), max(self.pout))
        self.a = [g.buf(size=4 * npad * width, usage=U.STORAGE | U.COPY_DST | U.COPY_SRC) for _ in range(2)]
        self.groups, self.ub = [], []
        last = len(self.L) - 1
        for i, L in enumerate(self.L):
            wo, so, bo = self.offs[i]
            u = np.array([npad, L["n_in"], L["n_out"], ACT[L["act"]], wo, so, bo, 0,
                          self.pin[i], self.pout[i],
                          L["n_in"] if i == 0 else self.pin[i],
                          L["n_out"] if i == last else self.pout[i]], np.uint32)
            ub = g.buf(u, usage=U.UNIFORM)
            bg = g.dev.create_bind_group(layout=self.pl.get_bind_group_layout(0), entries=[
                {"binding": 0, "resource": {"buffer": self.a[i % 2]}},
                {"binding": 1, "resource": {"buffer": self.a[(i + 1) % 2]}},
                {"binding": 2, "resource": {"buffer": self.wgt}}, {"binding": 3, "resource": {"buffer": self.scl}},
                {"binding": 4, "resource": {"buffer": self.bia}}, {"binding": 5, "resource": {"buffer": ub}}])
            rt = npad // self.BR
            self.groups.append((bg, (self.pout[i] // self.BO, min(rt, 65535), (rt + 65534) // 65535)))
            self.ub.append(ub)
        self.out, self.rows = self.a[len(self.L) % 2], n

    def predict(self, X, reps=1):
        n = X.shape[0]
        if n != self.rows:
            self._alloc(n)
        q = self.g.dev.queue
        q.write_buffer(self.a[0], 0, np.ascontiguousarray(X, np.float32))
        enc = self.g.dev.create_command_encoder()
        for _ in range(reps):
            for bg, d in self.groups:
                cp = enc.begin_compute_pass(); cp.set_pipeline(self.pl); cp.set_bind_group(0, bg)
                cp.dispatch_workgroups(*d); cp.end()
        q.submit([enc.finish()])
        no = self.L[-1]["n_out"]
        return np.frombuffer(q.read_buffer(self.out, 0, 4 * n * no), np.float32).reshape(n, no).astype(float)


def main():
    g = Gpu()
    print("max workgroup storage on this device:", g.dev.limits["max-compute-workgroup-storage-size"])
    print("GPU:", g.name)
    shapes = [  # BR, BO, TR, TO, vec, KC, xs transposed
        (32, 32, 8, 8, False, 32, False),
        (64, 64, 16, 16, False, 32, False),
        (64, 64, 16, 16, False, 32, True),
        (64, 64, 16, 16, False, 16, True),
    ]
    rng = np.random.default_rng(0)
    for sizes, n in (([32, 256, 256, 256, 8], 200000), ([4, 128, 128, 128, 1], 400000), ([4, 64, 64, 2], 400000)):
        net = random_net(sizes)
        layers = fp4_layers(b.QuantizedNeuralNet(net, "nvfp4"))
        W = [dequant(L) for L in layers]
        flops = 2 * n * sum(a * c for a, c in zip(sizes[:-1], sizes[1:]))
        X = rng.normal(size=(n, sizes[0]))
        Xs = X[:3000]
        ref = forward_ref(layers, W, Xs)
        print(f"\nnet {'-'.join(map(str, sizes))}, {n} rows; GFLOP/s of the kernels (10 passes in one submission)")
        for sh in shapes:
            row = []
            for fp4 in (False, True):
                try:
                    t = TiledNet(g, layers, fp4, *sh)
                except Exception as e:
                    row.append(f"{'err':>8s}"); print("  ", sh, fp4, str(e).splitlines()[0][:100]); continue
                err = np.abs(t.predict(Xs) - ref).max() / np.abs(ref).max()
                t.predict(X, 1)
                t0 = time.perf_counter(); t.predict(X, 1); t1 = time.perf_counter() - t0
                t0 = time.perf_counter(); t.predict(X, 11); t11 = time.perf_counter() - t0
                gf = 10 * flops / (t11 - t1) / 1e9
                row.append(f"{gf:8.0f}{'' if err < 1e-5 else ' BAD %.0e' % err}")
            print(f"  {'xT' if sh[6] else '  '} BR {sh[0]:3d} BO {sh[1]:3d} KC {sh[5]:2d} threads {sh[2] * sh[3]:3d} ({sh[2]}x{sh[3]}) "
                  f"block {sh[0] // sh[2]}x{sh[1] // sh[3]}{' vec4' if sh[4] else '     '}  f32 {row[0]}  fp4 {row[1]}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
