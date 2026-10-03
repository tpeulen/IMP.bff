// One dense layer, y = activation(W x + b), tiled through workgroup memory.
//
// A workgroup of 256 threads owns 64 rows x 64 outputs and walks the
// contraction 32 at a time. The input tile and the weight tile go into
// workgroup memory, and each thread keeps a register block of 4 rows x 4
// outputs (rows tr + 16 i, outputs to + 16 j). A weight read from the buffer
// is reused by 64 rows and an input by 64 outputs. The shape is measured
// (benchmark/gpu_mlp_tile_tuning.py, okf/validation/fp4_nn_on_the_gpu.md):
// against 64 threads on 32 x 32 it is 1.4-1.7x, while larger register
// blocks (8 x 8) and vec4 reads were slower on an M1 Pro.
//
// The input tile is stored transposed, k-major, so the two tiles fill
// 16 384 bytes -- WebGPU's default workgroup-storage limit -- exactly; the
// row-major tile needs a pad column against bank conflicts and does not fit.
//
// The weights are f32 or FP4, per layer (`p.fmt`), and only the tile load
// differs. FP4 codes are decoded and scaled on the way into workgroup memory
// -- once per workgroup rather than once per row -- so FP4 runs as fast as
// f32 while holding the weights in an eighth of the space.
//
// Every width is padded to a multiple of 64 (the batch's own to 32) and every
// row count to a multiple of 64, with zeros, so nothing inside the loop is
// bounds-checked: padded
// weights are zero, and padded outputs are written as zero for the next layer
// to read. The two ends are not padded -- the batch arrives `n_in` wide and
// the answer leaves `n_out` wide -- because for the narrow networks this is
// for (a few inputs, a few outputs, many rows) the padding would be most of
// what crosses the bus. The input tile load pads them with zeros instead.
//
// One dispatch per layer, ping-ponging two activation buffers, so a forward
// pass crosses the plugin boundary once however deep the network is.

struct Params {
    n_rows: u32,   // padded to a multiple of 64
    n_in: u32,
    n_out: u32,
    act: u32,
    w_off: u32,    // in u32 words
    s_off: u32,    // in f32, FP4 only
    b_off: u32,
    fmt: u32,      // 0: f32 weights, 1: FP4 (E2M1) codes
    p_in: u32,     // n_in padded: to 32 for the batch, to 64 after a layer
    p_out: u32,    // n_out padded to 64
    x_stride: u32, // the input row stride: p_in, or n_in for the batch itself
    y_stride: u32, // the output row stride: p_out, or n_out for the answer
};

@group(0) @binding(0) var<storage, read>       xin:  array<f32>;
@group(0) @binding(1) var<storage, read_write> xout: array<f32>;
// f32: the bits of W^T, p_in x p_out. FP4: eight codes a word, word k / 8 of
// output o at (k / 8) * p_out + o, element 8 (k / 8) + j at bits 4 j -- the
// row's bytes, low nibble first, read as a little-endian u32 and transposed,
// so that neighbouring outputs read neighbouring words.
@group(0) @binding(2) var<storage, read>       wgt:  array<u32>;
// FP4: the decoded scale of each 16-element sub-block, (k / 16) * p_out + o.
@group(0) @binding(3) var<storage, read>       scl:  array<f32>;
@group(0) @binding(4) var<storage, read>       bia:  array<f32>;
@group(0) @binding(5) var<uniform>             p:    Params;

var<workgroup> xs: array<f32, 2048>;   // 32 k x 64 rows
var<workgroup> ws: array<f32, 2048>;   // 32 k x 64 outputs

// The codes are IMP::bff::internal::Activation in declaration order.
fn activate(v: f32, act: u32) -> f32 {
    if (act == 0u) { return v; }                                  // identity
    if (act == 1u) { return max(v, 0.0); }                        // relu
    if (act == 2u) { return tanh(v); }                            // tanh
    if (act == 3u) { return 1.0 / (1.0 + exp(-v)); }              // logistic
    if (act == 4u) {
        // log1p(exp(v)), written so that a large v does not overflow before
        // the logarithm brings it back.
        if (v > 20.0) { return v; }
        return log(1.0 + exp(v));                                 // softplus
    }
    if (act == 5u) { return v / (1.0 + exp(-v)); }                // silu
    if (act == 6u) { return sin(v); }                             // sin
    return v;
}

// An E2M1 code (s.ee.m) as f32, exactly. Four times the magnitude is the
// integer (2 + m) << e for a normal code and 2 m for e == 0. The shorter bit
// trick -- the code shifted into f32's exponent and mantissa, times 2^126 --
// is not used: it makes 0.5 an f32 subnormal, which WGSL may flush, and
// Metal does (0.5 decoded as 0).
fn e2m1(c: u32) -> f32 {
    let e = (c >> 1u) & 3u;
    let m = c & 1u;
    let v = f32(select((2u + m) << e, m << 1u, e == 0u)) * 0.25;
    return select(v, -v, (c & 8u) != 0u);
}

fn load_weights(lid: u32, k0: u32, o0: u32) {
    if (p.fmt == 0u) {
        for (var e = lid; e < 2048u; e = e + 256u) {
            ws[e] = bitcast<f32>(wgt[p.w_off + (k0 + e / 64u) * p.p_out + o0 + e % 64u]);
        }
        return;
    }
    // 4 words a column of the tile, 64 columns: one word a thread.
    let k8l = lid / 64u;
    let oo = lid % 64u;
    let k8 = k0 / 8u + k8l;
    let w = wgt[p.w_off + k8 * p.p_out + o0 + oo];
    let s = scl[p.s_off + (k8 / 2u) * p.p_out + o0 + oo];
    for (var j = 0u; j < 8u; j = j + 1u) {
        ws[(k8l * 8u + j) * 64u + oo] = e2m1((w >> (4u * j)) & 15u) * s;
    }
}

// The output tile is wid.x; the row tile is wid.y + 65535 wid.z, since a
// dispatch may not exceed 65 535 workgroups in a dimension (going over is a
// validation error that aborts the process rather than returning one).
@compute @workgroup_size(256)
fn layer(@builtin(local_invocation_index) lid: u32,
         @builtin(workgroup_id) wid: vec3<u32>) {
    let rt = wid.y + wid.z * 65535u;
    if (rt * 64u >= p.n_rows) { return; }   // the same for the whole workgroup
    let r0 = rt * 64u;
    let o0 = wid.x * 64u;
    let tr = lid / 16u;
    let to = lid % 16u;
    var acc: array<f32, 16>;
    for (var k0 = 0u; k0 < p.p_in; k0 = k0 + 32u) {
        // Read row-major (neighbouring threads, neighbouring addresses),
        // stored k-major.
        for (var e = lid; e < 2048u; e = e + 256u) {
            let k = k0 + e % 32u;
            var v = 0.0;
            if (k < p.n_in) { v = xin[(r0 + e / 32u) * p.x_stride + k]; }
            xs[(e % 32u) * 64u + e / 32u] = v;
        }
        load_weights(lid, k0, o0);
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk = kk + 1u) {
            let b0 = ws[kk * 64u + to];
            let b1 = ws[kk * 64u + to + 16u];
            let b2 = ws[kk * 64u + to + 32u];
            let b3 = ws[kk * 64u + to + 48u];
            let a0 = xs[kk * 64u + tr];
            let a1 = xs[kk * 64u + tr + 16u];
            let a2 = xs[kk * 64u + tr + 32u];
            let a3 = xs[kk * 64u + tr + 48u];
            acc[0] = acc[0] + a0 * b0;   acc[1] = acc[1] + a0 * b1;
            acc[2] = acc[2] + a0 * b2;   acc[3] = acc[3] + a0 * b3;
            acc[4] = acc[4] + a1 * b0;   acc[5] = acc[5] + a1 * b1;
            acc[6] = acc[6] + a1 * b2;   acc[7] = acc[7] + a1 * b3;
            acc[8] = acc[8] + a2 * b0;   acc[9] = acc[9] + a2 * b1;
            acc[10] = acc[10] + a2 * b2; acc[11] = acc[11] + a2 * b3;
            acc[12] = acc[12] + a3 * b0; acc[13] = acc[13] + a3 * b1;
            acc[14] = acc[14] + a3 * b2; acc[15] = acc[15] + a3 * b3;
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let r = r0 + tr + 16u * i;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let o = o0 + to + 16u * j;
            if (o < p.n_out) {
                xout[r * p.y_stride + o] = activate(acc[4u * i + j] + bia[p.b_off + o], p.act);
            } else if (p.y_stride == p.p_out) {
                xout[r * p.y_stride + o] = 0.0;   // the next layer's padding
            }
        }
    }
}
