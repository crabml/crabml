struct Meta {
    B: u32,
    M: u32,
    K: u32,
    _padding: u32,
};

struct BlockQ8_0 {
    d: u16,
    qs: array<i8, 32>,
};

@group(0) @binding(0)
var<storage, read> bufA: array<BlockQ8_0>;

@group(0) @binding(1)
var<storage, read> bufB: array<BlockQ8_0>;

@group(0) @binding(2)
var<storage, read_write> bufC: array<f32>;

@group(0) @binding(3)
var<storage, read> md: Meta;

// (B, M, K) * (B, K, 1) = (B, M, 1)
// split the work into (B, M)
// each thread is responsible for one element in the output

fn f16_bits_to_f32(n: u16) -> f32 {}
fn f32_to_f16_bits(v: f32) -> u16 {}

@compute @workgroup_size(1, 1, 32)
fn main(
    @builtin(global_invocation_id) gIdx: vec3<u32>,
) {
    let B = md.B;
    let M = md.M;
    let K = md.K;
    let bi = gIdx.x;
    let mi = gIdx.y * 4u % M;

    // A: (M, K)
    // B: (B, K)
    // C: (B, M)

    for (var ki = 0u; ki < K; ki += 32u) {
        var sum = 0u;
        let blkA = bufA[(bi * M * K + mi * K + ki) / 32u];
        let blkB = bufB[(bi * K + ki) / 32u];
        for (var j = 0u; j < 32u; j = j + 1u) {
            sum += u32(blkA.qs[j]) * u32(blkB.qs[j]);
        }
        let dA = f16_bits_to_f32(blkA.d);
        let dB = f16_bits_to_f32(blkB.d);
        C[bi * M + mi] += f32(sum) * dA * dB;
    }
}