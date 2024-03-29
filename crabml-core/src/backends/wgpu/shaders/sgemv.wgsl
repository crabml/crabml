struct Meta {
    B: u32,
    M: u32,
    K: u32,
    _padding: u32,
};

@group(0) @binding(0)
var<storage, read> bufA: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> bufB: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> md: Meta;

@group(0) @binding(3)
var<storage, read_write> bufC: array<vec4<f32>>;

// (M, K) * (K, 1) = (M, 1)
// split the work by M / 32

@compute @workgroup_size(1, 8, 1)
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

    var tmp = vec4<f32>();
    for (var ki = 0u; ki < K; ki += 4u) {
        let bc = bufB[(bi * K + ki) / 4u];
        let x = dot(bufA[mi * K / 4u + ki / 4u], bc);
        let y = dot(bufA[(mi + 1u) * K / 4u + ki / 4u], bc);
        let z = dot(bufA[(mi + 2u) * K / 4u + ki / 4u], bc);
        let w = dot(bufA[(mi + 3u) * K / 4u + ki / 4u], bc);
        tmp += vec4<f32>(x, y, z, w);
    }
    bufC[(bi * M + mi) / 4u] = tmp;
}