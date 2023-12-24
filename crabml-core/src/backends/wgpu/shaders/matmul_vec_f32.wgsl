struct Meta {
    M: u32,
    K: u32,
    N: u32,
    _padding: u32,
};

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read> md: Meta;

@group(0) @binding(3)
var<storage, read_write> C: array<f32>;

// (M, K) * (K, 1) = (M, 1)
// split the work by M / 32
@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let M = md.M;
    let N = md.N;
    let K = md.K;
    let m = global_id.x;
    for (var k = 0u; k < K; k++) {
        C[m] += A[m * K + k] * B[k];
    }
}