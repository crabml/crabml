struct Meta {
    M: u32,
    N: u32,
    K: u32,
    _padding: u32,
};

@group(0) @binding(0)
var<storage, read> input_w: array<f32>;

@group(0) @binding(1)
var<storage, read> input_x: array<f32>;

@group(0) @binding(2)
var<storage, read> input_m: Meta;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

// (M, N) * (N, 1) = (M, 1)
// split the work by M / 32
@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let m = workgroup_id.x * 32u + local_id.x;
    for (var n = 0u; n < input_m.N; n++) {
        output[m] += input_w[m * input_m.N + n] * input_x[n];
    }
}