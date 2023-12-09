// (m, n, k) * (m, k) = (m, n)
struct Meta {
    M: u32,
    N: u32,
    K: u32,
    _padding_0: u32,
    strides_0: vec3<u32>,
    _padding_1: u32,
    repeats_0: vec3<u32>,
    _padding_2: u32,
};

@group(0) @binding(0)
var<storage, read> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read> input_1: array<f32>;

@group(0) @binding(2)
var<storage, read> input_m: Meta;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let m = workgroup_id.x * 32u + local_id.x;
    if (m >= input_m.M) {
        return;
    }
    for (var n = 0u; n < input_m.N; n = n + 1u) {
        var sum = 0.0f;
        for (var k = 0u; k < input_m.K; k = k + 1u) {
            let a = input_0[input_m.N * input_m.K * m + input_m.K * n + k];
            let b = input_1[input_m.K * m + k];
            sum += a * b;
        }
        output[m * input_m.N + n] += sum;
    }

    // passed wrong value in strides
    output[0] = f32(input_m.repeats_0.x);
    output[1] = f32(input_m.repeats_0.y);
    output[2] = f32(input_m.repeats_0.z);
}