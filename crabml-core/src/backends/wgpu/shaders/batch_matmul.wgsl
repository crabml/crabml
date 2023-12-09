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
    let mi = workgroup_id.x * 32u + local_id.x;
    if (mi >= input_m.M) {
        return;
    }
    for (var ni = 0u; ni < input_m.N; ni = ni + 1u) {
        var sum = 0.0f;
        for (var ki = 0u; ki < input_m.K; ki = ki + 1u) {
            let a = input_0[
                mi / input_m.repeats_0.x * input_m.strides_0.x +
                ni / input_m.repeats_0.y * input_m.strides_0.y +
                ki / input_m.repeats_0.z * input_m.strides_0.z
            ];
            let b = input_1[input_m.K * mi + ki];
            sum += a * b;
        }
        output[mi * input_m.N + ni] = sum;
    } 
}