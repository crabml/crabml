struct Meta {
    M: u32,
    N: u32,
}

@group(0) @binding(0)
var<storage, read_write> input: array<f32>;

@group(0) @binding(1)
var<storage, read> input_m: Meta;

@compute @workgroup_size(16)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let mi = workgroup_id.x * 16u + local_id.x;
    if (mi >= input_m.M) {
        return;
    }

    var max = 0.0f;
    for (var ni = 0u; ni < input_m.N; ni = ni + 1u) {
        let idx = mi * input_m.N + ni;
        if (input[idx] > max) {
            max = input[idx];
        }
    }

    var sum = 0.0f;
    for (var ni = 0u; ni < input_m.N; ni = ni + 1u) {
        let idx = mi * input_m.N + ni;
        input[idx] = exp(input[idx] - max);
        sum += input[idx];
    }

    for (var ni = 0u; ni < input_m.N; ni = ni + 1u) {
        let idx = mi * input_m.N + ni;
        input[idx] = input[idx] / sum;
    }
}