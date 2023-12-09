struct Meta {
    M: u32,
    N: u32,
}

@group(0) @binding(0)
var<storage, read_write> input: array<f32>;

@group(0) @binding(1)
var<storage, read> input_m: Meta;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let mi = workgroup_id.x * 32u + local_id.x;
    if (mi >= input_m.M) {
        return;
    }

    // v = v / (1 + (-v).exp())
    for (var ni = 0u; ni < input_m.N; ni = ni + 1u) {
        let i = mi * input_m.N + ni;
        input[i] = input[i] / (1.0f + exp(-input[i]));
    }
}