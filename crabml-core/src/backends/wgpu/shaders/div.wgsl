struct Meta {
    M: u32, // number of vectors
    N: u32, // length of each vector
}

@group(0) @binding(0)
var<storage, read_write> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read> input_1: array<f32>;

@group(0) @binding(2)
var<storage, read> input_m: Meta;

// each workgroup will process a single vector

@compute
@workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let workgroup_size = 32u;
    let mi = 32u * workgroup_id.x + local_id.x;
    if mi >= input_m.M {
        return;
    }

    for (var ni = 0u; ni < input_m.N; ni = ni + 1u) {
        let idx = mi * input_m.N + ni;
        input_0[idx] /= input_1[idx % arrayLength(&input_1)];
    }
}