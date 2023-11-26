struct Meta {
    M: u32, // number of vectors
    N: u32, // length of each vector
}

@group(0) @binding(0)
var<storage, read_write> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read> input_1: array<f32>;

@group(0) @binding(2)
var<storage, read> input_meta: Meta;

// each workgroup will process a single vector

@compute
@workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let workgroup_size = 32u;
    let local_chunk_size = input_meta.N / workgroup_size;

    for (var i = 0u; i < local_chunk_size; i = i + 1u) {
        let idx = workgroup_id.x * input_meta.N + local_id.x * local_chunk_size + i;
        input_0[idx] *= input_1[idx % arrayLength(&input_1)];
    }
}