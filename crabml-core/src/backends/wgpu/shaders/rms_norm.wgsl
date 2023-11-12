struct Meta {
    M: u32, // number of vectors
    N: u32, // length of each vector
    eps: f32,
};

@group(0) @binding(0)
var<storage, read_write> input: array<f32>;

@group(0) @binding(1)
var<storage, read> input_m: Meta;

// workgroup local to reduce squared sum
var<workgroup> thread_sums: array<f32, 64>;

// each workgroup normalize a single vector

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let workgroup_size: u32 = 64u;
    let local_chunk_size = input_m.N / workgroup_size;

    // calculate each thread's chunk of the squared sum
    for (var i = 0u; i < local_chunk_size; i += 1u) {
        let idx = input_m.N * workgroup_id.x + local_id.x * local_chunk_size + i;
        thread_sums[local_id.x] += input[idx] * input[idx];
    }
    workgroupBarrier();

    // reduce squared sum
    if local_id.x == 0u {
        for (var i = 1u; i < workgroup_size; i += 1u) {
            thread_sums[0] += thread_sums[i];
        }
    }
    workgroupBarrier();

    // normalize to output
    for (var i = 0u; i < local_chunk_size; i += 1u) {
        let idx = input_m.N * workgroup_id.x + local_id.x * local_chunk_size + i;
        input[idx] = input[idx] / sqrt(thread_sums[0] + input_m.eps);
    }
}