struct Meta {
    B: u32, // number of vectors
    M: u32, // length of each vector
    eps: f32,
    _padding: f32,
};

@group(0) @binding(0)
var<storage, read_write> buf: array<f32>;

@group(0) @binding(1)
var<storage, read> bufM: Meta;

// workgroup local to reduce squared sum
var<workgroup> thread_sums: array<f32, 64>;

// each workgroup normalize a single vector

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let workgroup_size: u32 = 32u;
    let local_chunk_size = bufM.M / workgroup_size;

    // calculate each thread's chunk of the squared sum
    for (var i = 0u; i < local_chunk_size; i += 1u) {
        let idx = bufM.M * workgroup_id.x + local_id.x * local_chunk_size + i;
        thread_sums[local_id.x] += buf[idx] * buf[idx];
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
        let idx = bufM.M * workgroup_id.x + local_id.x * local_chunk_size + i;
        let scale = 1.0 / sqrt((thread_sums[0] / f32(bufM.M)) + bufM.eps);
        buf[idx] *= scale;
    }
}