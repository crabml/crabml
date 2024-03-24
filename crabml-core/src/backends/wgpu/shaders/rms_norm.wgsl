struct Meta {
    nBatch: u32, // number of vectors
    nDims: u32, // length of each vector
    eps: f32,
    _padding: f32,
};

@group(0) @binding(0)
var<storage, read_write> buf: array<f32>;

@group(0) @binding(1)
var<storage, read> bufM: Meta;

// workgroup local to reduce squared sum
var<workgroup> threadSums: array<f32, 64>;

// each workgroup normalize a single vector

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroupID: vec3<u32>,
    @builtin(local_invocation_id) localID: vec3<u32>,
) {
    let nDims = bufM.nDims;
    let eps = bufM.eps;

    let workgroupSize: u32 = 32u;
    let localChunkSize = nDims / workgroupSize;

    // calculate each thread's chunk of the squared sum
    for (var i = 0u; i < localChunkSize; i += 1u) {
        let idx = nDims * workgroupID.x + localID.x * localChunkSize + i;
        threadSums[localID.x] += buf[idx] * buf[idx];
    }
    workgroupBarrier();

    // reduce squared sum
    if localID.x == 0u {
        for (var i = 1u; i < workgroupSize; i += 1u) {
            threadSums[0] += threadSums[i];
        }
    }
    workgroupBarrier();

    // normalize to output
    for (var i = 0u; i < localChunkSize; i += 1u) {
        let idx = nDims * workgroupID.x + localID.x * localChunkSize + i;
        let scale = 1.0 / sqrt((threadSums[0] / f32(nDims)) + eps);
        buf[idx] *= scale;
    }
}