struct Meta {
    nBatch: u32, // number of vectors
    nDims: u32, // length of vector
    pos: u32,
    nHeads: u32,
    nRopeDims: u32,
    _padding: vec3<u32>,
};

@group(0) @binding(0)
var<storage, read_write> input: array<f32>;

@group(0) @binding(1)
var<storage, read> bufM: Meta;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroupID: vec3<u32>,
    @builtin(local_invocation_id) localID: vec3<u32>,
) {
    let nHeadDims = bufM.nDims / bufM.nHeads;
    let gidx = workgroupID.x * 32u + localID.x;

    // process each vector in one thread. if there's only one vector, only gidx == 0 makes sense
    if gidx >= bufM.nBatch {
        return;
    }

    for (var h = 0u; h < bufM.nHeads; h++) {
        for (var i = 0u; i < bufM.nRopeDims / 2u; i++) {
            let thetaScale = pow(10000.0, -2.0 * f32(i) / f32(nHeadDims));
            let theta = f32(bufM.pos) * thetaScale;

            let cosTheta = cos(theta);
            let sinTheta = sin(theta);
            let qpOffset = gidx * bufM.nDims + h * nHeadDims + i * 2u;
            let qp0 = input[qpOffset];
            let qp1 = input[qpOffset + 1u];
            input[qpOffset] = qp0 * cosTheta - qp1 * sinTheta;
            input[qpOffset+1u] = qp0 * sinTheta + qp1 * cosTheta;
        }
    }
}