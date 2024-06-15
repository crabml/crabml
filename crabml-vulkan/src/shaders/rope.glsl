#version 450

layout(set = 0, binding = 0) buffer InputBuffer {
    float bufA[];
};

layout(push_constant) uniform PushConstants {
    uint nBatch; // number of vectors
    uint nDims;  // length of vector
    uint pos;
    uint nHeads;
    uint nRopeDims;
} pcs;

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint nHeadDims = pcs.nDims / pcs.nHeads;
    uint gidx = gl_GlobalInvocationID.x;

    // process each vector in one thread. if there's only one vector, only gidx == 0 makes sense
    if (gidx >= pcs.nBatch) {
        return;
    }

    for (uint h = 0u; h < pcs.nHeads; h++) {
        for (uint i = 0u; i < pcs.nRopeDims / 2u; i++) {
            float thetaScale = pow(10000.0, -2.0 * float(i) / float(nHeadDims));
            float theta = float(pcs.pos) * thetaScale;

            float cosTheta = cos(theta);
            float sinTheta = sin(theta);
            uint qpOffset = gidx * pcs.nDims + h * nHeadDims + i * 2u;
            float qp0 = bufA[qpOffset];
            float qp1 = bufA[qpOffset + 1u];
            bufA[qpOffset] = qp0 * cosTheta - qp1 * sinTheta;
            bufA[qpOffset + 1u] = qp0 * sinTheta + qp1 * cosTheta;
        }
    }
}