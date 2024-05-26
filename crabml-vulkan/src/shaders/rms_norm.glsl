#version 450

layout(set = 0, binding = 0) buffer InputBuffer {
    float bufA[];
};

layout(push_constant) uniform PushConstants {
    uint numRows;
    uint numDims;
    float eps;
} pcs;

// each workgroup processes a row
// each thread processes a chunk
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

shared float[32] sketches;

void main() {
    uint rowIdx = gl_WorkGroupID.x;
    uint rowSize = pcs.numDims;
    uint chunkIdx = gl_LocalInvocationID.x;
    uint chunkSize = rowSize / 32;


    // calculate sum of squares
    sketches[chunkIdx] = 0.0;
    for (uint i = 0; i < chunkSize; i++) {
        uint idx = rowIdx * rowSize + chunkIdx * chunkSize + i;
        sketches[chunkIdx] += bufA[idx] * bufA[idx];
    }
    barrier();

    // get rms of the sum of squares
    if (chunkIdx == 0) {
        float squareSum = 0.0;
        for (uint i = 0; i < 32; i++) {
            squareSum += sketches[i];
        }
        sketches[0] = sqrt(squareSum / rowSize + pcs.eps);
    }
    barrier();
    float scale = 1.0 / sketches[0];

    // normalize by rms value
    for (uint i = 0; i < chunkSize; i++) {
        uint idx = rowIdx * rowSize + chunkIdx * chunkSize + i;
        bufA[idx] *= scale;
    }
}