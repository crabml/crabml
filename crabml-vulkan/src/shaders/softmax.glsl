#version 450

layout(set = 0, binding = 0) buffer InputBuffer {
    float bufA[];
};

layout(push_constant) uniform PushConstants {
    uint numRows;
    uint numCols;
} pcs;

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint row = gl_WorkGroupID.x;

    if (row >= pcs.numRows) {
        return;
    }

    // the starting index for this row
    uint rowOffset = row * pcs.numCols;

    // 1. get the max value
    float maxVal = bufA[rowOffset];
    for (uint col = 1; col < pcs.numCols; ++col) {
        maxVal = max(maxVal, bufA[rowOffset + col]);
    }

    // 2. compute the exponentials and their sum
    float sumExp = 0.0;
    for (uint col = 0; col < pcs.numCols; ++col) {
        bufA[rowOffset + col] = exp(bufA[rowOffset + col] - maxVal);
        sumExp += bufA[rowOffset + col];
    }

    // 3. normalize the values to get the softmax
    for (uint col = 0; col < pcs.numCols; ++col) {
        bufA[rowOffset + col] /= sumExp;
    }
}