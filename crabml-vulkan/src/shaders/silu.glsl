#version 450

layout(local_size_x = 32) in;

layout(set = 0, binding = 0) buffer InputBufferA {
    float bufA[];
};

const int OP_ADD = 43;
const int OP_SUB = 45;
const int OP_MUL = 42;
const int OP_DIV = 47;

void main() {
    uint gidx = gl_GlobalInvocationID.x;

    if (gidx >= bufA.length()) {
        return;
    }

    bufA[gidx] = bufA[gidx] / (1.0 + exp(-bufA[gidx]));
}