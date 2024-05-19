#version 450

layout(local_size_x = 32) in;

layout(set = 0, binding = 0) buffer InputBufferA {
    float bufA[];
};

layout(set = 0, binding = 1) buffer InputBufferB {
    float bufB[];
};

layout(push_constant) uniform PushConstants {
    uint nElems;
    uint op;
    uint use_scalar_rhs;
    float scalar_rhs;
} pcs;

const int OP_ADD = 43;
const int OP_SUB = 45;
const int OP_MUL = 42;
const int OP_DIV = 47;

void main() {
    uint idxA = gl_GlobalInvocationID.x;

    if (idxA >= pcs.nElems) {
        return;
    }

    float rhs = 0.0;
    if (pcs.use_scalar_rhs > 0) {
        rhs = pcs.scalar_rhs;
    } else {
        uint idxB = idxA % bufB.length();
        rhs = bufB[idxB];
    }

    switch (pcs.op) {
        case OP_ADD:
            bufA[idxA] += rhs;
            break;
        case OP_SUB:
            bufA[idxA] -= rhs;
            break;
        case OP_MUL:
            bufB[idxA] *= rhs;
            break;
        case OP_DIV:
            bufB[idxA] /= rhs;
            break;
    }
}