#version 450

layout(set = 0, binding = 0) buffer BufA {
    float bufa[];
};

layout(set = 0, binding = 1) buffer BufB {
    float bufb[];
};

layout(set = 0, binding = 2) buffer BufC {
    float bufc[];
};

layout(push_constant) uniform PushConstants {
    uint B;
    uint M;
    uint K;
    uint N;
    uvec3 stricesB;
} pcs;

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint gidx = gl_WorkGroupID.x * 32u + gl_LocalInvocationID.x;

    uint ni = gidx % pcs.N;
    uint mi = ((gidx - ni) / pcs.N) % pcs.M;
    uint bi = (gidx - ni - mi * pcs.N) / (pcs.M * pcs.N);

    float sum = 0.0;
    for (uint ki = 0u; ki < pcs.K; ki++) {
        float a = bufa[bi * pcs.M * pcs.K + mi * pcs.K + ki];
        float b = bufb[pcs.stricesB.x * bi + ki * pcs.stricesB.y + ni * pcs.stricesB.z];
        sum += a * b;
    }

    bufc[bi * pcs.M * pcs.N + mi * pcs.N + ni] = sum;
}