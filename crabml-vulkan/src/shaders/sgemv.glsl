#version 450

layout(set = 0, binding = 0) buffer BufA {
    vec4 bufA[];
};

layout(set = 0, binding = 1) buffer BufB {
    vec4 bufB[];
};

layout(set = 0, binding = 2) buffer BufC {
    vec4 bufC[];
};

layout(push_constant) uniform PushConstant {
    uint B;
    uint M;
    uint K;
} pcs;

layout(local_size_x = 1, local_size_y = 8, local_size_z = 1) in;

void main() {
    uint B = pcs.B;
    uint M = pcs.M;
    uint K = pcs.K;
    uint bi = gl_GlobalInvocationID.x;
    uint mi = (gl_GlobalInvocationID.y * 4u) % M;

    vec4 tmp = vec4(0.0);
    for (uint ki = 0u; ki < K; ki += 4u) {
        vec4 bc = bufB[(bi * K + ki) / 4u];
        float x = dot(bufA[mi * K / 4u + ki / 4u], bc);
        float y = dot(bufA[(mi + 1u) * K / 4u + ki / 4u], bc);
        float z = dot(bufA[(mi + 2u) * K / 4u + ki / 4u], bc);
        float w = dot(bufA[(mi + 3u) * K / 4u + ki / 4u], bc);
        tmp += vec4(x, y, z, w);
    }
    bufC[(bi * M + mi) / 4u] = tmp;
}