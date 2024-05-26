#version 450

const float COEF_A = 0.044715;
const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876;

layout(set = 0, binding = 0) buffer InputBuffer {
    float buf[];
};

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint gidx = gl_GlobalInvocationID.x;

    if (gidx >= buf.length()) {
        return;
    }

    float x = buf[gidx];
    buf[gidx] = 0.5 * x * tanh(1.0 + (SQRT_2_OVER_PI * x * (1.0 + COEF_A * x * x)));
}