struct Meta {
    M: u32,
    N: u32,
}

@group(0) @binding(0)
var<storage, read_write> input: array<f32>;

@group(0) @binding(1)
var<storage, read> input_m: Meta;

const COEF_A: f32 = 0.044715;
const SQRT_2_OVER_PI: f32 = 0.79788456080286535587989211986876;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let mi = workgroup_id.x * 32u + local_id.x;
    if (mi >= input_m.M) {
        return;
    }

    // x = 0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + COEF_A * x * x)).tanh())
    for (var ni = 0u; ni < input_m.N; ni = ni + 1u) {
        let i = mi * input_m.N + ni;
        let x = input[i];
        input[i] = 0.5 * x * tanh(1.0 + (SQRT_2_OVER_PI * x * (1.0 + COEF_A * x * x)));
    }
}