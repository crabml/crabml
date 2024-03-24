@group(0) @binding(0)
var<storage, read_write> input: array<f32>;

const COEF_A: f32 = 0.044715;
const SQRT_2_OVER_PI: f32 = 0.79788456080286535587989211986876;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let gidx = workgroup_id.x * 32u + local_id.x;
    if gidx >= arrayLength(&input) {
        return;
    }

    // x = 0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + COEF_A * x * x)).tanh())
    let x = input[gidx];
    input[gidx] = 0.5 * x * tanh(1.0 + (SQRT_2_OVER_PI * x * (1.0 + COEF_A * x * x)));
}