struct Meta {
    M: u32, // number of vectors
    N: u32, // length of vector
    pos: u32,
    n_heads: u32,
    rope_dims: u32,
    _padding: vec3<u32>,
};

@group(0) @binding(0)
var<storage, read_write> input: array<f32>;

@group(0) @binding(1)
var<storage, read> input_m: Meta;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let head_size = input_m.N / input_m.n_heads;
    let idx_m = workgroup_id.x * 32u + local_id.x;

    // process each vector in one thread. if there's only one vector, only idx_m == 0 makes sense
    if idx_m > input_m.M {
        return;
    }

    for (var h = 0u; h < input_m.n_heads; h++) {
        for (var i = 0u; i < input_m.rope_dims / 2u; i++) {
            let theta_scale = pow(10000.0, -2.0 * f32(i) / f32(head_size));
            let theta = f32(input_m.pos) * theta_scale;

            let cos_theta = cos(theta);
            let sin_theta = sin(theta);
            let qp_offset = idx_m * input_m.N + h * head_size + i * 2u;
            let qp0 = input[qp_offset];
            let qp1 = input[qp_offset + 1u];
            input[qp_offset] = qp0 * cos_theta - qp1 * sin_theta;
            input[qp_offset+1u] = qp0 * sin_theta + qp1 * cos_theta;
        }
    }
}