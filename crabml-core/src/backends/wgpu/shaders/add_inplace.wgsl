@group(0) @binding(0)
var<storage, read_write> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

struct Strider {
    dims: u32,
    shape: vec4<u32>,
    strides: vec4<u32>,
};

@group(0) @binding(2)
var<storage, read> striders: array<Strider>;

fn strided_index(input_idx: u32, s: Strider) -> u32 {
    var idx: u32 = input_idx;
    var result: u32 = 0u;
    for (var i: u32 = s.dims - 1u; i >= 0u; i = i - 1u) {
        let dim_len = s.shape[i];
        let dim_stride = s.strides[i];
        let dim_val = idx % dim_len;
        result += dim_val * dim_stride;
        idx = (idx - dim_val) / dim_len;
    }
    return result;
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    let lhs_idx = strided_index(gidx, striders[0]);
    let rhs_idx = strided_index(gidx, striders[1]);
    lhs[lhs_idx] = lhs[lhs_idx] + rhs[rhs_idx];
}