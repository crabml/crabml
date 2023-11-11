@group(0) @binding(0)
var<storage, read_write> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

// dims: strider_vals[0]
// shape: strider_vals[1..1+dims]
// strides: strider_vals[1+dims..1+dims*2]
@group(0) @binding(2)
var<storage, read> strider_vals: array<u32>;

fn strided_index(input_idx: u32) -> u32 {
    let dims = strider_vals[0];
    var idx: u32 = input_idx;
    var result: u32 = 0u;
    for (var i: u32 = dims - 1u; i >= 0u; i = i - 1u) {
        let dim_len = strider_vals[1u + i];
        let dim_stride = strider_vals[1u + dims + i];
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
    let sidx = strided_index(gidx);
    lhs[sidx] = lhs[sidx] + rhs[sidx];
}