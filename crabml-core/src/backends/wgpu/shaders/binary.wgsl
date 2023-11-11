@group(0) @binding(0)
var<storage, read_write> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@compute
@workgroup_size(64)
fn add_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    let rhs_idx = gidx % arrayLength(&rhs);
    lhs[gidx] += rhs[rhs_idx];
}

@compute
@workgroup_size(64)
fn mul_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    let rhs_idx = gidx % arrayLength(&rhs);
    lhs[gidx] *= rhs[rhs_idx];
}

@compute
@workgroup_size(64)
fn div_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    let rhs_idx = gidx % arrayLength(&rhs);
    lhs[gidx] /= rhs[rhs_idx];
}