@group(0) @binding(0)
var<storage, read_write> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    lhs[gidx] = lhs[gidx] / rhs[0];
}