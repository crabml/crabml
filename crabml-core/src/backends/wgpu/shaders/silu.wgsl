@group(0) @binding(0)
var<storage, read_write> input: array<f32>;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let gidx = workgroup_id.x * 32u + local_id.x;
    if gidx >= arrayLength(&input) {
        return;
    }

    // v = v / (1 + (-v).exp())
    input[gidx] = input[gidx] / (1.0f + exp(-input[gidx]));
}