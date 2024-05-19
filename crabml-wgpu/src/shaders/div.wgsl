struct Meta {
    N: u32, // elments count
}

@group(0) @binding(0)
var<storage, read_write> buf0: array<f32>;

@group(0) @binding(1)
var<storage, read> buf1: array<f32>;

@group(0) @binding(2)
var<storage, read> bufM: Meta;

@compute
@workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let idx = 32u * workgroup_id.x + local_id.x;
    if idx >= bufM.N {
        return;
    }

    buf0[idx] /= buf1[idx % arrayLength(&buf1)];
}