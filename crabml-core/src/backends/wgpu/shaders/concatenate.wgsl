struct Meta {
    shape1: vec4<u32>,
    shape2: vec4<u32>,
    strides1: vec4<u32>,
    strides2: vec4<u32>,
    axis: u32,
    dims: u32,
};

@group(0) @binding(0)
var<storage, read_write> buf1: array<f32>;

@group(0) @binding(1)
var<storage, read> buf2: array<f32>;

@group(0) @binding(2)
var<storage, read> bufm: Meta;

fn concatenate_2d(global_id: vec3<u32>) {
    let y = global_id.x % bufm.shape2.y;
    let x = (global_id.x - y) / bufm.shape2.y;

    let buf1_base = bufm.shape1[bufm.axis] * bufm.strides1[bufm.axis];
    let buf1_offset = x * bufm.strides1.x + y * bufm.strides1.y + buf1_base;
    let buf2_offset = x * bufm.strides2.x + y * bufm.strides2.y;
    buf1[buf1_offset] = buf2[buf2_offset];
}

fn concatenate_3d(global_id: vec3<u32>) {
    let z = global_id.x % bufm.shape2.z;
    let y = (global_id.x - z) / bufm.shape2.z % bufm.shape2.y;
    let x = (global_id.x - z - y) / (bufm.shape2.y * bufm.shape2.z);

    let buf1_base = bufm.shape1[bufm.axis] * bufm.strides1[bufm.axis];
    let buf1_offset = x * bufm.strides1.x + y * bufm.strides1.y + z * bufm.strides1.z + buf1_base;
    let buf2_offset = x * bufm.strides2.x + y * bufm.strides2.y + z * bufm.strides2.z;
    buf1[buf1_offset] = buf2[buf2_offset];
}

@compute
@workgroup_size(16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    if bufm.dims == 3u {
        concatenate_3d(global_id);
    } else if bufm.dims == 2u {
        concatenate_2d(global_id);
    }
}