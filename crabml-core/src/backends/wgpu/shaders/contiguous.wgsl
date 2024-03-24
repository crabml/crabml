struct Meta {
    shape: vec4<u32>,
    strides: vec4<u32>,
    nDims: u32,
};

@group(0) @binding(0)
var<storage, read_write> buf1: array<f32>;

@group(0) @binding(1)
var<storage, read> buf2: array<f32>;

@group(0) @binding(2)
var<storage, read> bufM: Meta;

fn contiguous2D(globalID: vec3<u32>) {
    let y = globalID.x % bufM.shape.y;
    let x = (globalID.x - y) / bufM.shape.y;
    
    let idxOrig = bufM.strides.x * x + bufM.strides.y * y;
    let idxCont = x * bufM.shape.y + y;
    buf1[idxCont] = buf2[idxOrig];
}

fn contiguous3D(globalID: vec3<u32>) {
    let z = globalID.x % bufM.shape.z;
    let y = (globalID.x - z) / bufM.shape.z % bufM.shape.y;
    let x = (globalID.x - z - y * bufM.shape.z) / (bufM.shape.y * bufM.shape.z);

    let idxOrig = bufM.strides.x * x + bufM.strides.y * y + bufM.strides.z * z;
    let idxCont = x * bufM.shape.y * bufM.shape.z + y * bufM.shape.z + z;
    buf1[idxCont] = buf2[idxOrig];
}

@compute
@workgroup_size(16)
fn main(
    @builtin(global_invocation_id) globalID: vec3<u32>,
) {
    if bufM.nDims == 2u {
        contiguous2D(globalID);
    } else {
        contiguous3D(globalID);
    }
}