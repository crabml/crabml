struct BlockQ8_0 {
    // f16 is supposed to be supported in webgpu spec, but is still wip in
    // the webgpu-rs implementation. 
    d: f32,
    // webgpu doesn't support i8 array, we have to pack them into a u32
    // by pack4xi8(e: vec4<i32>) -> u32, and unpack them by unpack4xi8(u32)
    // -> vec4<i32>.
    qs: array<u32, 4>,
};

@group(0) @binding(0)
var<storage, write> dstBuf: array<BlockQ8_0>;

@group(0) @binding(1)
var<storage, read> srcBuf: array<f32>;

var<workgroup> sharedMax: array<f32, 32>;
var<workgroup> sharedQs: array<i32, 32>;

// each thread group handle a block of 32 elems

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(global_invocation_id) gIdx: vec3<u32>,
    @builtin(workgroup_id) wIdx: vec3<u32>,
) {
    let dstIdx = wIdx.x;
    let srcIdx = gIdx.x;
    let blkIdx = srcIdx % 32;

    if dstIdx >= arrayLength(dstBuf) {
        return;
    }

    // find the max abs value
    sharedMax[blkIdx] = abs(srcBuf[srcIdx]);
    for (var offset = 16u; offset > 0; offset /= 2) {
        if blkIdx < offset {
            sharedMax[blkIdx] = max(sharedMax[blkIdx], sharedMax[blkIdx + offset]);
        }
        workgroupBarrier();
    }
    let d = f32(sharedMax[0]) / 127.0;

    if blkIdx == 0 {
        dstBuf[dstIdx].d = d;
    }

    // quantize and pack the quantized values into qs
    sharedQs[blkIdx] = i32(srcBuf[srcIdx] / d);
    if blkIdx % 4 == 0 {
        let qpack = pack4xI8(vec4<i32>(
            sharedQs[blkIdx],
            sharedQs[blkIdx + 1],
            sharedQs[blkIdx + 2],
            sharedQs[blkIdx + 3]
        ));
        dstBuf[dstIdx].qs[blkIdx / 4] = qpack;
    }
}