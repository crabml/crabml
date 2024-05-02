struct BlockQ8_0 {
    d: f16,
    qs: array<i8, 32>,
};

@group(0) @binding(0)
var<storage, write> dstBuf: array<BlockQ8_0>;

@group(0) @binding(1)
var<storage, read> srcBuf: array<f32>;

var<workgroup> sharedMax: array<f32, 32>;

// each thread group handle a block of 32 elems

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(global_invocation_id) gIdx: vec3<u32>,
    @builtin(workgroup_id) wIdx: vec3<u32>,
) {
    let dstIdx = wIdx.x;
    let srcIdx = gIdx.x;
    let blkIdx = srcIdx % 32;

    if dstIdx >= dstBuf.size() {
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

    // quantize
    dstBuf[dstIdx].qs[blkIdx] = i8(srcBuf[srcIdx] / d);
    if blkIdx == 0 {
        dstBuf[dstIdx].d = f16(d);
    }
}