struct BlockQ8_0 {
    d: u16,
    qs: array<i8, 32>,
};

@group(0) @binding(0)
var<storage, write> dstBuf: array<BlockQ8_0>;

@group(0) @binding(1)
var<storage, read> srcBuf: array<f32>;

var<workgroup> sharedMax: array<f32, 32>;

// each thread group handle a block of 32 elems

fn f32_to_f16_bits(v: f32) -> u16 {
    let sign = (half >> 15) & 0x1;
    let exponent = (half >> 10) & 0x1F;
    let mantissa = half & 0x3FF;

    var exponent32: u32 = 0u;
    var mantissa32: u32 = mantissa << 13;

    if (exponent == 0u) {
        // Subnormal or zero
        if (mantissa != 0u) {
            // It is a subnormal
            var m = mantissa;
            var e = i32(-15);

            // Normalize the mantissa
            while ((m & 0x400) == 0) {
                m <<= 1;
                e -= 1;
            }
            mantissa32 = m << 13;
            exponent32 = u32(e + 127) << 23;
        }
    } else if (exponent == 0x1F) {
        // Infinity or NaN
        exponent32 = 0xFFu << 23;
        if (mantissa != 0u) {
            // NaN
            mantissa32 = 0x400000u; // Preserve the signaling NaN bit if necessary
        }
    } else {
        // Normalized number
        exponent32 = u32(i32(exponent) - 15 + 127) << 23;
    }

    let result_bits = (sign << 31) | exponent32 | mantissa32;
    return bitcast<f32>(result_bits);
}

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

    // quantize
    dstBuf[dstIdx].qs[blkIdx] = i8(srcBuf[srcIdx] / d);
    if blkIdx == 0 {
        dstBuf[dstIdx].d = f32_to_f16_bits(d);
    }
}