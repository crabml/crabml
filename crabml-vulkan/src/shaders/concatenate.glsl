#version 450

layout(set = 0, binding = 0) buffer Buf1 {
    float dstBuf[];
};

layout(set = 0, binding = 1) buffer Buf2 {
    float srcBuf[];
};

layout(push_constant) uniform PushConstants {
    uvec4 shape1;
    uvec4 shape2;
    uvec4 strides1;
    uvec4 strides2;
    uint axis;
    uint dims;
    uint nElems;
} pcs;

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

void concatenate2D(uint srcOffset) {
    uint y = srcOffset % pcs.shape2.y;
    uint x = (srcOffset - y) / pcs.shape2.y;

    uint dstBase = pcs.shape1[pcs.axis] * pcs.strides1[pcs.axis];
    uint dstOffset = x * pcs.strides1.x + y * pcs.strides1.y + dstBase;
    dstBuf[dstOffset] = srcBuf[srcOffset];
}

void concatenate3D(uint srcOffset) {
    uint z = srcOffset % pcs.shape2.z;
    uint y = (srcOffset - z) / pcs.shape2.z % pcs.shape2.y;
    uint x = (srcOffset - z - y) / (pcs.shape2.y * pcs.shape2.z);

    uint dstBase = pcs.shape1[pcs.axis] * pcs.strides1[pcs.axis];
    uint dstOffset = x * pcs.strides1.x + y * pcs.strides1.y + z * pcs.strides1.z + dstBase;
    dstBuf[dstOffset] = srcBuf[srcOffset];
}

void main() {
    uvec3 globalId = gl_GlobalInvocationID;
    uint srcOffset = globalId.x;

    if (srcOffset >= pcs.nElems) {
        return;
    }

    if (pcs.dims == 3u) {
        concatenate3D(srcOffset);
    } else if (pcs.dims == 2u) {
        concatenate2D(srcOffset);
    }
}