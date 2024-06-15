#version 450

layout(set = 0, binding = 0) buffer BufDst {
    float bufDst[];
};

layout(set = 0, binding = 1) buffer BufSrc {
    float bufSrc[];
};

layout(push_constant) uniform PushConstants {
    uvec4 shape;
    uvec4 strides;
    uint nDims;
    uint nElms;
} pcs;

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

void contiguous2D(uint dstOffset) {
    uint y = dstOffset % pcs.shape.y;
    uint x = (dstOffset - y) / pcs.shape.y;
    
    uint srcOffset = pcs.strides.x * x + pcs.strides.y * y;
    bufDst[dstOffset] = bufSrc[srcOffset];
}

void contiguous3D(uint dstOffset) {
    uint z = dstOffset % pcs.shape.z;
    uint y = (dstOffset - z) / pcs.shape.z % pcs.shape.y;
    uint x = (dstOffset - z - y * pcs.shape.z) / (pcs.shape.y * pcs.shape.z);

    uint srcOffset = pcs.strides.x * x + pcs.strides.y * y + pcs.strides.z * z;
    bufDst[dstOffset] = bufSrc[srcOffset];
}


void main() {
    uint dstOffset = gl_GlobalInvocationID.x;

    if (dstOffset >= pcs.nElms) {
        return;
    }

    if (pcs.nDims == 2u) {
        contiguous2D(dstOffset);
    } else {
        contiguous3D(dstOffset);
    }
}