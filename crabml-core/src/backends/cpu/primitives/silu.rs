use crate::backends::cpu::buf::buf_f32::exp_f32_cached;
use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::error::Result;

pub fn silu_inplace<'a>(device: CpuTensorDeviceRef<'a>, buf: &mut CpuTensorBuf<'a>) -> Result<()> {
    let exp_cache = device.exp_cache.as_ref();
    buf.as_f32_mut().chunks_exact_mut(4).for_each(|v| {
        let nexp0 = exp_f32_cached(-v[0], exp_cache);
        let nexp1 = exp_f32_cached(-v[1], exp_cache);
        let nexp2 = exp_f32_cached(-v[2], exp_cache);
        let nexp3 = exp_f32_cached(-v[3], exp_cache);
        v[0] /= 1.0 + nexp0;
        v[1] /= 1.0 + nexp1;
        v[2] /= 1.0 + nexp2;
        v[3] /= 1.0 + nexp3;
    });
    Ok(())
}
