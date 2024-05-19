use crate::cpu::buf::buf_f32::exp_f32_cached;
use crate::cpu::buf::CpuTensorBuf;
use crate::cpu::CpuTensorDeviceRef;
use crate::error::Result;

pub fn silu_inplace<'a>(device: CpuTensorDeviceRef<'a>, buf: &mut CpuTensorBuf<'a>) -> Result<()> {
    let exp_cache = device.exp_cache.as_ref();
    buf.as_f32_mut().iter_mut().for_each(|vp| {
        let nexp = exp_f32_cached(-*vp, exp_cache);
        *vp /= 1.0 + nexp;
    });
    Ok(())
}
