use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

pub fn add_inplace<'a>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    assert!(buf1.len() == buf2.len());
    assert!(strider1.shape() == strider2.shape());
    assert!(strider1.is_contiguous());
    assert!(strider2.is_contiguous());

    buf1.iter_f32_mut()
        .zip(buf2.iter_f32())
        .for_each(|(ia, ib)| {
            *ia += ib;
        });

    Ok(())
}
