use crate::backends::cpu::buf::CpuTensorBuf;
use crate::error::Result;
use crate::tensor::TensorStrider;

pub fn div_inplace<'a>(
    buf1: &mut CpuTensorBuf<'a>,
    buf2: &CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) -> Result<()> {
    assert!(buf1.len() == buf2.len() || buf2.len() == 1);
    assert!(strider1.shape() == strider2.shape() || buf2.len() == 1);
    assert!(strider1.is_contiguous());
    assert!(strider2.is_contiguous());

    if buf2.len() == 1 {
        let ib = buf2.iter().next().unwrap();
        buf1.iter_mut().for_each(|ia| {
            *ia /= ib;
        });
        return Ok(());
    }

    buf1.iter_mut().zip(buf2.iter()).for_each(|(ia, ib)| {
        *ia /= ib;
    });
    Ok(())
}
