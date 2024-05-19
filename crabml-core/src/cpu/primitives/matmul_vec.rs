use crate::cpu::buf::CpuTensorBuf;
use crate::cpu::CpuTensorDeviceRef;
use crate::tensor::metrics::TimeMetric;
use crate::tensor::TensorStrider;

/// only dense GEMV is supported
/// (m, k) @ k -> (m, )
/// (m, k) @ (b, k) -> (b, m)
pub fn matmul_vec<'a>(
    device: &CpuTensorDeviceRef<'a>,
    bufa: &CpuTensorBuf<'a>,
    bufb: &CpuTensorBuf<'a>,
    bufc: &mut CpuTensorBuf<'a>,
    strider1: &TensorStrider,
    strider2: &TensorStrider,
) {
    assert!(strider1.is_contiguous());
    assert!(strider2.is_contiguous());
    assert!(strider1.shape().last() == strider2.shape().last());

    let (m, k) = (strider1.shape()[0], strider1.shape()[1]);
    gemv_dense_2d_2d(device, bufa, bufb, bufc, m, k);
}

#[allow(clippy::too_many_arguments)]
fn gemv_dense_2d_2d(
    device: &CpuTensorDeviceRef,
    bufa: &CpuTensorBuf,     // (m, k)
    bufb: &CpuTensorBuf,     // (b, k)
    bufc: &mut CpuTensorBuf, // (b, m)
    m: usize,
    k: usize,
) {
    let metrics = device.metrics.clone();
    let bufc = bufc.as_f32_mut();

    let bufb = &{
        let _t = metrics.matmul_quantize_walltime.track();
        bufb.quantize(bufa.vec_dot_rhs_dtype()).unwrap()
    };
    let thread_num = device.thread_num();

    // each thread handles 1/thread_num of the elements in the C matrix. thread_num is allowed
    // to be even.
    let work_len = bufc.len() / thread_num;
    let chunk_len = 16;

    let _t = metrics.matmul_walltime.track();

    // track walltime of each thread, we can compare the longest one with total walltime, the difference
    // represents the cost of thread synchronization cost.
    let work_walltimes: Vec<TimeMetric> = vec![TimeMetric::new(); thread_num + 1];
    let total_walltime = TimeMetric::new();
    {
        let _t = total_walltime.track();

        device.thread_pool().lock().unwrap().scoped(|s| {
            bufc.chunks_mut(work_len)
                .enumerate()
                .zip(work_walltimes.clone())
                .for_each(|((work_idx, work_buf), work_walltime)| {
                    s.spawn(move || {
                        let _t = work_walltime.track();
                        work_buf.chunks_mut(chunk_len).enumerate().for_each(
                            |(chunk_idx, chunk_buf)| {
                                let elem_idx = work_idx * work_len + chunk_idx * chunk_len;
                                let mi = elem_idx % m;
                                let bi = (elem_idx - mi) / m;
                                for (i, cval) in chunk_buf.iter_mut().enumerate() {
                                    *cval = bufa.vec_dot((mi + i) * k, bufb, bi * k, k);
                                }
                            },
                        );
                    });
                });
        });
    }
}
