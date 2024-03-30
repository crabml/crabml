use std::sync::LazyLock;
use std::sync::Mutex;
use std::time::Instant;

use crate::backends::cpu::buf::CpuTensorBuf;
use crate::backends::cpu::thread_pool::ThreadPool;
use crate::backends::cpu::CpuTensorDeviceRef;
use crate::tensor::metrics::TimeMetric;
use crate::tensor::TensorStrider;

static POOL: LazyLock<Mutex<ThreadPool>> = LazyLock::new(|| Mutex::new(ThreadPool::new(2)));

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
    bufa: &CpuTensorBuf,
    bufb: &CpuTensorBuf,
    bufc: &mut CpuTensorBuf,
    m: usize,
    k: usize,
) {
    let bufc = bufc.as_f32_mut();
    let bufb = &bufb.quantize(bufa.vec_dot_rhs_dtype()).unwrap();
    let thread_num = device.thread_num();
    let split_size = bufc.len() / thread_num;
    let chunk_size = 16;
    assert!(split_size % chunk_size == 0);

    let metrics = device.metrics.clone();
    let _t = metrics.matmul_walltime.track();

    let thread_metrics: Vec<TimeMetric> = vec![TimeMetric::new(), TimeMetric::new()];
    let total_time = TimeMetric::new();
    let mut scoped_time = Instant::now();
    {
        let _t = total_time.track();

        // TODO: 看每个 thread 执行的时间
        POOL.lock().unwrap().scoped(|s| {
            bufc.chunks_exact_mut(split_size)
                .enumerate()
                .zip(thread_metrics.clone())
                .for_each(|((sn, sbuf), metric)| {
                    s.spawn(move || {
                        let _t = metric.track();
                        sbuf.chunks_exact_mut(chunk_size)
                            .enumerate()
                            .for_each(|(cn, cbuf)| {
                                let offset = sn * split_size + cn * chunk_size;
                                let mi = offset % m;
                                let bi = (offset - mi) / m;
                                for (i, cval) in cbuf.iter_mut().enumerate() {
                                    *cval = bufa.vec_dot((mi + i) * k, bufb, bi * k, k);
                                }
                            });
                    });
                });
            scoped_time = Instant::now();
        });
    }
    let max_thread_nanos = thread_metrics
        .iter()
        .map(|m| m.as_nanos())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    metrics
        .matmul_non_compute_walltime
        .increment_nanos(total_time.as_nanos() - max_thread_nanos);
}
