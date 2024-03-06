use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// stores the metrics on the tensor's privimives
#[derive(Debug, Default, Clone)]
pub struct TensorMetrics {
    pub rms_norm_walltime: TimeMetric,
    pub add_walltime: TimeMetric,
    pub total_walltime: TimeMetric,
    pub mul_walltime: TimeMetric,
    pub rope_walltime: TimeMetric,
    pub softmax_walltime: TimeMetric,
    pub activate_walltime: TimeMetric,
    pub matmul_walltime: TimeMetric,
    pub matmul_quantize_walltime: TimeMetric,
    pub matmul_vec_dot_walltime: TimeMetric,
    pub batch_matmul_walltime: TimeMetric,
    pub alloc_walltime: TimeMetric,
    pub sample_walltime: TimeMetric,
    pub forward_walltime: TimeMetric,
    pub save_kvcache_walltime: TimeMetric,
    pub mqa_walltime: TimeMetric,
    pub ffn_walltime: TimeMetric,
    pub copy_walltime: TimeMetric,
}

impl TensorMetrics {
    pub fn reset(&self) {
        self.rms_norm_walltime.reset();
        self.add_walltime.reset();
        self.mul_walltime.reset();
        self.rope_walltime.reset();
        self.softmax_walltime.reset();
        self.matmul_walltime.reset();
        self.activate_walltime.reset();
        self.total_walltime.reset();
        self.matmul_quantize_walltime.reset();
        self.matmul_vec_dot_walltime.reset();
        self.batch_matmul_walltime.reset();
        self.alloc_walltime.reset();
        self.sample_walltime.reset();
        self.forward_walltime.reset();
        self.save_kvcache_walltime.reset();
        self.mqa_walltime.reset();
        self.ffn_walltime.reset();
        self.copy_walltime.reset();
    }

    pub fn as_vec(&self) -> Vec<(String, f64)> {
        vec![
            (
                "rms_norm_walltime".to_string(),
                self.rms_norm_walltime.as_millis(),
            ),
            (
                "forward_walltime".to_string(),
                self.forward_walltime.as_millis(),
            ),
            (
                "sample_walltime".to_string(),
                self.sample_walltime.as_millis(),
            ),
            ("add_walltime".to_string(), self.add_walltime.as_millis()),
            (
                "activate_walltime".to_string(),
                self.activate_walltime.as_millis(),
            ),
            (
                "alloc_walltime".to_string(),
                self.alloc_walltime.as_millis(),
            ),
            (
                "total_walltime".to_string(),
                self.total_walltime.as_millis(),
            ),
            ("rope_walltime".to_string(), self.rope_walltime.as_millis()),
            (
                "softmax_walltime".to_string(),
                self.softmax_walltime.as_millis(),
            ),
            ("mul_walltime".to_string(), self.mul_walltime.as_millis()),
            (
                "matmul_walltime".to_string(),
                self.matmul_walltime.as_millis(),
            ),
            (
                "matmul_vec_dot_walltime".to_string(),
                self.matmul_vec_dot_walltime.as_millis(),
            ),
            (
                "matmul_quantize_walltime".to_string(),
                self.matmul_quantize_walltime.as_millis(),
            ),
            (
                "batch_matmul_walltime".to_string(),
                self.batch_matmul_walltime.as_millis(),
            ),
            (
                "save_kv_cache_walltime".to_string(),
                self.save_kvcache_walltime.as_millis(),
            ),
            ("mqa_walltime".to_string(), self.mqa_walltime.as_millis()),
            ("ffn_walltime".to_string(), self.ffn_walltime.as_millis()),
            ("copy_walltime".to_string(), self.copy_walltime.as_millis()),
        ]
    }
}

#[derive(Clone, Debug, Default)]
pub struct TimeMetric {
    pub inner: Arc<AtomicU64>,
}

pub struct TimeMetricGuard {
    m: TimeMetric,
    start_at: std::time::Instant,
}

impl TimeMetric {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn reset(&self) {
        self.inner.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn as_millis(&self) -> f64 {
        self.inner.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000000.0
    }

    pub fn track(&self) -> TimeMetricGuard {
        TimeMetricGuard {
            m: self.clone(),
            start_at: std::time::Instant::now(),
        }
    }
}

impl Drop for TimeMetricGuard {
    fn drop(&mut self) {
        let elapsed = self.start_at.elapsed().as_nanos() as u64;
        self.m
            .inner
            .fetch_add(elapsed, std::sync::atomic::Ordering::Relaxed);
    }
}
