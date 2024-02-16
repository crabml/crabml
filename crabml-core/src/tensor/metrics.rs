use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// stores the metrics on the tensor's privimives
#[derive(Debug, Default, Clone)]
pub struct TensorDeviceMetrics {
    pub rms_norm_milliseconds: TimeMetric,
    pub add_milliseconds: TimeMetric,
    pub mul_milliseconds: TimeMetric,
    pub matmul_milliseconds: TimeMetric,
    pub batch_matmul_milliseconds: TimeMetric,
}

impl TensorDeviceMetrics {
    pub fn reset(&self) {
        self.rms_norm_milliseconds.reset();
        self.add_milliseconds.reset();
        self.mul_milliseconds.reset();
        self.matmul_milliseconds.reset();
        self.batch_matmul_milliseconds.reset();
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

    pub fn track(&self) -> TimeMetricGuard {
        TimeMetricGuard {
            m: self.clone(),
            start_at: std::time::Instant::now(),
        }
    }
}

impl Drop for TimeMetricGuard {
    fn drop(&mut self) {
        let elapsed = self.start_at.elapsed().as_millis() as u64;
        self.m
            .inner
            .fetch_add(elapsed, std::sync::atomic::Ordering::Relaxed);
    }
}
