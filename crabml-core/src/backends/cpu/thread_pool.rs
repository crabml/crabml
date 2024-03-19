use std::sync::Arc;

struct Work {
    thunk: Option<Box<dyn FnOnce() + Send + 'static>>,
    latch: Option<Arc<countdown_latch::CountDownLatch>>,
}

/// A threadpool that acts as a handle to a number
/// of threads spawned at construction.
pub struct Pool {
    tx: crossbeam_channel::Sender<Work>,
}

impl Pool {
    /// Construct a threadpool with the given number of threads.
    /// Minimum value is `1`.
    pub fn new(n: u32) -> Pool {
        assert!(n >= 1);

        let (tx, rx) = crossbeam_channel::bounded(0);

        for _ in 0..n {
            let rx: crossbeam_channel::Receiver<Work> = rx.clone();
            std::thread::spawn(move || {
                while let Ok(mut work) = rx.recv() {
                    let thunk = work.thunk.take().unwrap();
                    thunk();

                    if let Some(latch) = work.latch.take() {
                        latch.count_down();
                    }
                }
            });
        }

        Pool { tx }
    }
}
