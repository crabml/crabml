type Thunk = Box<dyn FnOnce() + Send + 'static>;

struct Work {
    thunk: Option<Thunk>,
    wg: Option<crossbeam_utils::sync::WaitGroup>,
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

                    if let Some(wg) = work.wg.take() {
                        drop(wg)
                    }
                }
            });
        }

        Pool { tx }
    }

    pub fn scoped<F>(&self, f: F)
    where F: FnOnce(&mut Scope) + Send + 'static {
        let mut scope = Scope { thunks: Vec::new() };
        f(&mut scope);

        let wg = crossbeam_utils::sync::WaitGroup::new();
        for thunk in scope.into_inner().into_iter() {
            let work = Work {
                thunk: Some(thunk),
                wg: Some(wg.clone()),
            };
            self.tx.send(work).unwrap();
        }

        wg.wait();
    }
}

pub struct Scope {
    thunks: Vec<Thunk>,
}

impl Scope {
    pub fn spawn(&mut self, thunk: Thunk) {
        self.thunks.push(thunk)
    }

    pub fn into_inner(self) -> Vec<Thunk> {
        self.thunks
    }
}
