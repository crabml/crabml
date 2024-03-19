use std::mem;

type Thunk<'a> = Box<dyn FnOnce() + Send + 'a>;

type Work = (Thunk<'static>, crossbeam_utils::sync::WaitGroup);

/// A threadpool that acts as a handle to a number
/// of threads spawned at construction.
pub struct ThreadPool {
    tx: crossbeam_channel::Sender<Work>,
}

impl ThreadPool {
    /// Construct a threadpool with the given number of threads.
    /// Minimum value is `1`.
    pub fn new(n: u32) -> Self {
        assert!(n >= 1);

        let (tx, rx) = crossbeam_channel::bounded(4);

        for _ in 0..n {
            let rx: crossbeam_channel::Receiver<Work> = rx.clone();
            std::thread::spawn(move || {
                while let Ok((thunk, wg)) = rx.recv() {
                    thunk();
                    drop(wg)
                }
            });
        }

        Self { tx }
    }

    pub fn scoped<'scope, F>(&self, f: F)
    where F: FnOnce(&mut Scope<'scope>) + Send + 'scope {
        let mut scope = Scope::<'scope> {
            thunks: Vec::new(),
            _phantom: std::marker::PhantomData,
        };

        f(&mut scope);

        let wg = crossbeam_utils::sync::WaitGroup::new();
        for thunk in scope.into_inner().into_iter() {
            let work = (thunk, wg.clone());
            self.tx.send(work).unwrap();
        }

        wg.wait();
    }
}

pub struct Scope<'scope> {
    thunks: Vec<Thunk<'static>>,
    _phantom: std::marker::PhantomData<&'scope ()>,
}

impl<'scope> Scope<'scope> {
    pub fn spawn<'a, F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'a,
    {
        let b = unsafe { mem::transmute::<Thunk<'a>, Thunk<'static>>(Box::new(f)) };
        self.thunks.push(b)
    }

    pub fn into_inner(self) -> Vec<Thunk<'static>> {
        self.thunks
    }
}
