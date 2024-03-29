use std::mem;
use std::sync::mpsc;

type Thunk<'a> = Box<dyn FnOnce() + Send + 'a>;

type Work = (Thunk<'static>, crossbeam_utils::sync::WaitGroup);

/// A threadpool that acts as a handle to a number
/// of threads spawned at construction.
pub struct ThreadPool {
    senders: Vec<mpsc::Sender<Work>>,
}

impl ThreadPool {
    /// Construct a threadpool with the given number of threads.
    /// Minimum value is `1`.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1);

        let mut senders = vec![];
        for _ in 0..n {
            let (sender, receiver) = mpsc::channel::<Work>();
            senders.push(sender);
            std::thread::spawn(move || {
                while let Ok((thunk, wg)) = receiver.recv() {
                    thunk();
                    drop(wg)
                }
            });
        }

        Self { senders }
    }

    pub fn scoped<'scope, F>(&self, f: F)
    where F: FnOnce(&mut Scope<'scope>) + Send + 'scope {
        let mut scope = Scope::<'scope> {
            thunks: Vec::new(),
            _phantom: std::marker::PhantomData,
        };

        f(&mut scope);

        let thunks_vec = scope.into_inner();
        let thunks_len = thunks_vec.len();
        if thunks_len == 0 {
            return;
        }

        let mut thunks_iter = thunks_vec.into_iter();
        let first_thunk = thunks_iter.next().unwrap();

        let wg = crossbeam_utils::sync::WaitGroup::new();
        for (i, thunk) in thunks_iter.enumerate() {
            let work = (thunk, wg.clone());
            let thread_idx = i % self.senders.len();
            self.senders[thread_idx].send(work).unwrap();
        }

        // execute the first thunk in the current thread
        first_thunk();

        // await the rest of the thunks
        wg.wait();
    }
}

pub struct Scope<'scope> {
    thunks: Vec<Thunk<'static>>,
    _phantom: std::marker::PhantomData<&'scope ()>,
}

impl<'scope> Scope<'scope> {
    pub fn spawn<'a, F>(&mut self, f: F)
    where F: FnOnce() + Send + 'a {
        let b = unsafe { mem::transmute::<Thunk<'a>, Thunk<'static>>(Box::new(f)) };
        self.thunks.push(b)
    }

    pub fn into_inner(self) -> Vec<Thunk<'static>> {
        self.thunks
    }
}
