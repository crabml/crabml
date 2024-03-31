use std::mem;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

type Thunk<'a> = Box<dyn FnOnce() + Send + 'a>;

type Work = (Thunk<'static>, Arc<AtomicUsize>, Instant);

/// A threadpool that acts as a handle to a number
/// of threads spawned at construction.
#[derive(Debug)]
pub struct ThreadPool {
    senders: Vec<crossbeam_channel::Sender<Work>>,
}

impl ThreadPool {
    /// Construct a threadpool with the given number of threads.
    /// Minimum value is `1`.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1);

        let mut senders: Vec<crossbeam_channel::Sender<Work>> = vec![];
        for _ in 0..n {
            let (sender, receiver) = crossbeam_channel::unbounded();
            senders.push(sender);
            std::thread::spawn(move || {
                while let Ok((thunk, counter, _dispatched_time)) = receiver.recv() {
                    thunk();
                    counter.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                }
            });
        }

        Self { senders }
    }

    pub fn scoped<'scope, F>(&self, f: F)
    where F: FnOnce(&mut Scope<'scope>) + Send + 'scope {
        let mut scope = Scope::<'scope> {
            thunks: Vec::with_capacity(4),
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

        let counter = Arc::new(AtomicUsize::new(thunks_len));
        for (i, thunk) in thunks_iter.enumerate() {
            let work = (thunk, counter.clone(), Instant::now());
            let thread_idx = i % self.senders.len();
            self.senders[thread_idx].send(work).unwrap();
        }

        // execute the first thunk in the current thread.
        first_thunk();

        // await the completion of the remaining thunks. a busy loop here appears to be faster than
        // using crossbeam's WaitGroup. generating a token can trigger 100+ task coordination operations
        // (using condvar inside), which might introduce an overhead about 2ms for each token generation.
        while counter.load(std::sync::atomic::Ordering::Relaxed) > 1 {}
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
