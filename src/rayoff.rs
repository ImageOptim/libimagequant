use once_cell::unsync::OnceCell;
use std::slice::ChunksMut;

pub(crate) struct ThreadLocal<T>(OnceCell<T>);

impl<T> ThreadLocal<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self(OnceCell::new())
    }

    #[inline(always)]
    pub fn get_or(&self, f: impl FnOnce() -> T) -> &T {
        self.0.get_or_init(f)
    }

    #[inline(always)]
    pub fn get_or_try<E>(&self, f: impl FnOnce() -> Result<T, E>) -> Result<&T, E> {
        self.0.get_or_try_init(f)
    }
}

impl<T> IntoIterator for ThreadLocal<T> {
    type Item = T;

    type IntoIter = std::option::IntoIter<T>;

    #[inline(always)]
    fn into_iter(mut self) -> Self::IntoIter {
        self.0.take().into_iter()
    }
}

pub(crate) trait FakeRayonIter: Sized {
    fn par_bridge(self) -> Self;
}


impl<T> FakeRayonIter for T where Self: Sized {
    fn par_bridge(self) -> Self { self }
}

pub(crate) trait FakeRayonIntoIter<T> {
    fn par_chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<T>;
}

impl<'a, T> FakeRayonIntoIter<T> for &'a mut [T] {
    #[inline(always)]
    fn par_chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<T> {
        self.chunks_mut(chunk_size)
    }
}

impl<'a, T> FakeRayonIntoIter<T> for Box<[T]> {
    #[inline(always)]
    fn par_chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<T> {
        self.chunks_mut(chunk_size)
    }
}

pub(crate) struct SpawnMock;

impl SpawnMock {
    pub fn spawn<F, R>(&self, f: F) -> R where F: FnOnce(SpawnMock) -> R {
        f(SpawnMock)
    }
}

pub(crate) fn scope<F, R>(f: F) -> R where F: FnOnce(SpawnMock) -> R {
    f(SpawnMock)
}

pub(crate) fn num_cpus() -> usize {
    1
}
