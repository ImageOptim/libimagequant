use core::mem::{self, MaybeUninit};
use core::slice;

#[cfg(all(not(feature = "std"), feature = "no_std"))]
use std::{boxed::Box, vec::Vec};

#[cfg(feature = "_internal_c_ffi")]
use core::ffi::c_void;

#[derive(Clone)]
pub struct SeaCow<'a, T> {
    inner: SeaCowInner<'a, T>,
}

unsafe impl<T: Send> Send for SeaCowInner<'_, T> {}
unsafe impl<T: Sync> Sync for SeaCowInner<'_, T> {}

/// Rust assumes `*const T` is never `Send`/`Sync`, but it can be.
/// This is fudge for https://github.com/rust-lang/rust/issues/93367
#[repr(transparent)]
#[derive(Copy, Clone)]
pub(crate) struct Pointer<T>(pub *const T);

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct PointerMut<T>(pub *mut T);

unsafe impl<T: Send + Sync> Send for Pointer<T> {}
unsafe impl<T: Send + Sync> Sync for Pointer<T> {}
unsafe impl<T: Send + Sync> Send for PointerMut<T> {}
unsafe impl<T: Send + Sync> Sync for PointerMut<T> {}

impl<T> SeaCow<'static, T> {
    #[inline]
    #[must_use]
    pub fn boxed(data: Box<[T]>) -> Self {
        Self { inner: SeaCowInner::Boxed(data) }
    }
}

impl<'a, T> SeaCow<'a, T> {
    #[inline]
    #[must_use]
    pub const fn borrowed(data: &'a [T]) -> Self {
        Self { inner: SeaCowInner::Borrowed(data) }
    }

    /// The pointer must be `malloc`-allocated
    #[inline]
    #[cfg(feature = "_internal_c_ffi")]
    #[must_use]
    pub unsafe fn c_owned(ptr: *mut T, len: usize, free_fn: unsafe extern "C" fn(*mut c_void)) -> Self {
        debug_assert!(!ptr.is_null());
        debug_assert!(len > 0);

        Self {
            inner: SeaCowInner::Owned { ptr, len, free_fn },
        }
    }

    #[inline]
    #[cfg(feature = "_internal_c_ffi")]
    pub(crate) fn make_owned(&mut self, free_fn: unsafe extern "C" fn(*mut c_void)) {
        if let SeaCowInner::Borrowed(slice) = self.inner {
            self.inner = SeaCowInner::Owned { ptr: slice.as_ptr().cast_mut(), len: slice.len(), free_fn };
        }
    }
}

impl<T: Clone> Clone for SeaCowInner<'_, T> {
    #[inline(never)]
    fn clone(&self) -> Self {
        let slice = match self {
            Self::Borrowed(data) => return Self::Borrowed(data),
            #[cfg(feature = "_internal_c_ffi")]
            Self::Owned { ptr, len, free_fn: _ } => unsafe { slice::from_raw_parts(*ptr, *len) },
            Self::Boxed(data) => &**data,
        };
        let mut v = Vec::new();
        v.try_reserve_exact(slice.len()).unwrap();
        v.extend_from_slice(slice);
        Self::Boxed(v.into_boxed_slice())
    }
}

enum SeaCowInner<'a, T> {
    #[cfg(feature = "_internal_c_ffi")]
    Owned { ptr: *mut T, len: usize, free_fn: unsafe extern "C" fn(*mut c_void) },
    Borrowed(&'a [T]),
    Boxed(Box<[T]>),
}

#[cfg(feature = "_internal_c_ffi")]
impl<T> Drop for SeaCowInner<'_, T> {
    fn drop(&mut self) {
        if let Self::Owned { ptr, free_fn, .. } = self {
            unsafe {
                (free_fn)((*ptr).cast());
            }
        }
    }
}

impl<T> SeaCow<'_, T> {
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        match &self.inner {
            #[cfg(feature = "_internal_c_ffi")]
            SeaCowInner::Owned { ptr, len, .. } => unsafe { slice::from_raw_parts(*ptr, *len) },
            SeaCowInner::Borrowed(a) => a,
            SeaCowInner::Boxed(x) => x,
        }
    }
}

pub(crate) struct RowBitmap<'a, T> {
    rows: &'a [Pointer<T>],
    width: usize,
}
unsafe impl<T: Send + Sync> Send for RowBitmap<'_, T> {}

pub(crate) struct RowBitmapMut<'a, T> {
    rows: MutCow<'a, [PointerMut<T>]>,
    width: usize,
}
unsafe impl<T: Send + Sync> Send for RowBitmapMut<'_, T> {}

impl<T> RowBitmapMut<'_, MaybeUninit<T>> {
    #[inline]
    pub(crate) unsafe fn assume_init<'maybeowned>(&'maybeowned mut self) -> RowBitmap<'maybeowned, T> {
        #[allow(clippy::transmute_ptr_to_ptr)]
        RowBitmap {
            width: self.width,
            rows: mem::transmute::<&'maybeowned [PointerMut<MaybeUninit<T>>], &'maybeowned [Pointer<T>]>(self.rows.borrow_mut()),
        }
    }
}

impl<T> RowBitmap<'_, T> {
    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        let width = self.width;
        self.rows.iter().map(move |row| {
            unsafe { slice::from_raw_parts(row.0, width) }
        })
    }
}

enum MutCow<'a, T: ?Sized> {
    Owned(Box<T>),
    #[allow(dead_code)] /// This is optional, for FFI only
    Borrowed(&'a mut T),
}

impl<T: ?Sized> MutCow<'_, T> {
    #[must_use]
    pub fn borrow_mut(&mut self) -> &mut T {
        match self {
            Self::Owned(a) => a,
            Self::Borrowed(a) => a,
        }
    }
}

impl<'a, T: Sync + Send + Copy + 'static> RowBitmapMut<'a, T> {
    #[inline]
    #[must_use]
    pub fn new_contiguous(data: &mut [T], width: usize) -> Self {
        Self {
            rows: MutCow::Owned(data.chunks_exact_mut(width).map(|r| PointerMut(r.as_mut_ptr())).collect()),
            width,
        }
    }

    /// Inner pointers must be valid for `'a` too, and at least `width` large each
    #[inline]
    #[cfg(feature = "_internal_c_ffi")]
    #[must_use]
    pub unsafe fn new(rows: &'a mut [*mut T], width: usize) -> Self {
        Self {
            rows: MutCow::Borrowed(&mut *(rows as *mut [*mut T] as *mut [PointerMut<T>])),
            width,
        }
    }

    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> + Send {
        let width = self.width;
        self.rows.borrow_mut().iter().map(move |row| {
            unsafe { slice::from_raw_parts_mut(row.0, width) }
        })
    }

    pub(crate) fn chunks(&mut self, chunk_size: usize) -> impl Iterator<Item = RowBitmapMut<'_, T>> {
        self.rows.borrow_mut().chunks_mut(chunk_size).map(|chunk| RowBitmapMut {
            width: self.width,
            rows: MutCow::Borrowed(chunk),
        })
    }

    #[must_use]
    pub(crate) fn len(&mut self) -> usize {
        self.rows.borrow_mut().len()
    }
}
