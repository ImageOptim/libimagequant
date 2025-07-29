//! Exports API for C programs and C-FFI-compatible languages. See `libimagequant.h` or <https://pngquant.org/lib/> for C docs.
//!
//! This crate is not supposed to be used in Rust directly. For Rust, see the parent [imagequant](https://lib.rs/imagequant) crate.
#![cfg_attr(all(not(feature = "std"), feature = "no_std"), no_std)]

#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

#[cfg(all(not(feature = "std"), feature = "no_std"))]
extern crate alloc as std;

use core::ffi::{c_char, c_int, c_uint, c_void};
use core::mem::{ManuallyDrop, MaybeUninit};
use core::{mem, ptr, slice};
use imagequant::capi::*;
use imagequant::Error::LIQ_OK;
use imagequant::*;
use std::ffi::CString;
use std::boxed::Box;

pub use imagequant::Error as liq_error;

#[repr(C)]
pub struct liq_attr {
    magic_header: MagicTag,
    inner: Attributes,
    c_api_free: unsafe extern "C" fn(*mut c_void),
}

#[repr(C)]
pub struct liq_image<'pixels> {
    magic_header: MagicTag,
    inner: ManuallyDrop<Image<'pixels>>,
    c_api_free: unsafe extern "C" fn(*mut c_void),
}

#[repr(C)]
pub struct liq_result {
    magic_header: MagicTag,
    inner: QuantizationResult,
}

#[repr(C)]
pub struct liq_histogram {
    magic_header: MagicTag,
    inner: Histogram,
}

pub type liq_palette = Palette;
pub type liq_histogram_entry = HistogramEntry;
pub type liq_color = RGBA;

pub type liq_log_callback_function = unsafe extern "C" fn(liq: &liq_attr, message: *const c_char, user_info: AnySyncSendPtr);
pub type liq_log_flush_callback_function = unsafe extern "C" fn(liq: &liq_attr, user_info: AnySyncSendPtr);
pub type liq_progress_callback_function = unsafe extern "C" fn(progress_percent: f32, user_info: AnySyncSendPtr) -> c_int;
pub type liq_image_get_rgba_row_callback = unsafe extern "C" fn(row_out: *mut MaybeUninit<RGBA>, row: c_int, width: c_int, user_info: AnySyncSendPtr);

bitflags::bitflags! {
    #[repr(C)]
    #[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone, Copy)]
    pub struct liq_ownership: c_int {
        /// Moves ownership of the rows array. It will free it using `free()` or custom allocator.
        const LIQ_OWN_ROWS = 4;
        /// Moves ownership of the pixel data. It will free it using `free()` or custom allocator.
        const LIQ_OWN_PIXELS = 8;
        /// Makes a copy of the pixels, so the `liq_image` is not tied to pixel's lifetime.
        const LIQ_COPY_PIXELS = 16;
    }
}

#[repr(transparent)]
#[derive(PartialEq, Debug, Copy, Clone)]
pub(crate) struct MagicTag(*const u8);
// Safety: Rust overreacts about C pointers. Data behind this ptr isn't used.
unsafe impl Sync for MagicTag {}
unsafe impl Send for MagicTag {}

pub(crate) static LIQ_ATTR_MAGIC: MagicTag = MagicTag(b"liq_attr_magic\0".as_ptr());
pub(crate) static LIQ_IMAGE_MAGIC: MagicTag = MagicTag(b"liq_image_magic\0".as_ptr());
pub(crate) static LIQ_RESULT_MAGIC: MagicTag = MagicTag(b"liq_result_magic\0".as_ptr());
pub(crate) static LIQ_HISTOGRAM_MAGIC: MagicTag = MagicTag(b"liq_histogram_magic\0".as_ptr());
pub(crate) static LIQ_FREED_MAGIC: MagicTag = MagicTag(b"liq_freed_magic\0".as_ptr());

#[no_mangle]
#[inline(never)]
unsafe extern "C" fn liq_received_invalid_pointer(ptr: *const u8) -> bool {
    if ptr.is_null() {
        return true;
    }
    let _ = ptr::read_volatile(ptr);
    false
}

macro_rules! bad_object {
    ($obj:expr, $tag:expr) => {{
        let obj = &*$obj;
        #[allow(unused_unsafe)]
        #[allow(clippy::ptr_as_ptr)]
        let bork = if cfg!(miri) { false } else { unsafe { liq_received_invalid_pointer((obj as *const _ as *const u8)) } };
        (bork || (($obj).magic_header != $tag))
    }};
}

impl Drop for liq_attr {
    fn drop(&mut self) {
        if bad_object!(self, LIQ_ATTR_MAGIC) { return; }
        self.magic_header = LIQ_FREED_MAGIC;
    }
}
impl Drop for liq_image<'_> {
    fn drop(&mut self) {
        if bad_object!(self, LIQ_IMAGE_MAGIC) { return; }
        unsafe { ManuallyDrop::drop(&mut self.inner); }
        self.magic_header = LIQ_FREED_MAGIC;
    }
}
impl Drop for liq_result {
    fn drop(&mut self) {
        if bad_object!(self, LIQ_RESULT_MAGIC) { return; }
        self.magic_header = LIQ_FREED_MAGIC;
    }
}
impl Drop for liq_histogram {
    fn drop(&mut self) {
        if bad_object!(self, LIQ_HISTOGRAM_MAGIC) { return; }
        self.magic_header = LIQ_FREED_MAGIC;
    }
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_version() -> c_uint {
    LIQ_VERSION
}

#[no_mangle]
#[inline(never)]
#[deprecated]
pub extern "C" fn liq_set_min_opacity(_: &mut liq_attr, _: c_int) -> liq_error {
    LIQ_OK
}

#[no_mangle]
#[inline(never)]
#[deprecated]
pub extern "C" fn liq_get_min_opacity(_: &liq_attr) -> c_int {
    0
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_set_last_index_transparent(attr: &mut liq_attr, is_last: c_int) {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return; }
    attr.inner.set_last_index_transparent(is_last != 0);
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_palette(result: &mut liq_result) -> Option<&liq_palette> {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return None; }
    Some(liq_get_palette_impl(&mut result.inner))
}

/// A `void*` pointer to any data, as long as it's thread-safe
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct AnySyncSendPtr(pub *mut c_void);

impl Default for AnySyncSendPtr {
    fn default() -> Self {
        Self(ptr::null_mut())
    }
}

/// C callback user is responsible for ensuring safety
unsafe impl Send for AnySyncSendPtr {}
unsafe impl Sync for AnySyncSendPtr {}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_attr_set_progress_callback(attr: &mut liq_attr, callback: liq_progress_callback_function, user_info: AnySyncSendPtr) {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return; }
    let cb = move |f| if callback(f, user_info) == 0 { ControlFlow::Break} else { ControlFlow::Continue};
    attr.inner.set_progress_callback(cb);
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_result_set_progress_callback(result: &mut liq_result, callback: liq_progress_callback_function, user_info: AnySyncSendPtr) {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return; }
    result.inner.set_progress_callback(move |f| if callback(f, user_info) == 0 { ControlFlow::Break} else { ControlFlow::Continue});
}

#[allow(clippy::cast_ptr_alignment)]
unsafe fn attr_to_liq_attr_ptr(ptr: &Attributes) -> &liq_attr {
    let liq_attr = ptr::NonNull::<liq_attr>::dangling();
    let outer_addr = ptr::addr_of!(*liq_attr.as_ptr()) as isize;
    let inner_addr = ptr::addr_of!((*liq_attr.as_ptr()).inner) as isize;

    &*(ptr as *const Attributes).cast::<u8>().offset(outer_addr - inner_addr).cast::<liq_attr>()
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_set_log_callback(attr: &mut liq_attr, callback: liq_log_callback_function, user_info: AnySyncSendPtr) {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return; }
    attr.inner.set_log_callback(move |attr, msg| {
        if let Ok(tmp) = CString::new(msg) {
            callback(attr_to_liq_attr_ptr(attr), tmp.as_ptr(), user_info);
        }
    });
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_set_log_flush_callback(attr: &mut liq_attr, callback: liq_log_flush_callback_function, user_info: AnySyncSendPtr) {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return; }
    attr.inner.set_log_flush_callback(move |attr| callback(attr_to_liq_attr_ptr(attr), user_info));
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_set_max_colors(attr: &mut liq_attr, colors: c_uint) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return Error::InvalidPointer; }
    attr.inner.set_max_colors(colors).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_max_colors(attr: &liq_attr) -> c_uint {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return !0; }
    attr.inner.max_colors()
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_set_min_posterization(attr: &mut liq_attr, bits: c_int) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return Error::InvalidPointer; }
    attr.inner.set_min_posterization(bits as u8).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_min_posterization(attr: &liq_attr) -> c_uint {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return !0; }
    attr.inner.min_posterization().into()
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_set_speed(attr: &mut liq_attr, speed: c_int) -> liq_error {
    attr.inner.set_speed(speed).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_speed(attr: &liq_attr) -> c_uint {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return !0; }
    attr.inner.speed()
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_set_quality(attr: &mut liq_attr, minimum: c_uint, target: c_uint) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return Error::InvalidPointer; }
    attr.inner.set_quality(minimum as u8, target as u8).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_min_quality(attr: &liq_attr) -> c_uint {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return !0; }
    attr.inner.quality().0.into()
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_max_quality(attr: &liq_attr) -> c_uint {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return !0; }
    attr.inner.quality().1.into()
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_quantize_image(attr: &mut liq_attr, img: &mut liq_image) -> Option<Box<liq_result>> {
    if bad_object!(attr, LIQ_ATTR_MAGIC) ||
       bad_object!(img, LIQ_IMAGE_MAGIC) { return None; }
    let img = &mut img.inner;
    let attr = &mut attr.inner;

    attr.quantize(img).ok().map(|inner| Box::new(liq_result {
        magic_header: LIQ_RESULT_MAGIC,
        inner,
    }))
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_write_remapped_image(result: &mut liq_result, input_image: &mut liq_image, buffer_bytes: *mut MaybeUninit<u8>, buffer_size: usize) -> liq_error {
    if bad_object!(result, LIQ_RESULT_MAGIC) ||
       bad_object!(input_image, LIQ_IMAGE_MAGIC) { return Error::InvalidPointer; }
    let input_image = &mut input_image.inner;
    let result = &mut result.inner;

    if liq_received_invalid_pointer(buffer_bytes.cast()) { return Error::InvalidPointer; }

    let required_size = (input_image.width()) * (input_image.height());
    if buffer_size < required_size { return Error::BufferTooSmall; }
    let buffer_bytes = slice::from_raw_parts_mut(buffer_bytes, required_size);
    liq_write_remapped_image_impl(result, input_image, buffer_bytes).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_write_remapped_image_rows(result: &mut liq_result, input_image: &mut liq_image, row_pointers: *mut *mut MaybeUninit<u8>) -> liq_error {
    if bad_object!(result, LIQ_RESULT_MAGIC) ||
       bad_object!(input_image, LIQ_IMAGE_MAGIC) { return Error::InvalidPointer; }
    let input_image = &mut input_image.inner;
    let result = &mut result.inner;

    if liq_received_invalid_pointer(row_pointers.cast()) { return Error::InvalidPointer; }
    let rows = slice::from_raw_parts_mut(row_pointers, input_image.height());

    liq_write_remapped_image_rows_impl(result, input_image, rows).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_image_add_fixed_color(img: &mut liq_image, color: liq_color) -> liq_error {
    if bad_object!(img, LIQ_IMAGE_MAGIC) { return Error::InvalidPointer; }
    img.inner.add_fixed_color(color).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_histogram_add_fixed_color(hist: &mut liq_histogram, color: liq_color, gamma: f64) -> liq_error {
    if bad_object!(hist, LIQ_HISTOGRAM_MAGIC) { return Error::InvalidPointer; }
    let hist = &mut hist.inner;

    hist.add_fixed_color(color, gamma).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_image_get_width(img: &liq_image) -> c_uint {
    if bad_object!(img, LIQ_IMAGE_MAGIC) { return !0; }
    img.inner.width() as _
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_image_get_height(img: &liq_image) -> c_uint {
    if bad_object!(img, LIQ_IMAGE_MAGIC) { return !0; }
    img.inner.height() as _
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_image_destroy(_: Option<Box<liq_image>>) {}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_image_set_background<'pixels>(img: &mut liq_image<'pixels>, background: Box<liq_image<'pixels>>) -> liq_error {
    if bad_object!(img, LIQ_IMAGE_MAGIC) ||
       bad_object!(background, LIQ_IMAGE_MAGIC) { return Error::InvalidPointer; }
    let background = unsafe { ManuallyDrop::take(&mut ManuallyDrop::new(background).inner) };
    img.inner.set_background(background).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_image_set_importance_map(img: &mut liq_image, importance_map: *mut u8, buffer_size: usize, ownership: liq_ownership) -> liq_error {
    if bad_object!(img, LIQ_IMAGE_MAGIC) { return Error::InvalidPointer; }
    let free_fn = img.c_api_free;
    let img = &mut img.inner;

    if buffer_size == 0 || liq_received_invalid_pointer(importance_map) { return Error::InvalidPointer; }
    let required_size = img.width() * img.height();
    if buffer_size < required_size {
        return Error::BufferTooSmall;
    }

    let importance_map_slice = slice::from_raw_parts(importance_map, required_size);
    if ownership == liq_ownership::LIQ_COPY_PIXELS {
        img.set_importance_map(importance_map_slice).err().unwrap_or(LIQ_OK)
    } else if ownership == liq_ownership::LIQ_OWN_PIXELS {
        let copy: Box<[u8]> = importance_map_slice.into();
        free_fn(importance_map.cast());
        img.set_importance_map(copy).err().unwrap_or(LIQ_OK);
        LIQ_OK
    } else {
        Error::Unsupported
    }
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_image_set_memory_ownership(img: &mut liq_image, ownership_flags: liq_ownership) -> liq_error {
    if bad_object!(img, LIQ_IMAGE_MAGIC) { return Error::InvalidPointer; }

    let both = liq_ownership::LIQ_OWN_ROWS | liq_ownership::LIQ_OWN_PIXELS;

    if ownership_flags.is_empty() || (ownership_flags | both) != both {
        return Error::ValueOutOfRange;
    }

    let own_rows = ownership_flags.contains(liq_ownership::LIQ_OWN_ROWS);
    let own_pixels = ownership_flags.contains(liq_ownership::LIQ_OWN_PIXELS);
    liq_image_set_memory_ownership_impl(&mut img.inner, own_rows, own_pixels, img.c_api_free).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_histogram_create(attr: &liq_attr) -> Option<Box<liq_histogram>> {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return None; }
    Some(Box::new(liq_histogram {
        magic_header: LIQ_HISTOGRAM_MAGIC,
        inner: Histogram::new(&attr.inner),
    }))
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_histogram_destroy(_hist: Option<Box<liq_histogram>>) {}

#[no_mangle]
#[inline(never)]
#[deprecated(note = "custom allocators are no longer supported")]
pub extern "C" fn liq_attr_create_with_allocator(_unused: *mut c_void, free: unsafe extern "C" fn(*mut c_void)) -> Option<Box<liq_attr>> {
    let attr = Box::new(liq_attr {
        magic_header: LIQ_ATTR_MAGIC,
        inner: Attributes::new(),
        c_api_free: free,
    });
    debug_assert_eq!(ptr::addr_of!(*attr), unsafe { attr_to_liq_attr_ptr(&attr.inner) } as *const liq_attr);
    Some(attr)
}

#[no_mangle]
#[inline(never)]
#[allow(deprecated)]
#[cfg_attr(all(feature = "std", feature = "no_std"), deprecated(note = "Cargo features configuration issue: both std and no_std features are enabled in imagequant-sys\nYou must disable default features to use no_std."))]
pub extern "C" fn liq_attr_create() -> Option<Box<liq_attr>> {
    liq_attr_create_with_allocator(ptr::null_mut(), libc::free)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_attr_copy(attr: &liq_attr) -> Option<Box<liq_attr>> {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return None; }
    Some(Box::new(liq_attr {
        magic_header: LIQ_ATTR_MAGIC,
        inner: attr.inner.clone(),
        c_api_free: attr.c_api_free,
    }))
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_attr_destroy(_attr: Option<Box<liq_attr>>) {}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_result_destroy(_res: Option<Box<liq_result>>) {}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_set_output_gamma(result: &mut liq_result, gamma: f64) -> liq_error {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return Error::InvalidPointer; }
    result.inner.set_output_gamma(gamma).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_set_dithering_level(result: &mut liq_result, dither_level: f32) -> liq_error {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return Error::InvalidPointer; }
    result.inner.set_dithering_level(dither_level).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_output_gamma(result: &liq_result) -> f64 {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return -1.; }
    result.inner.output_gamma()
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_quantization_error(result: &liq_result) -> f64 {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return -1.; }
    result.inner.quantization_error().unwrap_or(-1.)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_remapping_error(result: &liq_result) -> f64 {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return -1.; }
    result.inner.remapping_error().unwrap_or(-1.)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_quantization_quality(result: &liq_result) -> c_int {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return -1; }
    result.inner.quantization_quality().map_or(-1, c_int::from)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_remapping_quality(result: &liq_result) -> c_int {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return -1; }
    result.inner.remapping_quality().map_or(-1, c_int::from)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_image_quantize(img: &mut liq_image, attr: &mut liq_attr, write_only_output: &mut MaybeUninit<Option<Box<liq_result>>>) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) ||
       bad_object!(img, LIQ_IMAGE_MAGIC) { return Error::InvalidPointer; }
    let attr = &mut attr.inner;
    let img = &mut img.inner;

    let res = attr.quantize(img)
        .map(|inner| liq_result {
            magic_header: LIQ_RESULT_MAGIC,
            inner,
        });
    store_boxed_result(res, write_only_output)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_histogram_quantize(hist: &mut liq_histogram, attr: &liq_attr, write_only_output: &mut MaybeUninit<Option<Box<liq_result>>>) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) ||
       bad_object!(hist, LIQ_HISTOGRAM_MAGIC) { return Error::InvalidPointer; }
    let attr = &attr.inner;
    let hist = &mut hist.inner;

    let res = hist.quantize(attr)
        .map(|inner| liq_result {
            magic_header: LIQ_RESULT_MAGIC,
            inner,
        });
    store_boxed_result(res, write_only_output)
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_result_from_palette(
    attr: &liq_attr,
    palette: *const RGBA,
    palette_size: c_uint,
    gamma: f64,
    write_only_output: &mut MaybeUninit<Option<Box<liq_result>>>,
) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) {
        return Error::InvalidPointer;
    }
    let Ok(palette_size) = palette_size.try_into() else {
        return Error::ValueOutOfRange;
    };
    if liq_received_invalid_pointer(palette.cast()) {
        return Error::InvalidPointer;
    }

    let attr = &attr.inner;
    let palette = slice::from_raw_parts(palette, palette_size);

    let res = QuantizationResult::from_palette(attr, palette, gamma).map(|inner| liq_result {
        magic_header: LIQ_RESULT_MAGIC,
        inner,
    });
    store_boxed_result(res, write_only_output)
}

#[inline]
fn store_boxed_result<T>(res: Result<T, liq_error>, out: &mut MaybeUninit<Option<Box<T>>>) -> liq_error {
    match res {
        Ok(res) => { out.write(Some(Box::new(res))); LIQ_OK },
        Err(err) => { out.write(None); err },
    }
}

pub(crate) fn check_image_size(attr: &liq_attr, width: u32, height: u32) -> bool {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return false; }

    if width == 0 || height == 0 {
        return false;
    }

    if width as usize > c_int::MAX as usize / mem::size_of::<liq_color>() / height as usize ||
       width as usize > c_int::MAX as usize / 16 / mem::size_of::<f32>() ||
       height as usize > c_int::MAX as usize / mem::size_of::<usize>()
    {
        return false;
    }
    true
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_image_create_custom(attr: &liq_attr, row_callback: liq_image_get_rgba_row_callback, user_info: AnySyncSendPtr, width: c_uint, height: c_uint, gamma: f64)
 -> Option<Box<liq_image<'static>>> {
    let cb: Box<dyn Fn(&mut [MaybeUninit<RGBA>], usize) + Send + Sync> = Box::new(move |row, y| row_callback(row.as_mut_ptr(), y as _, row.len() as _, user_info));
    liq_image_create_custom_impl(&attr.inner, cb, width as _, height as _, gamma)
        .map(move |inner| Box::new(liq_image {
            magic_header: LIQ_IMAGE_MAGIC,
            inner: ManuallyDrop::new(inner),
            c_api_free: attr.c_api_free,
        }))
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_image_create_rgba_rows<'rows>(attr: &liq_attr, rows: *const *const RGBA, width: c_uint, height: c_uint, gamma: f64) -> Option<Box<liq_image<'rows>>> {
    if !check_image_size(attr, width, height) { return None; }
    if rows.is_null() { return None; }
    let rows = slice::from_raw_parts(rows, height as _);
    liq_image_create_rgba_rows_impl(&attr.inner, rows, width as _,  height as _, gamma)
        .map(move |inner| Box::new(liq_image {
            magic_header: LIQ_IMAGE_MAGIC,
            inner: ManuallyDrop::new(inner),
            c_api_free: attr.c_api_free,
        }))
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_image_create_rgba<'pixels>(attr: &liq_attr, pixels: *const liq_color, width: c_uint, height: c_uint, gamma: f64) -> Option<Box<liq_image<'pixels>>> {
    if liq_received_invalid_pointer(pixels.cast()) { return None; }
    if !check_image_size(attr, width, height) { return None; }
    let rows = (0..height as usize).map(move |i| pixels.add(width as usize * i)).collect();
    liq_image_create_rgba_bitmap_impl(&attr.inner, rows, width as _, height as _, gamma)
        .map(move |inner| Box::new(liq_image {
            magic_header: LIQ_IMAGE_MAGIC,
            inner: ManuallyDrop::new(inner),
            c_api_free: attr.c_api_free,
        }))
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_histogram_add_colors(input_hist: &mut liq_histogram, attr: &liq_attr, entries: *const HistogramEntry, num_entries: c_int, gamma: f64) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) ||
       bad_object!(input_hist, LIQ_HISTOGRAM_MAGIC) { return Error::InvalidPointer; }
    let input_hist = &mut input_hist.inner;
    if num_entries == 0 {
        return LIQ_OK;
    }

    let Ok(num_entries) = num_entries.try_into() else {
        return Error::ValueOutOfRange;
    };

    if liq_received_invalid_pointer(entries.cast()) { return Error::InvalidPointer; }

    let entries = slice::from_raw_parts(entries, num_entries);

    input_hist.add_colors(entries, gamma).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_histogram_add_image(input_hist: &mut liq_histogram, attr: &liq_attr, input_image: &mut liq_image) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) ||
       bad_object!(input_hist, LIQ_HISTOGRAM_MAGIC) ||
       bad_object!(input_image, LIQ_IMAGE_MAGIC) { return Error::InvalidPointer; }
    let attr = &attr.inner;
    let input_hist = &mut input_hist.inner;
    let input_image = &mut input_image.inner;

    input_hist.add_image(attr, input_image).err().unwrap_or(LIQ_OK)
}

/// This is just to exist in backtraces of crashes that aren't mine
#[no_mangle]
#[inline(never)]
pub unsafe extern "Rust" fn liq_executing_user_callback(callback: liq_image_get_rgba_row_callback, temp_row: &mut [MaybeUninit<liq_color>], row: usize, user_info: AnySyncSendPtr) {
    callback(temp_row.as_mut_ptr(), row as _, temp_row.len() as _, user_info);
}

#[test]
fn links_and_runs() {
    unsafe {
        assert!(liq_version() >= 40000);
        let attr = liq_attr_create().unwrap();
        let mut hist = liq_histogram_create(&attr).unwrap();
        assert_eq!(LIQ_OK, liq_histogram_add_fixed_color(&mut hist, liq_color {r: 0, g: 0, b: 0, a: 0}, 0.));
        liq_histogram_add_colors(&mut hist, &attr, ptr::null(), 0, 0.);

        let mut res = MaybeUninit::uninit();

        // this is fine, because there is 1 fixed color to generate
        assert_eq!(LIQ_OK, liq_histogram_quantize(&mut hist, &attr, &mut res));
        let res = res.assume_init().unwrap();

        liq_result_destroy(Some(res));
        liq_histogram_destroy(Some(hist));
        liq_attr_destroy(Some(attr));
    }
}

#[test]
#[allow(deprecated)]
fn link_every_symbol() {

    let x = liq_attr_create as *const c_void as usize
        + liq_attr_create_with_allocator as *const c_void as usize
        + liq_attr_copy as *const c_void as usize
        + liq_attr_destroy as *const c_void as usize
        + liq_set_max_colors as *const c_void as usize
        + liq_get_max_colors as *const c_void as usize
        + liq_set_speed as *const c_void as usize
        + liq_get_speed as *const c_void as usize
        + liq_set_min_posterization as *const c_void as usize
        + liq_get_min_posterization as *const c_void as usize
        + liq_set_quality as *const c_void as usize
        + liq_get_min_quality as *const c_void as usize
        + liq_get_max_quality as *const c_void as usize
        + liq_set_last_index_transparent as *const c_void as usize
        + liq_image_create_rgba_rows as *const c_void as usize
        + liq_image_create_rgba as *const c_void as usize
        + liq_image_set_memory_ownership as *const c_void as usize
        + liq_set_log_callback as *const c_void as usize
        + liq_set_log_flush_callback as *const c_void as usize
        + liq_attr_set_progress_callback as *const c_void as usize
        + liq_result_set_progress_callback as *const c_void as usize
        + liq_image_create_custom as *const c_void as usize
        + liq_image_set_background as *const c_void as usize
        + liq_image_set_importance_map as *const c_void as usize
        + liq_image_add_fixed_color as *const c_void as usize
        + liq_image_get_width as *const c_void as usize
        + liq_image_get_height as *const c_void as usize
        + liq_image_destroy as *const c_void as usize
        + liq_histogram_create as *const c_void as usize
        + liq_histogram_add_image as *const c_void as usize
        + liq_histogram_add_colors as *const c_void as usize
        + liq_histogram_add_fixed_color as *const c_void as usize
        + liq_histogram_destroy as *const c_void as usize
        + liq_quantize_image as *const c_void as usize
        + liq_histogram_quantize as *const c_void as usize
        + liq_image_quantize as *const c_void as usize
        + liq_result_from_palette as *const c_void as usize
        + liq_set_dithering_level as *const c_void as usize
        + liq_set_output_gamma as *const c_void as usize
        + liq_get_output_gamma as *const c_void as usize
        + liq_get_palette as *const c_void as usize
        + liq_write_remapped_image as *const c_void as usize
        + liq_write_remapped_image_rows as *const c_void as usize
        + liq_get_quantization_error as *const c_void as usize
        + liq_get_quantization_quality as *const c_void as usize
        + liq_result_destroy as *const c_void as usize
        + liq_get_remapping_error as *const c_void as usize
        + liq_get_remapping_quality as *const c_void as usize
        + liq_version as *const c_void as usize;
    assert_ne!(!0, x);
}

#[test]
fn c_callback_test_c() {
    let mut called = 0;
    let mut res = unsafe {
        let mut a = liq_attr_create().unwrap();
        unsafe extern "C" fn get_row(output_row: *mut MaybeUninit<RGBA>, y: c_int, width: c_int, user_data: AnySyncSendPtr) {
            assert!((0..5).contains(&y));
            assert_eq!(123, width);
            for i in 0..width as isize {
                let n = i as u8;
                (*output_row.offset(i)).write(RGBA::new(n, n, n, n));
            }
            let user_data = user_data.0.cast::<i32>();
            *user_data += 1;
        }
        let mut img = liq_image_create_custom(&a, get_row, AnySyncSendPtr(ptr::addr_of_mut!(called).cast::<c_void>()), 123, 5, 0.).unwrap();
        liq_quantize_image(&mut a, &mut img).unwrap()
    };
    assert!(called > 5 && called < 50);
    let pal = liq_get_palette(&mut res).unwrap();
    assert_eq!(123, pal.count);
}

#[test]
fn ownership_bitflags() {
    assert_eq!(4 + 16, (liq_ownership::LIQ_OWN_ROWS | liq_ownership::LIQ_COPY_PIXELS).bits());
}

#[cfg(all(feature = "no_std_global_handlers", not(feature = "std"), feature = "no_std"))]
#[cfg(not(test))]
mod no_std_global_handlers {
    use std::alloc::{GlobalAlloc, Layout};

    #[cfg(panic = "unwind")]
    compile_error!("no_std imagequant-sys must be compiled with panic=abort\n\nset env var CARGO_PROFILE_DEV_PANIC=abort and CARGO_PROFILE_RELEASE_PANIC=abort\nor modify your Cargo.toml to add `panic=\"abort\"` to `[profile.release]` and `[profile.dev]`");

    #[panic_handler]
    fn panic(_info: &core::panic::PanicInfo) -> ! {
        unsafe { libc::abort(); }
    }

    #[global_allocator]
    static GLOBAL_ALLOCATOR: Mallocator = Mallocator;

    #[derive(Default)]
    pub struct Mallocator;

    unsafe impl GlobalAlloc for Mallocator {
         unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
             libc::malloc(layout.size() as _).cast()
         }

         unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
             libc::free(ptr.cast());
         }
    }
}
