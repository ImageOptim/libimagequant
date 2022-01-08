//! Exports API for C programs and C-FFI-compatible languages. See `libimagequant.h` or <https://pngquant.org/lib/> for C docs.

#![allow(clippy::missing_safety_doc)]

use crate::attr::*;
use crate::error::*;
use crate::hist::*;
use crate::image::*;
use crate::pal::*;
use crate::quant::*;
use crate::rows::PixelsSource;
use crate::seacow::*;
use std::ffi::CString;
use std::mem::MaybeUninit;
use std::os::raw::c_char;
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;

pub type liq_attr = crate::Attributes;
pub type liq_palette = crate::Palette;
pub type liq_image<'pixels, 'rows> = crate::image::Image<'pixels, 'rows>;
pub type liq_histogram_entry = HistogramEntry;
pub type liq_color = RGBA;
pub type liq_result = QuantizationResult;

pub type liq_log_callback_function = unsafe extern "C" fn(liq: &liq_attr, message: *const c_char, user_info: *mut c_void);
pub type liq_log_flush_callback_function = unsafe extern "C" fn(liq: &liq_attr, user_info: *mut c_void);
pub type liq_progress_callback_function = unsafe extern "C" fn(progress_percent: f32, user_info: *mut c_void) -> c_int;
pub type liq_image_get_rgba_row_callback = unsafe extern "C" fn(row_out: *mut MaybeUninit<RGBA>, row: c_int, width: c_int, user_info: *mut c_void);

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
unsafe fn liq_received_invalid_pointer(ptr: *const u8) -> bool {
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
        let bork = if cfg!(miri) { false } else { unsafe { liq_received_invalid_pointer((obj as *const _ as *const u8)) } };
        (bork || (($obj).magic_header != $tag))
    }};
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_version() -> c_uint {
    crate::LIQ_VERSION
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
    attr.set_last_index_transparent(is_last != 0);
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_palette(result: &mut liq_result) -> Option<&liq_palette> {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return None; }
    Some(result.int_palette())
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
    attr.set_progress_callback(move |f| if callback(f, user_info.0) == 0 { ControlFlow::Break} else { ControlFlow::Continue});
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_result_set_progress_callback(result: &mut liq_result, callback: liq_progress_callback_function, user_info: AnySyncSendPtr) {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return; }
    result.set_progress_callback(move |f| if callback(f, user_info.0) == 0 { ControlFlow::Break} else { ControlFlow::Continue});
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_set_log_callback(attr: &mut liq_attr, callback: liq_log_callback_function, user_info: AnySyncSendPtr) {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return; }
    attr.verbose_printf_flush();
    attr.set_log_callback(move |attr, msg| {
        if let Ok(tmp) = CString::new(msg) {
            callback(attr, tmp.as_ptr(), user_info.0)
        }
    });
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_set_log_flush_callback(attr: &mut liq_attr, callback: liq_log_flush_callback_function, user_info: AnySyncSendPtr) {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return; }
    attr.set_log_flush_callback(move |attr| callback(attr, user_info.0));
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_set_max_colors(attr: &mut liq_attr, colors: c_uint) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return LIQ_INVALID_POINTER; }
    attr.set_max_colors(colors)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_max_colors(attr: &liq_attr) -> c_uint {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return !0; }
    attr.max_colors()
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_set_min_posterization(attr: &mut liq_attr, bits: c_int) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return LIQ_INVALID_POINTER; }
    attr.set_min_posterization(bits as u8)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_min_posterization(attr: &liq_attr) -> c_uint {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return !0; }
    attr.min_posterization().into()
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_set_speed(attr: &mut liq_attr, speed: c_int) -> liq_error {
    attr.set_speed(speed)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_speed(attr: &liq_attr) -> c_uint {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return !0; }
    attr.speed()
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_set_quality(attr: &mut liq_attr, minimum: c_uint, target: c_uint) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return LIQ_INVALID_POINTER; }
    attr.set_quality(minimum as u8, target as u8)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_min_quality(attr: &liq_attr) -> c_uint {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return !0; }
    attr.quality().0.into()
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_max_quality(attr: &liq_attr) -> c_uint {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return !0; }
    attr.quality().1.into()
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_quantize_image(attr: &mut liq_attr, img: &mut Image) -> Option<Box<liq_result>> {
    if bad_object!(attr, LIQ_ATTR_MAGIC) ||
       bad_object!(img, LIQ_IMAGE_MAGIC) { return None; }

    let mut hist = Histogram::new(attr);
    hist.add_image(attr, img).ok()?;
    hist.quantize_internal(attr, false).ok().map(Box::new)
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_write_remapped_image(result: &mut liq_result, input_image: &mut Image, buffer_bytes: *mut MaybeUninit<u8>, buffer_size: usize) -> liq_error {
    if bad_object!(result, LIQ_RESULT_MAGIC) ||
       bad_object!(input_image, LIQ_IMAGE_MAGIC) { return LIQ_INVALID_POINTER; }

    if liq_received_invalid_pointer(buffer_bytes.cast()) { return LIQ_INVALID_POINTER; }

    let required_size = (input_image.width()) * (input_image.height());
    if buffer_size < required_size { return LIQ_BUFFER_TOO_SMALL; }
    let buffer_bytes = std::slice::from_raw_parts_mut(buffer_bytes, required_size);

    let rows = RowBitmapMut::new_contiguous(buffer_bytes, input_image.width());
    result.write_remapped_image_rows_internal(input_image, rows).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_write_remapped_image_rows(result: &mut liq_result, input_image: &mut Image, row_pointers: *mut *mut MaybeUninit<u8>) -> liq_error {
    if bad_object!(result, LIQ_RESULT_MAGIC) ||
       bad_object!(input_image, LIQ_IMAGE_MAGIC) { return LIQ_INVALID_POINTER; }

    if liq_received_invalid_pointer(row_pointers.cast()) { return LIQ_INVALID_POINTER; }

    let rows = std::slice::from_raw_parts_mut(row_pointers, input_image.height());
    let rows = RowBitmapMut::new(rows, input_image.width());

    result.write_remapped_image_rows_internal(input_image, rows).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_image_add_fixed_color(img: &mut Image, color: liq_color) -> liq_error {
    if bad_object!(img, LIQ_IMAGE_MAGIC) { return LIQ_INVALID_POINTER; }
    img.add_fixed_color(color)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_histogram_add_fixed_color(hist: &mut Histogram, color: liq_color, gamma: f64) -> liq_error {
    if bad_object!(hist, LIQ_HISTOGRAM_MAGIC) { return LIQ_INVALID_POINTER; }
    hist.add_fixed_color(color, gamma)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_image_get_width(img: &Image) -> c_uint {
    if bad_object!(img, LIQ_IMAGE_MAGIC) { return !0; }
    img.px.width
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_image_get_height(img: &Image) -> c_uint {
    if bad_object!(img, LIQ_IMAGE_MAGIC) { return !0; }
    img.px.height
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_image_destroy(_: Option<Box<Image>>) {}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_image_set_background<'pixels, 'rows>(img: &mut Image<'pixels, 'rows>, background: Box<Image<'pixels, 'rows>>) -> liq_error {
    if bad_object!(img, LIQ_IMAGE_MAGIC) ||
       bad_object!(background, LIQ_IMAGE_MAGIC) { return LIQ_INVALID_POINTER; }

    img.set_background(*background).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_image_set_importance_map(img: &mut Image, importance_map: *mut u8, buffer_size: usize, ownership: liq_ownership) -> liq_error {
    if bad_object!(img, LIQ_IMAGE_MAGIC) { return LIQ_INVALID_POINTER; }
    let buf = if buffer_size > 0 {
        if liq_received_invalid_pointer(importance_map) { return LIQ_INVALID_POINTER; }
        let required_size = img.width() * img.height();
        if buffer_size < required_size {
            return LIQ_BUFFER_TOO_SMALL;
        }

        let importance_map = std::slice::from_raw_parts_mut(importance_map, required_size);
        Some(if ownership == liq_ownership::LIQ_COPY_PIXELS {
            SeaCow::boxed(importance_map[..].into())
        } else if ownership == liq_ownership::LIQ_OWN_PIXELS {
            SeaCow::c_owned(importance_map.as_mut_ptr(), importance_map.len(), img.c_api_free.unwrap_or(libc::free))
        } else {
            return LIQ_UNSUPPORTED;
        })
    } else {
        None
    };

    img.set_importance_map_raw(buf);
    LIQ_OK
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_image_set_memory_ownership(img: &mut Image, ownership_flags: liq_ownership) -> liq_error {
    if bad_object!(img, LIQ_IMAGE_MAGIC) { return LIQ_INVALID_POINTER; }
    img.px.set_memory_ownership(ownership_flags, img.c_api_free.unwrap_or(libc::free)).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_histogram_create(attr: &liq_attr) -> Option<Box<Histogram>> {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return None; }
    Some(Box::new(Histogram::new(attr)))
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_histogram_destroy(_hist: Option<Box<Histogram>>) {}

#[no_mangle]
#[inline(never)]
#[deprecated(note = "custom allocators are no longer supported")]
pub extern "C" fn liq_attr_create_with_allocator(_unused: *mut c_void, free: Option<unsafe extern fn(*mut c_void)>) -> Option<Box<liq_attr>> {
    Some(Box::new(Attributes::with_free(free)))
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_attr_create() -> Option<Box<liq_attr>> {
    Some(Box::new(Attributes::new()))
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_attr_copy(attr: &liq_attr) -> Option<Box<liq_attr>> {
    if bad_object!(attr, LIQ_ATTR_MAGIC) { return None; }
    Some(Box::new(attr.clone()))
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
    if bad_object!(result, LIQ_RESULT_MAGIC) { return LIQ_INVALID_POINTER; }
    result.set_output_gamma(gamma)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_set_dithering_level(result: &mut liq_result, dither_level: f32) -> liq_error {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return LIQ_INVALID_POINTER; }
    result.set_dithering_level(dither_level)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_output_gamma(result: &liq_result) -> f64 {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return -1.; }
    result.output_gamma()
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_quantization_error(result: &liq_result) -> f64 {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return -1.; }
    result.quantization_error().unwrap_or(-1.)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_remapping_error(result: &liq_result) -> f64 {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return -1.; }
    result.remapping_error().unwrap_or(-1.)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_quantization_quality(result: &liq_result) -> c_int {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return -1; }
    result.quantization_quality().map(c_int::from).unwrap_or(-1)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_get_remapping_quality(result: &liq_result) -> c_int {
    if bad_object!(result, LIQ_RESULT_MAGIC) { return -1; }
    result.remapping_quality().map(c_int::from).unwrap_or(-1)
}

#[no_mangle]
#[inline(never)]
pub fn liq_image_quantize(img: &mut Image, attr: &liq_attr, write_only_output: &mut MaybeUninit<Option<Box<liq_result>>>) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) ||
       bad_object!(img, LIQ_IMAGE_MAGIC) { return LIQ_INVALID_POINTER; }

    let mut hist = Histogram::new(attr);
    let res = hist.add_image(attr, img).and_then(|_| hist.quantize_internal(attr, false));
    store_boxed_result(res, write_only_output)
}

#[no_mangle]
#[inline(never)]
pub fn liq_histogram_quantize(hist: &mut Histogram, attr: &liq_attr, write_only_output: &mut MaybeUninit<Option<Box<liq_result>>>) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) ||
       bad_object!(hist, LIQ_HISTOGRAM_MAGIC) { return LIQ_INVALID_POINTER; }

    store_boxed_result(hist.quantize_internal(attr, true), write_only_output)
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
        attr.verbose_print("  error: width and height must be > 0");
        return false;
    }

    if width as usize > c_int::MAX as usize / std::mem::size_of::<liq_color>() / height as usize ||
       width as usize > c_int::MAX as usize / 16 / std::mem::size_of::<f_pixel>() ||
       height as usize > c_int::MAX as usize / std::mem::size_of::<usize>()
    {
        attr.verbose_print("  error: image too large");
        return false;
    }
    true
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_image_create_custom(attr: &liq_attr, row_callback: liq_image_get_rgba_row_callback, user_info: AnySyncSendPtr, width: c_uint, height: c_uint, gamma: f64)
 -> Option<Box<Image<'static, 'static>>> {
    let db: Box<dyn Fn(&mut [MaybeUninit<RGBA>], usize) + Send + Sync> = Box::new(move |row, y| row_callback(row.as_mut_ptr(), y as _, row.len() as _, user_info.0));
    liq_image::new_internal(attr, PixelsSource::Callback(db), width, height, gamma).ok().map(Box::new)
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_image_create_rgba_rows<'rows>(attr: &liq_attr, rows: *const *const u8, width: c_uint, height: c_uint, gamma: f64) -> Option<Box<Image<'rows, 'static>>> {
    if !check_image_size(attr, width, height) { return None; }
    if rows.is_null() { return None; }
    let rows = std::slice::from_raw_parts(rows as *const *const liq_color, height as _);
    let rows = SeaCow::borrowed(rows);
    let rows_slice = rows.as_slice();
    if rows_slice.iter().any(|r| r.is_null()) {
        return None;
    }
    liq_image::new_internal(attr, PixelsSource::Pixels { rows, pixels: None }, width, height, gamma).ok().map(Box::new)
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_image_create_rgba(attr: &liq_attr, bitmap: *const liq_color, width: c_uint, height: c_uint, gamma: f64) -> Option<Box<Image>> {
    if liq_received_invalid_pointer(bitmap.cast()) { return None; }
    if !check_image_size(attr, width, height) { return None; }

    let rows = SeaCow::boxed((0..height as usize).map(move |i| bitmap.add(width as usize * i)).collect());
    liq_image::new_internal(attr, PixelsSource::Pixels { rows, pixels: None }, width, height, gamma).ok().map(Box::new)
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn liq_histogram_add_colors(input_hist: &mut Histogram, attr: &liq_attr, entries: *const HistogramEntry, num_entries: c_int, gamma: f64) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) ||
       bad_object!(input_hist, LIQ_HISTOGRAM_MAGIC) { return LIQ_INVALID_POINTER; }

    if num_entries == 0 {
        return LIQ_OK;
    }

    if liq_received_invalid_pointer(entries.cast()) { return LIQ_INVALID_POINTER; }

    let entries = std::slice::from_raw_parts(entries, num_entries as usize);

    input_hist.add_colors(entries, gamma).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn liq_histogram_add_image(input_hist: &mut Histogram, attr: &liq_attr, input_image: &mut Image) -> liq_error {
    if bad_object!(attr, LIQ_ATTR_MAGIC) ||
       bad_object!(input_hist, LIQ_HISTOGRAM_MAGIC) ||
       bad_object!(input_image, LIQ_IMAGE_MAGIC) { return LIQ_INVALID_POINTER; }

    input_hist.add_image(attr, input_image).err().unwrap_or(LIQ_OK)
}

#[no_mangle]
#[inline(never)]
pub unsafe fn liq_executing_user_callback(callback: liq_image_get_rgba_row_callback, temp_row: &mut [MaybeUninit<liq_color>], row: usize, user_info: *mut std::os::raw::c_void) {
    callback(temp_row.as_mut_ptr(), row as _, temp_row.len() as _, user_info);
}

#[test]
fn links_and_runs() {
    use std::ptr;
    unsafe {
        assert!(liq_version() >= 40000);
        let attr = liq_attr_create().unwrap();
        let mut hist = liq_histogram_create(&*attr).unwrap();
        assert_eq!(LIQ_OK, liq_histogram_add_fixed_color(&mut *hist, liq_color {r: 0, g: 0, b: 0, a: 0}, 0.));
        liq_histogram_add_colors(&mut *hist, &*attr, ptr::null(), 0, 0.);

        let mut res = MaybeUninit::uninit();

        // this is fine, because there is 1 fixed color to generate
        assert_eq!(LIQ_OK, liq_histogram_quantize(&mut *hist, &*attr, &mut res));
        let res = res.assume_init().unwrap();

        liq_result_destroy(Some(res));
        liq_histogram_destroy(Some(hist));
        liq_attr_destroy(Some(attr));
    }
}

#[test]
#[allow(deprecated)]
fn link_every_symbol() {
    use std::os::raw::c_void;

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
    use std::mem::MaybeUninit;
    use rgb::RGBA8 as RGBA;

    let mut called = 0;
    let mut res = unsafe {
        let mut a = liq_attr_create().unwrap();
        unsafe extern "C" fn get_row(output_row: *mut MaybeUninit<RGBA>, y: c_int, width: c_int, user_data: *mut c_void) {
            assert!((0..5).contains(&y));
            assert_eq!(123, width);
            for i in 0..width as isize {
                let n = i as u8;
                (*output_row.offset(i as isize)).write(RGBA::new(n, n, n, n));
            }
            let user_data = user_data as *mut i32;
            *user_data += 1;
        }
        let mut img = liq_image_create_custom(&a, get_row, AnySyncSendPtr((&mut called) as *mut _ as *mut c_void), 123, 5, 0.).unwrap();
        liq_quantize_image(&mut a, &mut img).unwrap()
    };
    assert!(called > 5 && called < 50);
    let pal = liq_get_palette(&mut res).unwrap();
    assert_eq!(123, pal.count);
}
