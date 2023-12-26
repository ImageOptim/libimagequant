//! These are internal unstable private helper methods for imagequant-sys.
//! For public stable a C FFI interface, see imagequant-sys crate instead.
#![allow(missing_docs)]
#![allow(clippy::missing_safety_doc)]

use crate::rows::RowCallback;
use crate::seacow::Pointer;
use crate::seacow::RowBitmapMut;
use crate::seacow::SeaCow;
use crate::Attributes;
use crate::Error;
use crate::Image;
use crate::pal::Palette;
use crate::QuantizationResult;
use crate::RGBA;
use std::mem::MaybeUninit;

pub const LIQ_VERSION: u32 = 40202;

#[must_use]
pub fn liq_get_palette_impl(r: &mut QuantizationResult) -> &Palette {
    r.int_palette()
}

#[must_use]
pub unsafe fn liq_image_create_rgba_rows_impl<'rows>(attr: &Attributes, rows: &'rows [*const RGBA], width: u32, height: u32, gamma: f64) -> Option<crate::image::Image<'rows>> {
    let rows = SeaCow::borrowed(&*(rows as *const [*const rgb::RGBA<u8>] as *const [Pointer<rgb::RGBA<u8>>]));
    let rows_slice = rows.as_slice();
    if rows_slice.iter().any(|r| r.0.is_null()) {
        return None;
    }
    crate::image::Image::new_internal(attr, crate::rows::PixelsSource::Pixels { rows, pixels: None }, width, height, gamma).ok()
}

#[must_use]
pub unsafe fn liq_image_create_rgba_bitmap_impl<'rows>(attr: &Attributes, rows: Box<[*const RGBA]>, width: u32, height: u32, gamma: f64) -> Option<crate::image::Image<'rows>> {
    let rows = SeaCow::boxed(std::mem::transmute::<Box<[*const RGBA]>, Box<[Pointer<RGBA>]>>(rows));
    let rows_slice = rows.as_slice();
    if rows_slice.iter().any(|r| r.0.is_null()) {
        return None;
    }
    crate::image::Image::new_internal(attr, crate::rows::PixelsSource::Pixels { rows, pixels: None }, width, height, gamma).ok()
}

#[must_use]
pub unsafe fn liq_image_create_custom_impl<'rows>(attr: &Attributes, row_callback: Box<RowCallback<'rows>>, width: u32, height: u32, gamma: f64) -> Option<Image<'rows>> {
    Image::new_internal(attr, crate::rows::PixelsSource::Callback(row_callback), width, height, gamma).ok()
}

pub unsafe fn liq_write_remapped_image_impl(result: &mut QuantizationResult, input_image: &mut Image, buffer_bytes: &mut [MaybeUninit<u8>]) -> Result<(), Error> {
    let rows = RowBitmapMut::new_contiguous(buffer_bytes, input_image.width());
    result.write_remapped_image_rows_internal(input_image, rows)
}

pub unsafe fn liq_write_remapped_image_rows_impl(result: &mut QuantizationResult, input_image: &mut Image, rows: &mut [*mut MaybeUninit<u8>]) -> Result<(), Error> {
    let rows = RowBitmapMut::new(rows, input_image.width());
    result.write_remapped_image_rows_internal(input_image, rows)
}

/// Not recommended
pub unsafe fn liq_image_set_memory_ownership_impl(image: &mut Image<'_>, own_rows: bool, own_pixels: bool, free_fn: unsafe extern fn(*mut std::os::raw::c_void)) -> Result<(), Error> {
    image.px.set_memory_ownership(own_rows, own_pixels, free_fn)
}
