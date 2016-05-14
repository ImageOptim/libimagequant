#![crate_type = "lib"]
#![allow(improper_ctypes)]

/// http://pngquant.org/lib/

extern crate libc;

pub use ffi::liq_error;
pub use ffi::liq_error::*;

use libc::{c_int, size_t};
use std::option::Option;
use std::vec::Vec;
use std::fmt;

#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.a {
            255 => write!(f, "#{:02x}{:02x}{:02x}", self.r, self.g, self.b),
            _ => write!(f, "rgba({},{},{},{})", self.r, self.g, self.b, self.a),
        }
    }
}

#[allow(dead_code)]
#[allow(non_camel_case_types)]
pub mod ffi {
    use libc::{c_int, size_t};
    use std::fmt;

    #[repr(C)]
    #[allow(missing_copy_implementations)]
    pub struct liq_attr;

    #[repr(C)]
    #[allow(missing_copy_implementations)]
    pub struct liq_image;

    #[repr(C)]
    #[allow(missing_copy_implementations)]
    pub struct liq_result;

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub enum liq_error {
        LIQ_OK = 0,
        LIQ_QUALITY_TOO_LOW = 99,
        LIQ_VALUE_OUT_OF_RANGE = 100,
        LIQ_OUT_OF_MEMORY,
        LIQ_NOT_READY,
        LIQ_BITMAP_NOT_AVAILABLE,
        LIQ_BUFFER_TOO_SMALL,
        LIQ_INVALID_POINTER,
    }

    #[allow(non_camel_case_types)]
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub enum liq_ownership {
        LIQ_OWN_ROWS=4,
        LIQ_OWN_PIXELS=8,
        LIQ_OWN_QUALITY_MAP=16,
    }


    #[allow(non_camel_case_types)]
    #[repr(C)]
    pub struct liq_palette {
        pub count: c_int,
        pub entries: [super::Color; 256],
    }

    impl fmt::Debug for liq_error {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{}", match *self {
                liq_error::LIQ_OK => "OK",
                liq_error::LIQ_QUALITY_TOO_LOW => "LIQ_QUALITY_TOO_LOW",
                liq_error::LIQ_VALUE_OUT_OF_RANGE => "VALUE_OUT_OF_RANGE",
                liq_error::LIQ_OUT_OF_MEMORY => "OUT_OF_MEMORY",
                liq_error::LIQ_NOT_READY => "NOT_READY",
                liq_error::LIQ_BITMAP_NOT_AVAILABLE => "BITMAP_NOT_AVAILABLE",
                liq_error::LIQ_BUFFER_TOO_SMALL => "BUFFER_TOO_SMALL",
                liq_error::LIQ_INVALID_POINTER => "INVALID_POINTER",
            })
        }
    }

    #[link(name="imagequant", kind="static")]
    extern {

        pub fn liq_attr_create() -> *mut liq_attr;
        pub fn liq_attr_copy(orig: &liq_attr) -> *mut liq_attr;
        pub fn liq_attr_destroy(attr: &mut liq_attr);

        pub fn liq_set_max_colors(attr: &mut liq_attr, colors: c_int) -> liq_error;
        pub fn liq_get_max_colors(attr: &liq_attr) -> c_int;
        pub fn liq_set_speed(attr: &mut liq_attr, speed: c_int) -> liq_error;
        pub fn liq_get_speed(attr: &liq_attr) -> c_int;
        pub fn liq_set_min_posterization(attr: &mut liq_attr, bits: c_int) -> liq_error;
        pub fn liq_get_min_posterization(attr: &liq_attr) -> c_int;
        pub fn liq_set_quality(attr: &mut liq_attr, minimum: c_int, maximum: c_int) -> liq_error;
        pub fn liq_get_min_quality(attr: &liq_attr) -> c_int;
        pub fn liq_get_max_quality(attr: &liq_attr) -> c_int;
        pub fn liq_set_last_index_transparent(attr: &mut liq_attr, is_last: c_int);

        pub fn liq_image_create_rgba_rows(attr: &liq_attr, rows: *const *const u8, width: c_int, height: c_int, gamma: f64) -> *mut liq_image;
        pub fn liq_image_create_rgba(attr: &liq_attr, bitmap: *const u8, width: c_int, height: c_int, gamma: f64) -> *mut liq_image;

        pub fn liq_image_get_width(img: &liq_image) -> c_int;
        pub fn liq_image_get_height(img: &liq_image) -> c_int;
        pub fn liq_image_destroy(img: &liq_image);

        pub fn liq_quantize_image(options: &liq_attr, input_image: &liq_image) -> *mut liq_result;

        pub fn liq_set_dithering_level(res: &liq_result, dither_level: f32) -> liq_error;
        pub fn liq_set_output_gamma(res: &liq_result, gamma: f64) -> liq_error;
        pub fn liq_get_output_gamma(result: &liq_result) -> f64;

        pub fn liq_get_palette(result: &liq_result) -> *const liq_palette;

        pub fn liq_write_remapped_image(result: &liq_result, input_image: &liq_image, buffer: *mut u8, buffer_size: size_t) -> liq_error;
        pub fn liq_write_remapped_image_rows(result: &liq_result, input_image: &liq_image, row_pointers: *const *mut u8) -> liq_error;

        pub fn liq_get_quantization_error(result: &liq_result) -> f64;
        pub fn liq_get_quantization_quality(result: &liq_result) -> c_int;

        pub fn liq_result_destroy(res: &liq_result);
    }
}

pub struct Attributes {
    handle: *mut ffi::liq_attr,
}


pub struct Image<'a> {
    handle: *mut ffi::liq_image,
    _marker: std::marker::PhantomData<&'a [u8]>,
}

pub struct QuantizationResult {
    handle: *mut ffi::liq_result,
}

impl Drop for Attributes {
    fn drop(&mut self) {
        unsafe {
            ffi::liq_attr_destroy(&mut *self.handle);
        }
    }
}

impl<'a> Drop for Image<'a> {
    fn drop(&mut self) {
        unsafe {
            ffi::liq_image_destroy(&mut *self.handle);
        }
    }
}

impl Drop for QuantizationResult {
    fn drop(&mut self) {
        unsafe {
            ffi::liq_result_destroy(&mut *self.handle);
        }
    }
}

impl Clone for Attributes {
    fn clone(&self) -> Attributes {
        unsafe {
            Attributes { handle: ffi::liq_attr_copy(&*self.handle) }
        }
    }
}

impl Attributes {
    pub fn new() -> Attributes {
        unsafe {
            Attributes { handle: ffi::liq_attr_create() }
        }
    }

    pub fn set_max_colors(&mut self, value: i32) -> liq_error {
        unsafe {
            ffi::liq_set_max_colors(&mut *self.handle, value)
        }
    }

    pub fn set_min_posterization(&mut self, value: i32) -> liq_error {
        unsafe {
            ffi::liq_set_min_posterization(&mut *self.handle, value)
        }
    }

    pub fn set_quality(&mut self, min: u32, max: u32) -> liq_error {
        unsafe {
            ffi::liq_set_quality(&mut *self.handle, min as c_int, max as c_int)
        }
    }

    pub fn set_speed(&mut self, value: i32) -> liq_error {
        unsafe {
            ffi::liq_set_speed(&mut *self.handle, value)
        }
    }

    pub fn set_last_index_transparent(&mut self, value: bool) -> () {
        unsafe {
            ffi::liq_set_last_index_transparent(&mut *self.handle, value as c_int)
        }
    }

    pub fn speed(&mut self) -> i32 {
        unsafe {
            ffi::liq_get_speed(&*self.handle)
        }
    }

    pub fn max_colors(&mut self) -> i32 {
        unsafe {
            ffi::liq_get_max_colors(&*self.handle)
        }
    }

    pub fn new_image<'a>(&self, bitmap: &'a [u8], width: usize, height: usize, gamma: f64) -> Option<Image<'a>> {
        Image::new(self, bitmap, width, height, gamma)
    }

    pub fn quantize(&mut self, image: &Image) -> Result<QuantizationResult,liq_error> {
        unsafe {
            match ffi::liq_quantize_image(&mut *self.handle, &mut *image.handle) {
                h if !h.is_null() => Ok(QuantizationResult { handle: h }),
                _ => Err(LIQ_QUALITY_TOO_LOW),
            }
        }
    }
}

pub fn new() -> Attributes {
    Attributes::new()
}

impl<'a> Image<'a> {
    pub fn new(attr: &Attributes, bitmap: &'a [u8], width: usize, height: usize, gamma: f64) -> Option<Image<'a>> {
        if bitmap.len() < width*height*4 {
            return None;
        }
        unsafe {
            match ffi::liq_image_create_rgba(&*attr.handle, bitmap.as_ptr(), width as c_int, height as c_int, gamma) {
                h if !h.is_null() => Some(Image {
                    handle: h,
                    _marker: std::marker::PhantomData::<&'a [u8]>,
                }),
                _ => None,
            }
        }
    }

    pub fn width(&mut self) -> usize {
        unsafe {
            ffi::liq_image_get_width(&*self.handle) as usize
        }
    }

    pub fn height(&mut self) -> usize {
        unsafe {
            ffi::liq_image_get_height(&*self.handle) as usize
        }
    }
}

impl QuantizationResult {

    pub fn set_dithering_level(&mut self, value: f32) -> liq_error {
        unsafe {
            ffi::liq_set_dithering_level(&mut *self.handle, value)
        }
    }

    pub fn set_output_gamma(&mut self, value: f64) -> liq_error {
        unsafe {
            ffi::liq_set_output_gamma(&mut *self.handle, value)
        }
    }

    pub fn output_gamma(&mut self) -> f64 {
        unsafe {
            ffi::liq_get_output_gamma(&*self.handle)
        }
    }

    pub fn quantization_quality(&mut self) -> isize {
        unsafe {
            ffi::liq_get_quantization_quality(&*self.handle) as isize
        }
    }

    pub fn palette(&mut self) -> Vec<Color> {
        unsafe {
            let pal = ffi::liq_get_palette(&mut *self.handle);
            (*pal).entries.to_vec()
        }
    }

    pub fn remapped(&mut self, image: &mut Image) -> Option<(Vec<Color>, Vec<u8>)> {
        unsafe {
            let len = image.width() * image.height();
            let mut buf = Vec::with_capacity(len);
            buf.set_len(len); // Creates uninitialized buffer
            match ffi::liq_write_remapped_image(&mut *self.handle, &mut *image.handle, buf.as_mut_ptr(), buf.len() as size_t) {
                LIQ_OK => Some((self.palette(), buf)),
                _ => None,
            }
        }
    }
}


#[test]
fn poke_it() {
    let width = 10usize;
    let height = 10usize;
    let mut fakebitmap = vec![255u8; 4*width*height];

    fakebitmap[0] = 0x55;
    fakebitmap[1] = 0x66;
    fakebitmap[2] = 0x77;

    // Configure the library
    let mut liq = Attributes::new();
    liq.set_speed(5);
    liq.set_quality(70, 99);

    // Describe the bitmap
    let ref mut img = liq.new_image(&fakebitmap[..], width, height, 0.0).unwrap();

    // The magic happens in quantize()
    let mut res = match liq.quantize(img) {
        Ok(res) => res,
        Err(err) => panic!("Quantization failed, because: {:?}", err),
    };

    // Enable dithering for subsequent remappings
    res.set_dithering_level(1.0);

    // You can reuse the result to generate several images with the same palette
    let (palette, pixels) = res.remapped(img).unwrap();

    assert_eq!(width*height, pixels.len());
    assert_eq!(100, res.quantization_quality());
    assert_eq!(Color{r:255,g:255,b:255,a:255}, palette[0]);
    assert_eq!(Color{r:0x55,g:0x66,b:0x77,a:255}, palette[1]);
}
