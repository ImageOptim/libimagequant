// http://pngquant.org/lib/

#![crate_id = "imagequant#2.2"]
#![crate_type = "lib"]

extern crate libc;

pub use ffi::liq_error;
use libc::{c_int, size_t};
use std::option::Option;
use std::vec::Vec;
use std::fmt;

pub struct Color {
    r: u8, g: u8, b: u8, a: u8,
}

impl fmt::Show for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.a {
            255 => write!(f.buf, "\\#{:02x}{:02x}{:02x}", self.r, self.g, self.b),
            _ => write!(f.buf, "rgba({},{},{},{})", self.r, self.g, self.b, self.a),
        }
    }
}


#[allow(dead_code)]
#[allow(non_camel_case_types)]
pub mod ffi {
    use libc::{c_int, size_t};
    use std::fmt;


    pub struct liq_attr();
    pub struct liq_image();
    pub struct liq_result();

    #[repr(C)]
    pub enum liq_error {
        LIQ_OK = 0,
        LIQ_VALUE_OUT_OF_RANGE = 100,
        LIQ_OUT_OF_MEMORY,
        LIQ_NOT_READY,
        LIQ_BITMAP_NOT_AVAILABLE,
        LIQ_BUFFER_TOO_SMALL,
        LIQ_INVALID_POINTER,
    }

    #[allow(non_camel_case_types)]
    #[repr(C)]
    pub enum liq_ownership {
        LIQ_OWN_ROWS=4,
        LIQ_OWN_PIXELS=8,
        LIQ_OWN_QUALITY_MAP=16,
    }


    #[allow(non_camel_case_types)]
    pub struct liq_palette {
        pub count: c_int,
        pub entries: [super::Color, ..256],
    }

    impl fmt::Show for liq_error {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.buf.write_str(match *self {
                LIQ_OK => "OK",
                LIQ_VALUE_OUT_OF_RANGE => "VALUE_OUT_OF_RANGE",
                LIQ_OUT_OF_MEMORY => "OUT_OF_MEMORY",
                LIQ_NOT_READY => "NOT_READY",
                LIQ_BITMAP_NOT_AVAILABLE => "BITMAP_NOT_AVAILABLE",
                LIQ_BUFFER_TOO_SMALL => "BUFFER_TOO_SMALL",
                LIQ_INVALID_POINTER => "INVALID_POINTER",
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

        pub fn liq_image_create_rgba_rows(attr: &liq_attr, rows: **u8, width: c_int, height: c_int, gamma: f64) -> *mut liq_image;
        pub fn liq_image_create_rgba(attr: &liq_attr, bitmap: *u8, width: c_int, height: c_int, gamma: f64) -> *mut liq_image;

        pub fn liq_image_get_width(img: &liq_image) -> c_int;
        pub fn liq_image_get_height(img: &liq_image) -> c_int;
        pub fn liq_image_destroy(img: &liq_image);

        pub fn liq_quantize_image(options: &liq_attr, input_image: &liq_image) -> *mut liq_result;

        pub fn liq_set_dithering_level(res: &liq_result, dither_level: f32) -> liq_error;
        pub fn liq_set_output_gamma(res: &liq_result, gamma: f64) -> liq_error;
        pub fn liq_get_output_gamma(result: &liq_result) -> f64;

        pub fn liq_get_palette(result: &liq_result) -> *liq_palette;

        pub fn liq_write_remapped_image(result: &liq_result, input_image: &liq_image, buffer: *mut u8, buffer_size: size_t) -> liq_error;
        pub fn liq_write_remapped_image_rows(result: &liq_result, input_image: &liq_image, row_pointers: **mut u8) -> liq_error;

        pub fn liq_get_quantization_error(result: &liq_result) -> f64;
        pub fn liq_get_quantization_quality(result: &liq_result) -> c_int;

        pub fn liq_result_destroy(res: &liq_result);
    }
}

pub struct Attributes {
    handle: *mut ffi::liq_attr,
}

pub struct Image {
    handle: *mut ffi::liq_image,
    memory_reference: ~[u8],
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

impl Drop for Image {
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
    pub fn new() -> ~Attributes {
        unsafe {
            ~Attributes { handle: ffi::liq_attr_create() }
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

    pub fn new_image(&self, bitmap: ~[u8], width: uint, height: uint, gamma: f64) -> Option<~Image> {
        Image::new(self, bitmap, width, height, gamma)
    }

    pub fn quantize(&mut self, image: &Image) -> Option<QuantizationResult> {
        unsafe {
            match ffi::liq_quantize_image(&mut *self.handle, &mut *image.handle) {
                h if h.is_not_null() => Some(QuantizationResult { handle: h }),
                _ => None,
            }
        }
    }
}

pub fn new() -> ~Attributes {
    Attributes::new()
}

impl Image {
    pub fn new(attr: &Attributes, bitmap: ~[u8], width: uint, height: uint, gamma: f64) -> Option<~Image> {
        if bitmap.len() < width*height*4 {
            return None;
        }
        unsafe {
            match ffi::liq_image_create_rgba(&*attr.handle, bitmap.as_ptr(), width as c_int, height as c_int, gamma) {
                h if h.is_not_null() => Some(~Image {
                    handle: h,
                    memory_reference: bitmap,
                }),
                _ => None,
            }
        }
    }

    pub fn width(&mut self) -> uint {
        unsafe {
            ffi::liq_image_get_width(&*self.handle) as uint
        }
    }

    pub fn height(&mut self) -> uint {
        unsafe {
            ffi::liq_image_get_height(&*self.handle) as uint
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

    pub fn quantization_quality(&mut self) -> int {
        unsafe {
            ffi::liq_get_quantization_quality(&*self.handle) as int
        }
    }

    pub fn palette(&mut self) -> Vec<Color> {
        unsafe {
            let pal = ffi::liq_get_palette(&mut *self.handle);
            Vec::from_fn((*pal).count as uint, |i| (*pal).entries[i])
        }
    }

    pub fn remapped(&mut self, image: &mut Image) -> Option<(Vec<Color>, Vec<u8>)> {
        unsafe {
            let mut buf = Vec::from_elem(image.width() * image.height(), 0 as u8);
            match ffi::liq_write_remapped_image(&mut *self.handle, &mut *image.handle, buf.as_mut_ptr(), buf.len() as size_t) {
                ffi::LIQ_OK => Some((self.palette(), buf)),
                _ => None,
            }
        }
    }
}


