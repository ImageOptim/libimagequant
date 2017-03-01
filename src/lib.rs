/// https://pngquant.org/lib/

pub mod ffi;

pub use ffi::liq_error;
pub use ffi::liq_error::*;
use std::os::raw::c_int;
use std::fmt;
use std::mem;
use std::ptr;

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

pub struct Histogram<'a> {
    attr: &'a Attributes,
    handle: *mut ffi::liq_histogram,
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

impl<'a> Drop for Histogram<'a> {
    fn drop(&mut self) {
        unsafe {
            ffi::liq_histogram_destroy(&mut *self.handle);
        }
    }
}

impl Clone for Attributes {
    fn clone(&self) -> Attributes {
        unsafe { Attributes { handle: ffi::liq_attr_copy(&*self.handle) } }
    }
}

impl Attributes {
    pub fn new() -> Attributes {
        let handle = unsafe { ffi::liq_attr_create() };
        assert!(!handle.is_null());
        Attributes { handle: handle }
    }

    pub fn set_max_colors(&mut self, value: i32) -> liq_error {
        unsafe { ffi::liq_set_max_colors(&mut *self.handle, value) }
    }

    pub fn set_min_posterization(&mut self, value: i32) -> liq_error {
        unsafe { ffi::liq_set_min_posterization(&mut *self.handle, value) }
    }

    pub fn set_quality(&mut self, min: u32, max: u32) -> liq_error {
        unsafe { ffi::liq_set_quality(&mut *self.handle, min as c_int, max as c_int) }
    }

    pub fn set_speed(&mut self, value: i32) -> liq_error {
        unsafe { ffi::liq_set_speed(&mut *self.handle, value) }
    }

    pub fn set_last_index_transparent(&mut self, value: bool) -> () {
        unsafe { ffi::liq_set_last_index_transparent(&mut *self.handle, value as c_int) }
    }

    pub fn speed(&mut self) -> i32 {
        unsafe { ffi::liq_get_speed(&*self.handle) }
    }

    pub fn max_colors(&mut self) -> i32 {
        unsafe { ffi::liq_get_max_colors(&*self.handle) }
    }

    pub fn new_image<'a, T: Copy + Clone>(&self, bitmap: &'a [T], width: usize, height: usize, gamma: f64) -> Option<Image<'a>> {
        Image::new(self, bitmap, width, height, gamma)
    }

    pub fn new_histogram(&self) -> Histogram {
        Histogram::new(&self)
    }

    pub fn quantize(&mut self, image: &Image) -> Result<QuantizationResult, liq_error> {
        unsafe {
            let mut h = ptr::null_mut();
            match ffi::liq_image_quantize(&mut *image.handle, &mut *self.handle, &mut h) {
                liq_error::LIQ_OK if !h.is_null() => Ok(QuantizationResult { handle: h }),
                err => Err(err),
            }
        }
    }
}

pub fn new() -> Attributes {
    Attributes::new()
}

impl<'a> Histogram<'a> {
    pub fn new(attr: &'a Attributes) -> Histogram<'a> {
        Histogram {
            attr: attr,
            handle: unsafe { ffi::liq_histogram_create(&*attr.handle) },
        }
    }

    pub fn add_image(&mut self, image: &Image) -> liq_error {
        unsafe {
            ffi::liq_histogram_add_image(&mut *self.handle, &*self.attr.handle, &*image.handle)
        }
    }

    pub fn quantize(&mut self) -> Result<QuantizationResult, liq_error> {
        unsafe {
            let mut h = ptr::null_mut();
            match ffi::liq_histogram_quantize(&mut *self.handle, &*self.attr.handle, &mut h) {
                liq_error::LIQ_OK if !h.is_null() => Ok(QuantizationResult { handle: h }),
                err => Err(err),
            }
        }
    }
}

impl<'a> Image<'a> {
    pub fn new<T: Copy + Clone>(attr: &Attributes, bitmap: &'a [T], width: usize, height: usize, gamma: f64) -> Option<Image<'a>> {
        match mem::size_of::<T>() {
            1 | 4 => {}
            _ => return None,
        }
        if bitmap.len() * mem::size_of::<T>() < width*height*4 {
            println!("Buffer length is {}x{} bytes, which is not enough for {}x{}x4 RGBA bytes", bitmap.len(), mem::size_of::<T>(), width, height);
            return None;
        }
        unsafe {
            match ffi::liq_image_create_rgba(&*attr.handle, mem::transmute(bitmap.as_ptr()), width as c_int, height as c_int, gamma) {
                h if !h.is_null() => Some(Image {
                    handle: h,
                    _marker: std::marker::PhantomData::<&'a [u8]>,
                }),
                _ => None,
            }
        }
    }

    pub fn width(&mut self) -> usize {
        unsafe { ffi::liq_image_get_width(&*self.handle) as usize }
    }

    pub fn height(&mut self) -> usize {
        unsafe { ffi::liq_image_get_height(&*self.handle) as usize }
    }
}

impl QuantizationResult {
    pub fn set_dithering_level(&mut self, value: f32) -> liq_error {
        unsafe { ffi::liq_set_dithering_level(&mut *self.handle, value) }
    }

    pub fn set_output_gamma(&mut self, value: f64) -> liq_error {
        unsafe { ffi::liq_set_output_gamma(&mut *self.handle, value) }
    }

    pub fn output_gamma(&mut self) -> f64 {
        unsafe { ffi::liq_get_output_gamma(&*self.handle) }
    }

    pub fn quantization_quality(&mut self) -> isize {
        unsafe { ffi::liq_get_quantization_quality(&*self.handle) as isize }
    }

    pub fn palette(&mut self) -> Vec<Color> {
        unsafe {
            let ref pal = *ffi::liq_get_palette(&mut *self.handle);
            pal.entries.iter().cloned().take(pal.count as usize).collect()
        }
    }

    pub fn remapped(&mut self, image: &mut Image) -> Option<(Vec<Color>, Vec<u8>)> {
        let len = image.width() * image.height();
        let mut buf = Vec::with_capacity(len);
        unsafe {
            buf.set_len(len); // Creates uninitialized buffer
            match ffi::liq_write_remapped_image(&mut *self.handle, &mut *image.handle, buf.as_mut_ptr(), buf.len()) {
                LIQ_OK => Some((self.palette(), buf)),
                _ => None,
            }
        }
    }
}

#[test]
fn takes_rgba() {
    let liq = Attributes::new();

    #[allow(dead_code)]
    #[derive(Copy, Clone)]
    struct RGBA {r:u8, g:u8, b:u8, a:u8};
    let img = vec![RGBA {r:0, g:0, b:0, a:0}; 8];


    liq.new_image(&img, 1, 1, 0.0).unwrap();
    liq.new_image(&img, 4, 2, 0.0).unwrap();
    liq.new_image(&img, 8, 1, 0.0).unwrap();
    assert!(liq.new_image(&img, 9, 1, 0.0).is_none());
    assert!(liq.new_image(&img, 4, 3, 0.0).is_none());

    #[allow(dead_code)]
    #[derive(Copy, Clone)]
    struct RGB {r:u8, g:u8, b:u8};
    let badimg = vec![RGB {r:0, g:0, b:0}; 8];
    assert!(liq.new_image(&badimg, 1, 1, 0.0).is_none());
    assert!(liq.new_image(&badimg, 100, 100, 0.0).is_none());
}

#[test]
fn histogram() {
    let attr = Attributes::new();
    let mut hist = attr.new_histogram();

    let bitmap1 = vec![0u8; 4];
    let image1 = attr.new_image(&bitmap1[..], 1, 1, 0.0).unwrap();
    hist.add_image(&image1);

    let bitmap2 = vec![255u8; 4];
    let image2 = attr.new_image(&bitmap2[..], 1, 1, 0.0).unwrap();
    hist.add_image(&image2);

    let mut res = hist.quantize().unwrap();
    let pal = res.palette();
    assert_eq!(2, pal.len());
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

    assert_eq!(width * height, pixels.len());
    assert_eq!(100, res.quantization_quality());
    assert_eq!(Color{r:255,g:255,b:255,a:255}, palette[0]);
    assert_eq!(Color{r:0x55,g:0x66,b:0x77,a:255}, palette[1]);
}
