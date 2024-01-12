//! <https://pngquant.org/lib/>
//!
//! Converts RGBA images to 8-bit with alpha channel.
//!
//! See `examples/` directory for example code.
#![doc(html_logo_url = "https://pngquant.org/pngquant-logo.png")]
#![deny(missing_docs)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::if_not_else)]
#![allow(clippy::inline_always)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::wildcard_imports)]
#![deny(clippy::semicolon_if_nothing_returned)]

mod attr;
mod blur;
mod error;
mod hist;
mod image;
mod kmeans;
mod mediancut;
mod nearest;
mod pal;
mod quant;
mod remap;
mod rows;
mod seacow;

#[cfg(not(feature = "threads"))]
mod rayoff;

#[cfg(feature = "threads")]
mod rayoff {
    pub(crate) fn num_cpus() -> usize { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) }
    pub(crate) use rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSliceMut};
    pub(crate) use rayon::in_place_scope as scope;
    pub(crate) use thread_local::ThreadLocal;
}

/// Use imagequant-sys crate instead
#[cfg(feature = "_internal_c_ffi")]
pub mod capi;

pub use attr::Attributes;
pub use attr::ControlFlow;
pub use error::Error;
pub use hist::Histogram;
pub use hist::HistogramEntry;
pub use image::Image;
#[doc(hidden)]
pub use pal::Palette;
pub use pal::RGBA;
pub use quant::QuantizationResult;

#[doc(hidden)]
#[deprecated(note = "Please use the imagequant::Error type. This will be removed")]
pub use error::Error as liq_error;

const LIQ_HIGH_MEMORY_LIMIT: usize = 1 << 26;

/// [Start here][Attributes]: creates new handle for library configuration
///
/// See [`Attributes`]
#[inline(always)]
#[must_use]
pub fn new() -> Attributes {
    Attributes::new()
}

#[test]
fn copy_img() {
    let tmp = vec![RGBA::new(1, 2, 3, 4); 10 * 100];
    let liq = Attributes::new();
    let _ = liq.new_image_stride(tmp, 10, 100, 10, 0.).unwrap();
}

#[test]
fn takes_rgba() {
    let liq = Attributes::new();

    let img = vec![RGBA {r:0, g:0, b:0, a:0}; 8];


    liq.new_image_borrowed(&img, 1, 1, 0.0).unwrap();
    liq.new_image_borrowed(&img, 4, 2, 0.0).unwrap();
    liq.new_image_borrowed(&img, 8, 1, 0.0).unwrap();
    assert!(liq.new_image_borrowed(&img, 9, 1, 0.0).is_err());
    assert!(liq.new_image_borrowed(&img, 4, 3, 0.0).is_err());
}

#[test]
fn histogram() {
    let attr = Attributes::new();
    let mut hist = Histogram::new(&attr);

    let bitmap1 = [RGBA {r:0, g:0, b:0, a:0}; 1];
    let mut image1 = attr.new_image(&bitmap1[..], 1, 1, 0.0).unwrap();
    hist.add_image(&attr, &mut image1).unwrap();

    let bitmap2 = [RGBA {r:255, g:255, b:255, a:255}; 1];
    let mut image2 = attr.new_image(&bitmap2[..], 1, 1, 0.0).unwrap();
    hist.add_image(&attr, &mut image2).unwrap();

    hist.add_colors(&[HistogramEntry {
        color: RGBA::new(255, 128, 255, 128),
        count: 10,
    }], 0.0).unwrap();

    let mut res = hist.quantize(&attr).unwrap();
    let pal = res.palette();
    assert_eq!(3, pal.len());
}

#[test]
fn poke_it() {
    let width = 10usize;
    let height = 10usize;
    let mut fakebitmap = vec![RGBA::new(255, 255, 255, 255); width * height];

    fakebitmap[0].r = 0x55;
    fakebitmap[0].g = 0x66;
    fakebitmap[0].b = 0x77;

    // Configure the library
    let mut liq = Attributes::new();
    liq.set_speed(5).unwrap();
    liq.set_quality(70, 99).unwrap();
    liq.set_min_posterization(1).unwrap();
    assert_eq!(1, liq.min_posterization());
    liq.set_min_posterization(0).unwrap();

    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering::SeqCst;
    use std::sync::Arc;

    let log_called = Arc::new(AtomicBool::new(false));
    let log_called2 = log_called.clone();
    liq.set_log_callback(move |_attr, _msg| {
        log_called2.store(true, SeqCst);
    });

    let prog_called = Arc::new(AtomicBool::new(false));
    let prog_called2 = prog_called.clone();
    liq.set_progress_callback(move |_perc| {
        prog_called2.store(true, SeqCst);
        ControlFlow::Continue
    });

    // Describe the bitmap
    let img = &mut liq.new_image(&fakebitmap[..], width, height, 0.0).unwrap();

    // The magic happens in quantize()
    let mut res = match liq.quantize(img) {
        Ok(res) => res,
        Err(err) => panic!("Quantization failed, because: {err:?}"),
    };

    // Enable dithering for subsequent remappings
    res.set_dithering_level(1.0).unwrap();

    // You can reuse the result to generate several images with the same palette
    let (palette, pixels) = res.remapped(img).unwrap();

    assert_eq!(width * height, pixels.len());
    assert_eq!(100, res.quantization_quality().unwrap());
    assert_eq!(RGBA { r: 255, g: 255, b: 255, a: 255 }, palette[0]);
    assert_eq!(RGBA { r: 0x55, g: 0x66, b: 0x77, a: 255 }, palette[1]);

    assert!(log_called.load(SeqCst));
    assert!(prog_called.load(SeqCst));
}

#[test]
fn set_importance_map() {
    let liq = new();
    let bitmap = &[RGBA::new(255, 0, 0, 255), RGBA::new(0u8, 0, 255, 255)];
    let mut img = liq.new_image(&bitmap[..], 2, 1, 0.).unwrap();
    let map = &[255, 0];
    img.set_importance_map(&map[..]).unwrap();
    let mut res = liq.quantize(&mut img).unwrap();
    let pal = res.palette();
    assert_eq!(1, pal.len(), "{pal:?}");
    assert_eq!(bitmap[0], pal[0]);
}

#[test]
fn thread() {
    let liq = Attributes::new();
    std::thread::spawn(move || {
        let b = vec![RGBA::new(0, 0, 0, 0); 1];
        liq.new_image_borrowed(&b, 1, 1, 0.).unwrap();
    }).join().unwrap();
}

#[test]
fn r_callback_test() {
    use std::mem::MaybeUninit;
    use std::sync::atomic::AtomicU16;
    use std::sync::atomic::Ordering::SeqCst;
    use std::sync::Arc;

    let called = Arc::new(AtomicU16::new(0));
    let called2 = called.clone();
    let mut res = {
        let a = new();
        let get_row = move |output_row: &mut [MaybeUninit<RGBA>], y: usize| {
            assert!((0..5).contains(&y));
            assert_eq!(123, output_row.len());
            for (n, out) in output_row.iter_mut().enumerate() {
                let n = n as u8;
                out.write(RGBA::new(n, n, n, n));
            }
            called2.fetch_add(1, SeqCst);
        };
        let mut img = unsafe {
            Image::new_fn(&a, get_row, 123, 5, 0.).unwrap()
        };
        a.quantize(&mut img).unwrap()
    };
    let called = called.load(SeqCst);
    assert!(called > 5 && called < 50);
    assert_eq!(123, res.palette().len());
}

#[test]
fn sizes() {
    use pal::PalF;
    use pal::Palette;
    assert!(std::mem::size_of::<PalF>() < crate::pal::MAX_COLORS*(8*4)+32, "{}", std::mem::size_of::<PalF>());
    assert!(std::mem::size_of::<QuantizationResult>() < std::mem::size_of::<PalF>() + std::mem::size_of::<Palette>() + 100, "{}", std::mem::size_of::<QuantizationResult>());
    assert!(std::mem::size_of::<Attributes>() < 200);
    assert!(std::mem::size_of::<Image>() < 300);
    assert!(std::mem::size_of::<Histogram>() < 200);
    assert!(std::mem::size_of::<crate::hist::HistItem>() <= 32);
}

#[doc(hidden)]
pub fn _unstable_internal_kmeans_bench() -> impl FnMut() {
    use crate::pal::PalF;
    use crate::pal::PalPop;

    let attr = new();
    let mut h = hist::Histogram::new(&attr);

    let e = (0..10000u32).map(|i| HistogramEntry {
        count: i.wrapping_mul(17) % 12345,
        color: RGBA::new(i as u8, (i.wrapping_mul(7) >> 2) as u8, (i.wrapping_mul(11) >> 11) as u8, 255),
    }).collect::<Vec<_>>();

    h.add_colors(&e, 0.).unwrap();
    let mut hist = h.finalize_builder(0.45455).unwrap();

    let lut = pal::gamma_lut(0.45455);
    let mut p = PalF::new();
    for i in 0..=255 {
        p.push(pal::f_pixel::from_rgba(&lut, RGBA::new(i|7, i, i, 255)), PalPop::new(1.));
    }

    move || {
        kmeans::Kmeans::iteration(&mut hist, &mut p, false).unwrap();
    }
}

trait PushInCapacity<T> {
    fn push_in_cap(&mut self, val: T);
}

impl<T> PushInCapacity<T> for Vec<T> {
    #[track_caller]
    #[inline(always)]
    fn push_in_cap(&mut self, val: T) {
        debug_assert!(self.capacity() != self.len());
        if self.capacity() != self.len() {
            self.push(val);
        }
    }
}

/// Rust is too conservative about sorting floats.
/// This library uses only finite values, so they're sortable.
#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
#[repr(transparent)]
struct OrdFloat<T>(pub(crate) T);

impl Eq for OrdFloat<f32> {
}

impl Ord for OrdFloat<f32> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal) }
}

impl Eq for OrdFloat<f64> {
}

impl Ord for OrdFloat<f64> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal) }
}

impl OrdFloat<f32> {
    pub fn new(v: f32) -> Self {
        debug_assert!(v.is_finite());
        Self(v)
    }
}

impl OrdFloat<f64> {
    pub fn new64(v: f64) -> Self {
        debug_assert!(v.is_finite());
        Self(v)
    }
}

#[test]
fn test_fixed_colors() {
    let attr = Attributes::new();
    let mut h = Histogram::new(&attr);
    let tmp = (0..128).map(|c| HistogramEntry {
        color: RGBA::new(c,c,c,255),
        count: 1,
    }).collect::<Vec<_>>();
    h.add_colors(&tmp, 0.).unwrap();
    for f in 200..255 {
        h.add_fixed_color(RGBA::new(f,f,f,255), 0.).unwrap();
    }
    let mut r = h.quantize(&attr).unwrap();
    let pal = r.palette();

    for (i, c) in (200..255).enumerate() {
        assert_eq!(pal[i], RGBA::new(c,c,c,255));
    }

    for c in 0..128 {
        assert!(pal[55..].iter().any(|&p| p == RGBA::new(c,c,c,255)));
    }
}
