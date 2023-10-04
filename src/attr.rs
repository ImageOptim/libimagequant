use crate::error::Error;
use crate::hist::Histogram;
use crate::image::Image;
use crate::pal::PalLen;
use crate::pal::MAX_COLORS;
use crate::pal::RGBA;
use crate::quant::{mse_to_quality, quality_to_mse, QuantizationResult};
use crate::remap::DitherMapMode;
use std::sync::Arc;

/// Starting point and settings for the quantization process
#[derive(Clone)]
pub struct Attributes {
    pub(crate) max_colors: PalLen,
    target_mse: f64,
    max_mse: Option<f64>,
    kmeans_iteration_limit: f64,
    kmeans_iterations: u16,
    feedback_loop_trials: u16,
    pub(crate) max_histogram_entries: u32,
    min_posterization_output: u8,
    min_posterization_input: u8,
    pub(crate) last_index_transparent: bool,
    pub(crate) use_contrast_maps: bool,
    pub(crate) single_threaded_dithering: bool,
    pub(crate) use_dither_map: DitherMapMode,
    speed: u8,
    pub(crate) progress_stage1: u8,
    pub(crate) progress_stage2: u8,
    pub(crate) progress_stage3: u8,

    progress_callback: Option<Arc<dyn Fn(f32) -> ControlFlow + Send + Sync>>,
    log_callback: Option<Arc<dyn Fn(&Attributes, &str) + Send + Sync>>,
    log_flush_callback: Option<Arc<dyn Fn(&Attributes) + Send + Sync>>,
}

impl Attributes {
    /// New handle for library configuration
    ///
    /// See also [`Attributes::new_image()`]
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        let mut attr = Self {
            target_mse: 0.,
            max_mse: None,
            max_colors: MAX_COLORS as PalLen,
            last_index_transparent: false,
            kmeans_iteration_limit: 0.,
            max_histogram_entries: 0,
            min_posterization_output: 0,
            min_posterization_input: 0,
            kmeans_iterations: 0,
            feedback_loop_trials: 0,
            use_contrast_maps: false,
            use_dither_map: DitherMapMode::None,
            single_threaded_dithering: false,
            speed: 0,
            progress_stage1: 0,
            progress_stage2: 0,
            progress_stage3: 0,
            progress_callback: None,
            log_callback: None,
            log_flush_callback: None,
        };
        let _ = attr.set_speed(4);
        attr
    }

    /// Make an image from RGBA pixels.
    ///
    /// The `pixels` argument can be `Vec<RGBA>`, or `Box<[RGBA]>` or `&[RGBA]`.
    /// See [`Attributes::new_image_borrowed`] for a non-copying alternative.
    ///
    /// Use 0.0 for gamma if the image is sRGB (most images are).
    #[inline]
    pub fn new_image<VecRGBA>(&self, pixels: VecRGBA, width: usize, height: usize, gamma: f64) -> Result<Image<'static>, Error> where VecRGBA: Into<Box<[RGBA]>> {
        Image::new(self, pixels, width, height, gamma)
    }

    /// Generate palette for the image
    pub fn quantize(&self, image: &mut Image<'_>) -> Result<QuantizationResult, Error> {
        let mut hist = Histogram::new(self);
        hist.add_image(self, image)?;
        hist.quantize_internal(self, false)
    }

    /// It's better to use `set_quality()`
    #[inline]
    pub fn set_max_colors(&mut self, colors: u32) -> Result<(), Error> {
        if !(2..=256).contains(&colors) {
            return Err(Error::ValueOutOfRange);
        }
        self.max_colors = colors as PalLen;
        Ok(())
    }

    /// Range 0-100, roughly like JPEG.
    ///
    /// If the minimum quality can't be met, the quantization will be aborted with an error.
    ///
    /// Default is min 0, max 100, which means best effort, and never aborts the process.
    ///
    /// If max is less than 100, the library will try to use fewer colors.
    /// Images with fewer colors are not always smaller, due to increased dithering it causes.
    pub fn set_quality(&mut self, minimum: u8, target: u8) -> Result<(), Error> {
        if !(0..=100).contains(&target) || target < minimum {
            return Err(Error::ValueOutOfRange);
        }
        if target < 30 {
            self.verbose_print("  warning: quality set too low");
        }
        self.target_mse = quality_to_mse(target);
        self.max_mse = Some(quality_to_mse(minimum));
        Ok(())
    }

    /// 1-10.
    ///
    /// Faster speeds generate images of lower quality, but may be useful
    /// for real-time generation of images.
    ///
    /// The default is 4.
    #[inline]
    pub fn set_speed(&mut self, value: i32) -> Result<(), Error> {
        if !(1..=10).contains(&value) {
            return Err(Error::ValueOutOfRange);
        }
        let mut iterations = (8 - value).max(0) as u16;
        iterations += iterations * iterations / 2;
        self.kmeans_iterations = iterations;
        self.kmeans_iteration_limit = 1. / f64::from(1 << (23 - value));
        self.feedback_loop_trials = (56 - 9 * value).max(0) as _;
        self.max_histogram_entries = ((1 << 17) + (1 << 18) * (10 - value)) as _;
        self.min_posterization_input = if value >= 8 { 1 } else { 0 };
        self.use_dither_map = if value <= 6 { DitherMapMode::Enabled } else { DitherMapMode::None };
        if self.use_dither_map != DitherMapMode::None && value < 3 {
            self.use_dither_map = DitherMapMode::Always;
        }
        self.use_contrast_maps = (value <= 7) || self.use_dither_map != DitherMapMode::None;
        self.single_threaded_dithering = value == 1;
        self.speed = value as u8;
        self.progress_stage1 = if self.use_contrast_maps { 20 } else { 8 };
        if self.feedback_loop_trials < 2 {
            self.progress_stage1 += 30;
        }
        self.progress_stage3 = (50 / (1 + value)) as u8;
        self.progress_stage2 = 100 - self.progress_stage1 - self.progress_stage3;
        Ok(())
    }

    /// Number of least significant bits to ignore.
    ///
    /// Useful for generating palettes for VGA, 15-bit textures, or other retro platforms.
    #[inline]
    pub fn set_min_posterization(&mut self, value: u8) -> Result<(), Error> {
        if !(0..=4).contains(&value) {
            return Err(Error::ValueOutOfRange);
        }
        self.min_posterization_output = value;
        Ok(())
    }

    /// Returns number of bits of precision truncated
    #[inline(always)]
    #[must_use]
    pub fn min_posterization(&self) -> u8 {
        self.min_posterization_output
    }

    /// Return currently set speed/quality trade-off setting
    #[inline(always)]
    #[must_use]
    pub fn speed(&self) -> u32 {
        self.speed.into()
    }

    /// Return max number of colors set
    #[inline(always)]
    #[must_use]
    pub fn max_colors(&self) -> u32 {
        self.max_colors.into()
    }

    /// Reads values set with `set_quality`
    #[must_use]
    pub fn quality(&self) -> (u8, u8) {
        (
            self.max_mse.map_or(0, mse_to_quality),
            mse_to_quality(self.target_mse),
        )
    }

    /// Describe dimensions of a slice of RGBA pixels
    ///
    /// Use 0.0 for gamma if the image is sRGB (most images are).
    #[inline]
    pub fn new_image_borrowed<'pixels>(&self, bitmap: &'pixels [RGBA], width: usize, height: usize, gamma: f64) -> Result<Image<'pixels>, Error> {
        Image::new_borrowed(self, bitmap, width, height, gamma)
    }

    /// Like `new_image_stride_borrowed`, but makes a copy of the pixels.
    ///
    /// The `pixels` argument can be `Vec<RGBA>`, or `Box<[RGBA]>` or `&[RGBA]`.
    #[inline]
    pub fn new_image_stride<VecRGBA>(&self, pixels: VecRGBA, width: usize, height: usize, stride: usize, gamma: f64) -> Result<Image<'static>, Error> where VecRGBA: Into<Box<[RGBA]>> {
        Image::new_stride(self, pixels, width, height, stride, gamma)
    }

    #[doc(hidden)]
    #[deprecated(note = "use new_image_stride")]
    #[cold]
    pub fn new_image_stride_copy(&self, bitmap: &[RGBA], width: usize, height: usize, stride: usize, gamma: f64) -> Result<Image<'static>, Error> {
        self.new_image_stride(bitmap, width, height, stride, gamma)
    }

    /// Set callback function to be called every time the library wants to print a message.
    ///
    /// To share data with the callback, use `Arc` or `Atomic*` types and `move ||` closures.
    #[inline]
    pub fn set_log_callback<F: Fn(&Attributes, &str) + Send + Sync + 'static>(&mut self, callback: F) {
        self.verbose_printf_flush();
        self.log_callback = Some(Arc::new(callback));
    }

    /// Callback for flushing output (if you buffer messages, that's the time to flush those buffers)
    #[inline]
    pub fn set_log_flush_callback<F: Fn(&Attributes) + Send + Sync + 'static>(&mut self, callback: F) {
        self.verbose_printf_flush();
        self.log_flush_callback = Some(Arc::new(callback));
    }

    /// Set callback function to be called every time the library makes a progress.
    /// It can be used to cancel operation early.
    ///
    /// To share data with the callback, use `Arc` or `Atomic*` types and `move ||` closures.
    #[inline]
    pub fn set_progress_callback<F: Fn(f32) -> ControlFlow + Send + Sync + 'static>(&mut self, callback: F) {
        self.progress_callback = Some(Arc::new(callback));
    }

    /// Move transparent color to the last entry in the palette
    ///
    /// This is less efficient for PNG, but required by some broken software
    #[inline(always)]
    pub fn set_last_index_transparent(&mut self, is_last: bool) {
        self.last_index_transparent = is_last;
    }

    // true == abort
    #[inline]
    #[must_use]
    pub(crate) fn progress(self: &Attributes, percent: f32) -> bool {
        if let Some(f) = &self.progress_callback {
            f(percent) == ControlFlow::Break
        } else {
            false
        }
    }

    #[inline(always)]
    pub(crate) fn verbose_print(self: &Attributes, msg: impl AsRef<str>) {
        fn _print(a: &Attributes, msg: &str) {
            if let Some(f) = &a.log_callback {
                f(a, msg);
            }
        }
        _print(self, msg.as_ref());
    }

    #[inline]
    pub(crate) fn verbose_printf_flush(self: &Attributes) {
        if let Some(f) = &self.log_flush_callback {
            f(self);
        }
    }

    #[must_use]
    pub(crate) fn feedback_loop_trials(&self, hist_items: usize) -> u16 {
        let mut feedback_loop_trials = self.feedback_loop_trials;
        if hist_items > 5000 {
            feedback_loop_trials = (feedback_loop_trials * 3 + 3) / 4;
        }
        if hist_items > 25000 {
            feedback_loop_trials = (feedback_loop_trials * 3 + 3) / 4;
        }
        if hist_items > 50000 {
            feedback_loop_trials = (feedback_loop_trials * 3 + 3) / 4;
        }
        if hist_items > 100_000 {
            feedback_loop_trials = (feedback_loop_trials * 3 + 3) / 4;
        }
        feedback_loop_trials
    }

    /// `max_mse`, `target_mse`, user asked for perfect quality
    pub(crate) fn target_mse(&self, hist_items_len: usize) -> (Option<f64>, f64, bool) {
        let max_mse = self.max_mse.map(|mse| mse * if hist_items_len <= MAX_COLORS { 0.33 } else { 1. });
        let aim_for_perfect_quality = self.target_mse == 0.;
        let mut target_mse = self.target_mse.max((f64::from(1 << self.min_posterization_output) / 1024.).powi(2));
        if let Some(max_mse) = max_mse {
            target_mse = target_mse.min(max_mse);
        }
        (max_mse, target_mse, aim_for_perfect_quality)
    }

    /// returns iterations, `iteration_limit`
    #[must_use]
    pub(crate) fn kmeans_iterations(&self, hist_items_len: usize, palette_error_is_known: bool) -> (u16, f64) {
        let mut iteration_limit = self.kmeans_iteration_limit;
        let mut iterations = self.kmeans_iterations;
        if hist_items_len > 5000 {
            iterations = (iterations * 3 + 3) / 4;
        }
        if hist_items_len > 25000 {
            iterations = (iterations * 3 + 3) / 4;
        }
        if hist_items_len > 50000 {
            iterations = (iterations * 3 + 3) / 4;
        }
        if hist_items_len > 100_000 {
            iterations = (iterations * 3 + 3) / 4;
            iteration_limit *= 2.;
        }
        if iterations == 0 && !palette_error_is_known && self.max_mse.is_some() {
            iterations = 1;
        }
        (iterations, iteration_limit)
    }

    #[inline]
    #[must_use]
    pub(crate) fn posterize_bits(&self) -> u8 {
        self.min_posterization_output.max(self.min_posterization_input)
    }
}

impl Drop for Attributes {
    fn drop(&mut self) {
        self.verbose_printf_flush();
    }
}

impl Default for Attributes {
    #[inline(always)]
    fn default() -> Attributes {
        Attributes::new()
    }
}

/// Result of callback in [`Attributes::set_progress_callback`]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum ControlFlow {
    /// Continue processing as normal
    Continue = 1,
    /// Abort processing and fail
    Break = 0,
}

#[test]
fn counters() {
    let mut a = Attributes::new();
    a.set_speed(10).unwrap();
    let (iter, _) = a.kmeans_iterations(1000, false);
    assert_eq!(iter, 0);
    a.set_quality(80, 90).unwrap();
    let (iter, limit) = a.kmeans_iterations(1000, false);
    assert_eq!(iter, 1);
    assert!(limit > 0. && limit < 0.01);

    let (iter, _) = a.kmeans_iterations(1000, true);
    assert_eq!(iter, 0);

    let mut a = Attributes::new();
    a.set_quality(50, 80).unwrap();

    let (max_mse, target_mse, aim_perfect) = a.target_mse(10000);
    let max_mse = max_mse.unwrap();
    assert!(!aim_perfect);
    assert!(target_mse > 0. && target_mse < 0.01);
    assert!(max_mse > 0. && max_mse > target_mse && max_mse < 0.01);
}

#[test]
fn getset() {
    let mut a = Attributes::new();
    assert!(a.set_quality(0, 101).is_err());
    assert!(a.set_quality(50, 49).is_err());
    assert!(a.feedback_loop_trials(1000) > 0);

    let (max_mse, target_mse, aim_perfect) = a.target_mse(10000);
    assert!(aim_perfect);
    assert!(target_mse < 0.0001);
    assert_eq!(max_mse, None);

    a.set_speed(5).unwrap();
    assert_eq!(5, a.speed());
    assert!(a.set_speed(99).is_err());
    assert!(a.set_speed(0).is_err());

    a.set_max_colors(5).unwrap();
    assert_eq!(5, a.max_colors());
    assert!(a.set_max_colors(0).is_err());

    a.set_min_posterization(2).unwrap();
    assert_eq!(2, a.min_posterization());
    assert_eq!(2, a.posterize_bits());
    assert!(a.set_min_posterization(8).is_err());

    let mut a = Attributes::new();
    a.set_speed(10).unwrap();
    assert_eq!(1, a.posterize_bits());
}
