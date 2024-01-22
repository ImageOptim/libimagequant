use crate::attr::{Attributes, ControlFlow};
use crate::error::*;
use crate::hist::HistogramInternal;
use crate::image::Image;
use crate::kmeans::Kmeans;
use crate::mediancut::mediancut;
use crate::pal::{PalIndexRemap, PalF, PalLen, PalPop, Palette, LIQ_WEIGHT_MSE, MAX_COLORS, MAX_TRANSP_A, RGBA};
use crate::remap::{mse_to_standard_mse, DitherMapMode, Remapped};
use crate::remap::{remap_to_palette, remap_to_palette_floyd};
use crate::seacow::RowBitmapMut;
use crate::OrdFloat;
use arrayvec::ArrayVec;
use std::cmp::Reverse;
use std::fmt;
use std::mem::MaybeUninit;

/// Remapping step, computed from [`Attributes::quantize()`]
pub struct QuantizationResult {
    remapped: Option<Box<Remapped>>,
    pub(crate) palette: PalF,
    progress_callback: Option<Box<dyn Fn(f32) -> ControlFlow + Send + Sync>>,
    pub(crate) int_palette: Palette,
    pub(crate) dither_level: f32,
    pub(crate) gamma: f64,
    pub(crate) palette_error: Option<f64>,
    pub(crate) min_posterization_output: u8,
    pub(crate) use_dither_map: DitherMapMode,
    pub(crate) single_threaded_dithering: bool,
}

impl QuantizationResult {
    pub(crate) fn new(attr: &Attributes, hist: HistogramInternal, freeze_result_colors: bool, gamma: f64) -> Result<Self, Error> {
        if attr.progress(f32::from(attr.progress_stage1)) { return Err(Aborted); }
        let (max_mse, target_mse, target_mse_is_zero) = attr.target_mse(hist.items.len());
        let (mut palette, palette_error) = find_best_palette(attr, target_mse, target_mse_is_zero, max_mse, hist)?;
        if freeze_result_colors {
            palette.iter_mut().for_each(|(_, p)| *p = p.to_fixed());
        }
        if attr.progress(f32::from(attr.progress_stage1) + f32::from(attr.progress_stage2) + f32::from(attr.progress_stage3) * 0.95) {
            return Err(Aborted);
        }
        if let (Some(palette_error), Some(max_mse)) = (palette_error, max_mse) {
            if palette_error > max_mse {
                attr.verbose_print(format!(
                    "  image degradation MSE={:0.3} (Q={}) exceeded limit of {:0.3} ({})",
                    mse_to_standard_mse(palette_error),
                    mse_to_quality(palette_error),
                    mse_to_standard_mse(max_mse),
                    mse_to_quality(max_mse)
                ));
                return Err(QualityTooLow);
            }
        }

        sort_palette(attr, &mut palette);

        Ok(Self {
            palette,
            gamma,
            palette_error,
            min_posterization_output: attr.min_posterization(),
            use_dither_map: attr.use_dither_map,
            remapped: None,
            progress_callback: None,
            int_palette: Palette {
                count: 0,
                entries: [Default::default(); MAX_COLORS],
            },
            dither_level: 1.,
            single_threaded_dithering: attr.single_threaded_dithering,
        })
    }

    /// This is 100% redundant and unnecessary. This work is done anyway when remap is called.
    /// However, this can be called before calling `image.set_background()`, so it may allow better parallelization while the background is generated on another thread.
    #[doc(hidden)]
    pub fn optionally_prepare_for_dithering_with_background_set(&mut self, image: &mut Image<'_>, output_buf: &mut [MaybeUninit<PalIndexRemap>]) -> Result<(), Error> {
        let mut output_pixels = RowBitmapMut::new_contiguous(output_buf, image.width());
        Self::optionally_generate_dither_map(self.use_dither_map, image, true, &mut output_pixels, &mut self.palette)?;
        Ok(())
    }

    #[inline(never)]
    pub(crate) fn write_remapped_image_rows_internal(&mut self, image: &mut Image, mut output_pixels: RowBitmapMut<'_, MaybeUninit<PalIndexRemap>>) -> Result<(), Error> {
        let progress_stage1 = if self.use_dither_map != DitherMapMode::None { 20 } else { 0 };
        if self.remap_progress(progress_stage1 as f32 * 0.25) {
            return Err(Error::Aborted);
        }

        image.free_histogram_inputs();

        let mut palette = self.palette.clone();
        let mut remapped = Box::new(Remapped {
            int_palette: Palette { count: 0, entries: [RGBA::default(); MAX_COLORS] },
            palette_error: None,
        });
        if self.dither_level == 0. {
            palette.init_int_palette(&mut remapped.int_palette, self.gamma, self.min_posterization_output);
            remapped.palette_error = Some(remap_to_palette(&mut image.px, image.background.as_deref_mut(), image.importance_map.as_deref(), &mut output_pixels, &mut palette)?.0);
        } else {
            let uses_background = image.background.is_some();
            let dither_map_error = Self::optionally_generate_dither_map(self.use_dither_map, image, uses_background, &mut output_pixels, &mut palette)?;
            if self.remap_progress(progress_stage1 as f32 * 0.5) {
                return Err(Error::Aborted);
            }

            let output_image_is_remapped = dither_map_error.is_some();
            let palette_error = dither_map_error.or(self.palette_error);

            // remapping above was the last chance to do K-Means iteration, hence the final palette is set after remapping
            palette.init_int_palette(&mut remapped.int_palette, self.gamma, self.min_posterization_output);
            remapped.palette_error = palette_error;
            let max_dither_error = (palette_error.unwrap_or(quality_to_mse(80)) * 2.4).max(quality_to_mse(35)) as f32;
            remap_to_palette_floyd(image, output_pixels, &palette, self, max_dither_error, output_image_is_remapped)?;
        }
        self.remapped = Some(remapped);
        Ok(())
    }

    fn optionally_generate_dither_map(use_dither_map: DitherMapMode, image: &mut Image<'_>, uses_background: bool, output_pixels: &mut RowBitmapMut<'_, MaybeUninit<PalIndexRemap>>, palette: &mut PalF) -> Result<Option<f64>, Error> {
        let is_image_huge = (image.px.width * image.px.height) > 2000 * 2000;
        let allow_dither_map = use_dither_map == DitherMapMode::Always || (!is_image_huge && use_dither_map != DitherMapMode::None);
        let generate_dither_map = allow_dither_map && image.dither_map.is_none();
        if !generate_dither_map {
            return Ok(None);
        }

        // If dithering (with dither map) is required, this image is used to find areas that require dithering
        let (palette_error, row_pointers_remapped) = remap_to_palette(&mut image.px, None, image.importance_map.as_deref(), output_pixels, palette)?;
        image.update_dither_map(&row_pointers_remapped, &*palette, uses_background)?;
        Ok(Some(palette_error))
    }

    /// Set to 1.0 to get nice smooth image
    pub fn set_dithering_level(&mut self, value: f32) -> Result<(), Error> {
        if !(0. ..=1.).contains(&value) {
            return Err(ValueOutOfRange);
        }

        self.remapped = None;
        self.dither_level = value;
        Ok(())
    }

    /// The default is sRGB gamma (~1/2.2)
    pub fn set_output_gamma(&mut self, value: f64) -> Result<(), Error> {
        if value <= 0. || value >= 1. {
            return Err(ValueOutOfRange);
        }

        self.remapped = None;
        self.gamma = value;

        Ok(())
    }

    /// Approximate gamma correction value used for the output
    ///
    /// Colors are converted from input gamma to this gamma
    #[inline]
    #[must_use]
    pub fn output_gamma(&self) -> f64 {
        self.gamma
    }

    /// Number 0-100 guessing how nice the input image will look if remapped to this palette
    #[must_use]
    pub fn quantization_quality(&self) -> Option<u8> {
        self.palette_error.map(mse_to_quality)
    }

    /// Approximate mean square error of the palette
    #[must_use]
    pub fn quantization_error(&self) -> Option<f64> {
        self.palette_error.map(mse_to_standard_mse)
    }

    /// Approximate mean square error of the palette used for the most recent remapping
    #[must_use]
    pub fn remapping_error(&self) -> Option<f64> {
        self.remapped.as_ref()
            .and_then(|re| re.palette_error)
            .or(self.palette_error)
            .map(mse_to_standard_mse)
    }

    /// Palette remapping error mapped back to 0-100 scale, same as the scale in [`Attributes::set_quality()`]
    #[must_use]
    pub fn remapping_quality(&self) -> Option<u8> {
        self.remapped.as_ref()
            .and_then(|re| re.palette_error)
            .or(self.palette_error)
            .map(mse_to_quality)
    }

    /// The final palette
    ///
    /// It's slighly better if you get palette from the [`remapped()`][Self::remapped] call instead
    #[inline]
    #[must_use]
    pub fn palette(&mut self) -> &[RGBA] {
        self.int_palette().as_slice()
    }

    pub(crate) fn int_palette(&mut self) -> &Palette {
        if let Some(remap) = self.remapped.as_ref() {
            debug_assert!(remap.int_palette.count > 0);
            &remap.int_palette
        } else {
            if self.int_palette.count == 0 {
                self.palette.init_int_palette(&mut self.int_palette, self.gamma, self.min_posterization_output);
            }
            &self.int_palette
        }
    }

    /// Callback called at various point of processing, which gets percentage of progress done,
    /// and can return `ControlFlow::Break` to abort further processing
    #[inline(always)]
    pub fn set_progress_callback<F: Fn(f32) -> ControlFlow + Sync + Send + 'static>(&mut self, callback: F) {
        self.progress_callback = Some(Box::new(callback));
    }

    // true == abort
    pub(crate) fn remap_progress(&self, percent: f32) -> bool {
        if let Some(cb) = &self.progress_callback {
            cb(percent) == ControlFlow::Break
        } else {
            false
        }
    }

    /// Remap image into a palette + indices.
    ///
    /// Returns the palette and a 1-byte-per-pixel uncompressed bitmap
    pub fn remapped(&mut self, image: &mut Image<'_>) -> Result<(Vec<RGBA>, Vec<PalIndexRemap>), Error> {
        let mut buf = Vec::new();
        let pal = self.remap_into_vec(image, &mut buf)?;
        Ok((pal, buf))
    }

    /// Remap image into an existing buffer. Use [`remapped()`][Self::remapped] if you don't have a pre-allocated buffer to reuse.
    ///
    /// Writes 1-byte-per-pixel uncompressed bitmap into the `Vec`.
    ///
    /// Returns the palette.
    #[inline]
    pub fn remap_into_vec(&mut self, image: &mut Image<'_>, buf: &mut Vec<PalIndexRemap>) -> Result<Vec<RGBA>, Error> {
        let len = image.width() * image.height();
        // Capacity is essential here, as it creates uninitialized buffer
        unsafe {
            buf.clear();
            buf.try_reserve_exact(len)?;
            self.remap_into(image, &mut buf.spare_capacity_mut()[..len])?;
            buf.set_len(len);
        }
        Ok(self.palette_vec())
    }

    /// Remap image into an existing buffer.
    ///
    /// This is a low-level call for use when existing memory has to be reused. Use [`remapped()`][Self::remapped] or [`remap_into_vec()`][Self::remap_into_vec] if possible.
    ///
    /// Writes 1-byte-per-pixel uncompressed bitmap into the pre-allocated buffer.
    ///
    /// You should call [`palette()`][Self::palette] _after_ this call, but not before it,
    /// because remapping refines the palette.
    #[inline]
    pub fn remap_into(&mut self, image: &mut Image<'_>, output_buf: &mut [MaybeUninit<PalIndexRemap>]) -> Result<(), Error> {
        let required_size = (image.width()) * (image.height());
        let output_buf = output_buf.get_mut(0..required_size).ok_or(BufferTooSmall)?;

        let rows = RowBitmapMut::new_contiguous(output_buf, image.width());
        self.write_remapped_image_rows_internal(image, rows)
    }

    /// The final palette, copied.
    ///
    /// It's slighly better if you get palette from the [`remapped()`][Self::remapped] call instead
    #[must_use]
    pub fn palette_vec(&mut self) -> Vec<RGBA> {
        let pal = self.palette();
        let mut out: Vec<RGBA> = Vec::new();
        if out.try_reserve_exact(pal.len()).is_ok() {
            out.extend_from_slice(pal);
        }
        out
    }

    /// Expected length of the palette
    ///
    /// Reads the length without finalizing the colors
    pub fn palette_len(&mut self) -> usize {
        self.palette.len()
    }
}

fn sort_palette(attr: &Attributes, palette: &mut PalF) {
    let last_index_transparent = attr.last_index_transparent;

    let mut tmp: ArrayVec<_, {MAX_COLORS}> = palette.iter_mut().map(|(c,p)| (*c, *p)).collect();
    tmp.sort_by_key(|(color, pop)| {
        let is_transparent = color.a <= MAX_TRANSP_A;
        (is_transparent == last_index_transparent, Reverse(OrdFloat::new(pop.popularity())))
    });
    palette.iter_mut().zip(tmp).for_each(|((dcol, dpop), (scol, spop))| {
        *dcol = scol;
        *dpop = spop;
    });

    if last_index_transparent {
        let alpha_index = palette.as_slice().iter().enumerate()
            .filter(|(_, c)| c.a <= MAX_TRANSP_A)
            .min_by_key(|(_, c)| OrdFloat::new(c.a))
            .map(|(i, _)| i);
        if let Some(alpha_index) = alpha_index {
            let last_index = palette.as_slice().len() - 1;
            palette.swap(last_index, alpha_index);
        }
    } else {
        let num_transparent = palette.as_slice().iter().enumerate()
            .filter(|(_, c)| c.a <= MAX_TRANSP_A)
            .map(|(i, _)| i + 1) // num entries, not index
            .max();
        if let Some(num_transparent) = num_transparent {
            attr.verbose_print(format!("  eliminated opaque tRNS-chunk entries...{} entr{} transparent", num_transparent, if num_transparent == 1 { "y" } else { "ies" }));
        }
    }
}

impl fmt::Debug for QuantizationResult {
    #[cold]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QuantizationResult(q={})", self.quantization_quality().unwrap_or(0))
    }
}

/// Repeats mediancut with different histogram weights to find palette with minimum error.
///
///  `feedback_loop_trials` controls how long the search will take. < 0 skips the iteration.
#[allow(clippy::or_fun_call)]
pub(crate) fn find_best_palette(attr: &Attributes, target_mse: f64, target_mse_is_zero: bool, max_mse: Option<f64>, mut hist: HistogramInternal) -> Result<(PalF, Option<f64>), Error> {
    // hist.items includes fixed colors already
    let few_input_colors = hist.items.len() <= attr.max_colors as usize;
    // actual target_mse passed to this method has extra diff from posterization
    if few_input_colors && target_mse_is_zero {
        return Ok(palette_from_histogram(&hist, attr.max_colors));
    }

    let mut max_colors = attr.max_colors;
    let total_trials = attr.feedback_loop_trials(hist.items.len()) as i16;
    let mut trials_left = total_trials;
    let mut best_palette = None;
    let mut target_mse_overshoot = if total_trials > 0 { 1.05 } else { 1. };
    let mut fails_in_a_row = 0;
    let mut palette_error = None;
    let mut palette = loop {
        let max_mse_per_color = target_mse.max(palette_error.unwrap_or(quality_to_mse(1))).max(quality_to_mse(51)) * 1.2;
        let mut new_palette = mediancut(&mut hist, max_colors, target_mse * target_mse_overshoot, max_mse_per_color)?
            .with_fixed_colors(attr.max_colors, &hist.fixed_colors);

        let stage_done = 1. - (f32::from(trials_left.max(0)) / f32::from(total_trials + 1)).powi(2);
        let overall_done = f32::from(attr.progress_stage1) + stage_done * f32::from(attr.progress_stage2);
        attr.verbose_print(format!("  selecting colors...{}%", (100. * stage_done) as u8));

        if trials_left <= 0 { break Some(new_palette); }

        let first_run_of_target_mse = best_palette.is_none() && target_mse > 0.;
        let total_error = Kmeans::iteration(&mut hist, &mut new_palette, !first_run_of_target_mse)?;
        if best_palette.is_none() || total_error < palette_error.unwrap_or(f64::MAX) || (total_error <= target_mse && new_palette.len() < max_colors as usize) {
            if total_error < target_mse && total_error > 0. {
                target_mse_overshoot = if (target_mse_overshoot * 1.25) < (target_mse / total_error) {target_mse_overshoot * 1.25 } else {target_mse / total_error }; // if number of colors could be reduced, try to keep it that way
            }
            palette_error = Some(total_error);
            max_colors = max_colors.min(new_palette.len() as PalLen + 1);
            trials_left -= 1;
            fails_in_a_row = 0;
            best_palette = Some(new_palette);
        } else {
            fails_in_a_row += 1;
            target_mse_overshoot = 1.;
            trials_left -= 5 + fails_in_a_row;
        }
        if attr.progress(overall_done) || trials_left <= 0 {
            break best_palette;
        }
    }.ok_or(ValueOutOfRange)?;

    refine_palette(&mut palette, attr, &mut hist, max_mse, &mut palette_error)?;

    Ok((palette, palette_error))
}


fn refine_palette(palette: &mut PalF, attr: &Attributes, hist: &mut HistogramInternal, max_mse: Option<f64>, palette_error: &mut Option<f64>) -> Result<(), Error> {
    let (iterations, iteration_limit) = attr.kmeans_iterations(hist.items.len(), palette_error.is_some());
    if iterations > 0 {
        attr.verbose_print("  moving colormap towards local minimum");
        let mut i = 0;
        while i < iterations {
            let stage_done = f32::from(i) / f32::from(iterations);
            let overall_done = f32::from(attr.progress_stage1) + f32::from(attr.progress_stage2) + stage_done * f32::from(attr.progress_stage3) * 0.89;
            if attr.progress(overall_done) {
                break;
            }

            let pal_err = Kmeans::iteration(hist, palette, false)?;
            debug_assert!(pal_err < 1e20);
            let previous_palette_error = *palette_error;
            *palette_error = Some(pal_err);

            if let Some(previous_palette_error) = previous_palette_error {
                if (previous_palette_error - pal_err).abs() < iteration_limit {
                    break;
                }
            }
            i += if pal_err > max_mse.unwrap_or(1e20) * 1.5 { 2 } else { 1 };
        }
    }
    Ok(())
}

#[cold]
fn palette_from_histogram(hist: &HistogramInternal, max_colors: PalLen) -> (PalF, Option<f64>) {
    let mut hist_pal = PalF::new();
    for item in hist.items.iter() {
        hist_pal.push(item.color, PalPop::new(item.perceptual_weight));
    }

    (hist_pal.with_fixed_colors(max_colors, &hist.fixed_colors), Some(0.))
}

pub(crate) fn quality_to_mse(quality: u8) -> f64 {
    if quality == 0 {
        return 1e20; // + epsilon for floating point errors
    }
    if quality >= 100 { return 0.; }
    let extra_low_quality_fudge = (0.016 / (0.001 + f64::from(quality)) - 0.001).max(0.);
    LIQ_WEIGHT_MSE * (extra_low_quality_fudge + 2.5 / (210. + f64::from(quality)).powf(1.2) * (100.1 - f64::from(quality)) / 100.)
}

pub(crate) fn mse_to_quality(mse: f64) -> u8 {
    for i in (1..101).rev() {
        if mse <= quality_to_mse(i) + 0.000001 { return i; };
    }
    0
}
