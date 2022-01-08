use std::os::raw::c_void;
use crate::attr::Attributes;
use crate::blur::{liq_blur, liq_max3, liq_min3};
use crate::error::*;
use crate::pal::{f_pixel, gamma_lut, PalF, MIN_OPAQUE_A, RGBA};
use crate::remap::DitherMapMode;
use crate::rows::{DynamicRows, PixelsSource};
use crate::seacow::RowBitmap;
use crate::seacow::SeaCow;
use crate::LIQ_HIGH_MEMORY_LIMIT;
use rgb::ComponentMap;
use std::mem::MaybeUninit;

/// Describes image dimensions for the library.
pub struct Image<'pixels, 'rows> {
    pub(crate) px: DynamicRows<'pixels, 'rows>,
    pub(crate) importance_map: Option<SeaCow<'static, u8>>,
    pub(crate) edges: Option<Box<[u8]>>,
    pub(crate) dither_map: Option<Box<[u8]>>,
    pub(crate) background: Option<Box<Image<'pixels, 'rows>>>,
    pub(crate) fixed_colors: Vec<f_pixel>,
    pub(crate) c_api_free: Option<unsafe extern fn(*mut c_void)>,
}

impl<'pixels, 'rows> Image<'pixels, 'rows> {
    pub(crate) fn free_histogram_inputs(&mut self) {
        self.importance_map = None;
        self.px.free_histogram_inputs();
    }

    pub(crate) fn new_internal(
        attr: &Attributes,
        pixels: PixelsSource<'pixels, 'rows>,
        width: u32,
        height: u32,
        gamma: f64,
    ) -> Result<Self, liq_error> {
        if !Self::check_image_size(width, height) {
            return Err(LIQ_VALUE_OUT_OF_RANGE);
        }

        if !(0. ..=1.).contains(&gamma) {
            attr.verbose_print("  error: gamma must be >= 0 and <= 1 (try 1/gamma instead)");
            return Err(LIQ_VALUE_OUT_OF_RANGE);
        }
        let img = Image {
            px: DynamicRows::new(
                width,
                height,
                pixels,
                if gamma > 0. { gamma } else { 0.45455 },
            ),
            importance_map: None,
            edges: None,
            dither_map: None,
            background: None,
            fixed_colors: Vec::new(),
            c_api_free: attr.c_api_free,
        };
        // if image is huge or converted pixels are not likely to be reused then don't cache converted pixels
        let low_memory_hint = !attr.use_contrast_maps && attr.use_dither_map == DitherMapMode::None;
        let limit = if low_memory_hint { LIQ_HIGH_MEMORY_LIMIT / 8 } else { LIQ_HIGH_MEMORY_LIMIT } / std::mem::size_of::<f_pixel>();
        if (img.width()) * (img.height()) > limit {
            attr.verbose_print("  conserving memory"); // for simplicity of this API there's no explicit pixels argument,
        }
        Ok(img)
    }

    fn check_image_size(width: u32, height: u32) -> bool {
        if width == 0 || height == 0 {
            return false;
        }
        if width.max(height) as usize > i32::MAX as usize ||
            width as usize > isize::MAX as usize / std::mem::size_of::<f_pixel>() / height as usize {
            return false;
        }
        true
    }

    pub(crate) fn update_dither_map(&mut self, remapped_image: &RowBitmap<'_, u8>, palette: &mut PalF) {
        let width = self.width();
        let edges = match self.edges.as_deref_mut() {
            Some(e) => e,
            None => return,
        };
        let colors = palette.as_slice();

        let mut prev_row: Option<&[_]> = None;
        let mut rows = remapped_image.rows().zip(edges.chunks_exact_mut(width)).peekable();
        while let Some((this_row, edges)) = rows.next() {
            let mut lastpixel = this_row[0];
            let mut lastcol = 0;
            for (col, px) in this_row.iter().copied().enumerate().skip(1) {
                if self.background.is_some() && (colors[px as usize]).a < MIN_OPAQUE_A {
                    // Transparency may or may not create an edge. When there's an explicit background set, assume no edge.
                    continue;
                }
                if px != lastpixel || col == width - 1 {
                    let mut neighbor_count = 10 * (col - lastcol);
                    let mut i = lastcol;
                    while i < col {
                        if let Some(prev_row) = prev_row {
                            let pixelabove = prev_row[i];
                            if pixelabove == lastpixel { neighbor_count += 15; };
                        }
                        if let Some((next_row, _)) = rows.peek() {
                            let pixelbelow = next_row[i];
                            if pixelbelow == lastpixel { neighbor_count += 15; };
                        }
                        i += 1;
                    }
                    while lastcol <= col {
                        let e = edges[lastcol];
                        edges[lastcol] = ((e as u16 + 128) as f32
                            * (255. / (255 + 128) as f32)
                            * (1. - 20. / (20 + neighbor_count) as f32))
                            as u8;
                        lastcol += 1;
                    }
                    lastpixel = px;
                }
            }
            prev_row = Some(this_row);
        }
        self.dither_map = self.edges.take();
    }

    /// Remap pixels assuming they will be displayed on this background.
    ///
    /// Pixels that match the background color will be made transparent if there's a fully transparent color available in the palette.
    ///
    /// The background image's pixels must outlive this image
    pub fn set_background(&mut self, background: Image<'pixels, 'rows>) -> Result<(), liq_error> {
        if background.background.is_some() {
            return Err(LIQ_UNSUPPORTED);
        }
        if self.px.width != background.px.width || self.px.height != background.px.height {
            return Err(LIQ_BUFFER_TOO_SMALL);
        }
        self.background = Some(Box::new(background));
        self.dither_map = None;
        Ok(())
    }

    /// Set which pixels are more important (and more likely to get a palette entry)
    ///
    /// The map must be `width`×`height` pixels large. Higher numbers = more important.
    pub fn set_importance_map(&mut self, map: &[u8]) -> Result<(), liq_error> {
        self.importance_map = Some(SeaCow::boxed(map.into()));
        Ok(())
    }

    #[inline]
    pub(crate) fn set_importance_map_raw(&mut self, map: Option<SeaCow<'static, u8>>) {
        self.importance_map = map;
    }

    /// Width of the image in pixels
    #[must_use]
    #[inline(always)]
    pub fn width(&self) -> usize {
        self.px.width as _
    }

    /// Height of the image in pixels
    #[must_use]
    #[inline(always)]
    pub fn height(&self) -> usize {
        self.px.height as _
    }

    /// Reserves a color in the output palette created from this image. It behaves as if the given color was used in the image and was very important.
    ///
    /// RGB values of Color are assumed to have the same gamma as the image.
    ///
    /// It must be called before the image is quantized.
    ///
    /// Returns error if more than 256 colors are added. If image is quantized to fewer colors than the number of fixed colors added, then excess fixed colors will be ignored.
    pub fn add_fixed_color(&mut self, color: RGBA) -> liq_error {
        if self.fixed_colors.len() > 255 { return LIQ_UNSUPPORTED; }
        let lut = gamma_lut(self.px.gamma);
        self.fixed_colors.push(f_pixel::from_rgba(&lut, RGBA {r: color.r, g: color.g, b: color.b, a: color.a}));
        LIQ_OK
    }

    #[inline(always)]
    pub(crate) fn gamma(&self) -> f64 {
        self.px.gamma
    }

    /// Builds two maps:
    ///    importance_map - approximation of areas with high-frequency noise, except straight edges. 1=flat, 0=noisy.
    ///    edges - noise map including all edges
    pub(crate) fn contrast_maps(&mut self) -> Result<(), liq_error> {
        let width = self.width();
        let height = self.height();
        if width < 4 || height < 4 || (3 * width * height) > LIQ_HIGH_MEMORY_LIMIT {
            return Ok(()); // shrug
        }

        let noise = &mut self.importance_map.get_or_insert_with(move || SeaCow::boxed(vec![0; height * width].into_boxed_slice())).as_mut_slice()[..width * height];
        let edges = &mut self.edges.get_or_insert_with(move || vec![0; width * height].into_boxed_slice())[..width * height];

        let mut tmp = vec![0; width * height];

        let mut rows_iter = self.px.all_rows_f()?.chunks_exact(width);

        let mut next_row = rows_iter.next().unwrap();
        let mut curr_row = next_row;
        let mut prev_row;

        for (noise_row, edges_row) in noise.chunks_exact_mut(width).zip(edges.chunks_exact_mut(width)) {
            prev_row = curr_row;
            curr_row = next_row;
            next_row = rows_iter.next().unwrap_or(next_row);
            let mut prev;
            let mut curr = curr_row[0].0;
            let mut next = curr;
            for i in 0..width {
                prev = curr;
                curr = next;
                next = curr_row[(i + 1).min(width - 1)].0;
                // contrast is difference between pixels neighbouring horizontally and vertically
                let horiz = (prev + next - curr * 2.).map(|c| c.abs()); // noise is amplified
                let prevl = prev_row[i].0;
                let nextl = next_row[i].0;
                let vert = (prevl + nextl - curr * 2.).map(|c| c.abs());
                let horiz = horiz.a.max(horiz.r).max(horiz.g.max(horiz.b));
                let vert = vert.a.max(vert.r).max(vert.g.max(vert.b));
                let edge = horiz.max(vert);
                let mut z = edge - (horiz - vert).abs() * 0.5;
                z = 1. - z.max(horiz.min(vert));
                z *= z;
                z *= z;
                // 85 is about 1/3rd of weight (not 0, because noisy pixels still need to be included, just not as precisely).
                noise_row[i] = (80. + z * 176.) as u8;
                edges_row[i] = ((1. - edge) * 256.) as u8;
            }
        }
        // noise areas are shrunk and then expanded to remove thin edges from the map
        liq_max3(noise, &mut tmp, width, height);
        liq_max3(&tmp, noise, width, height);
        liq_blur(noise, &mut tmp, width, height, 3);
        liq_max3(noise, &mut tmp, width, height);
        liq_min3(&tmp, noise, width, height);
        liq_min3(noise, &mut tmp, width, height);
        liq_min3(&tmp, noise, width, height);
        liq_min3(edges, &mut tmp, width, height);
        liq_max3(&tmp, edges, width, height);
        for (edges, noise) in edges.iter_mut().zip(noise) {
            *edges = (*noise).min(*edges);
        }
        Ok(())
    }

    /// Describe dimensions of a slice of RGBA pixels.
    ///
    /// See the [`rgb`] and [`bytemuck`](//lib.rs/bytemuck) crates for making `[RGBA]` slices from `[u8]` slices.
    ///
    /// Use `0.` for gamma if the image is sRGB (most images are).
    #[inline(always)]
    pub fn new(attr: &Attributes, pixels: &'pixels [RGBA], width: usize, height: usize, gamma: f64) -> Result<Self, liq_error> {
        Self::new_stride(attr, pixels, width, height, width, gamma)
    }

    /// Generate rows on demand using a callback function.
    ///
    /// The callback function should be cheap (e.g. just byte-swap pixels). It will be called multiple times per row. May be called from multiple threads at once.
    ///
    /// Use `0.` for gamma if the image is sRGB (most images are).
    ///
    /// ## Safety
    ///
    /// This function is marked as unsafe, because the callback function MUST initialize the entire row (call `write` on every `MaybeUninit` pixel).
    ///
    pub unsafe fn new_fn<F: 'static + Fn(&mut [MaybeUninit<RGBA>], usize) + Send + Sync>(attr: &Attributes, convert_row_fn: F, width: usize, height: usize, gamma: f64) -> Result<Self, liq_error> {
        Image::new_internal(attr, PixelsSource::Callback(Box::new(convert_row_fn)), width as u32, height as u32, gamma)
    }

    /// Stride is in pixels. Allows defining regions of larger images or images with padding without copying.
    ///
    /// Otherwise the same as [`Image::new`].
    #[inline(always)]
    pub fn new_stride(attr: &Attributes, pixels: &'pixels [RGBA], width: usize, height: usize, stride: usize, gamma: f64) -> Result<Self, liq_error> {
        Self::new_stride_internal(attr, SeaCow::borrowed(pixels), width, height, stride, gamma)
    }

    /// Create new image by copying `pixels` to an internal buffer, so that it makes a self-contained type.
    ///
    /// Otherwise the same as [`Image::new_stride`].
    #[inline]
    pub fn new_stride_copy(attr: &Attributes, pixels: &[RGBA], width: usize, height: usize, stride: usize, gamma: f64) -> Result<Image<'static, 'static>, liq_error> {
        Self::new_stride_internal(attr, SeaCow::boxed(pixels.into()), width, height, stride, gamma)
    }

    fn new_stride_internal<'a>(attr: &Attributes, pixels: SeaCow<'a, RGBA>, width: usize, height: usize, stride: usize, gamma: f64) -> Result<Image<'a, 'static>, liq_error> {
        let slice = pixels.as_slice();
        if slice.len() < (stride * height + width - stride) {
            attr.verbose_print(format!("Buffer length is {} bytes, which is not enough for {}×{}×4 RGBA bytes", slice.len()*4, stride, height));
            return Err(LIQ_BUFFER_TOO_SMALL);
        }

        let rows = SeaCow::boxed(slice.chunks(stride).map(|row| row.as_ptr()).collect());
        Image::new_internal(attr, PixelsSource::Pixels { rows, pixels: Some(pixels) }, width as u32, height as u32, gamma)
    }
}
