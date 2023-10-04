use crate::attr::Attributes;
use crate::blur::{liq_blur, liq_max3, liq_min3};
use crate::error::*;
use crate::pal::{f_pixel, PalF, PalIndexRemap, MAX_COLORS, MIN_OPAQUE_A, RGBA};
use crate::remap::DitherMapMode;
use crate::rows::{DynamicRows, PixelsSource};
use crate::seacow::Pointer;
use crate::seacow::RowBitmap;
use crate::seacow::SeaCow;
use crate::PushInCapacity;
use crate::LIQ_HIGH_MEMORY_LIMIT;
use rgb::ComponentMap;
use std::mem::MaybeUninit;

/// Describes image dimensions and pixels for the library
///
/// Create one using [`Attributes::new_image()`].
///
/// All images are internally in the RGBA format.
pub struct Image<'pixels> {
    pub(crate) px: DynamicRows<'pixels, 'pixels>,
    pub(crate) importance_map: Option<Box<[u8]>>,
    pub(crate) edges: Option<Box<[u8]>>,
    pub(crate) dither_map: Option<Box<[u8]>>,
    pub(crate) background: Option<Box<Image<'pixels>>>,
    pub(crate) fixed_colors: Vec<RGBA>,
}

impl<'pixels> Image<'pixels> {
    /// Makes an image from RGBA pixels.
    ///
    /// See the [`rgb`] and [`bytemuck`](https://lib.rs/bytemuck) crates for making `[RGBA]` slices from `[u8]` slices.
    ///
    /// The `pixels` argument can be `Vec<RGBA>`, or `Box<[RGBA]>` or `&[RGBA]`.
    ///
    /// If you want to supply RGB or ARGB pixels, convert them to RGBA first, or use [`Image::new_fn`] to supply your own pixel-swapping function.
    ///
    /// Use `0.` for gamma if the image is sRGB (most images are).
    #[inline(always)]
    pub fn new<VecRGBA>(attr: &Attributes, pixels: VecRGBA, width: usize, height: usize, gamma: f64) -> Result<Self, Error> where VecRGBA: Into<Box<[RGBA]>> {
        Self::new_stride(attr, pixels, width, height, width, gamma)
    }

    /// Describe dimensions of a slice of RGBA pixels.
    ///
    /// Same as [`Image::new`], except it doesn't copy the pixels, but holds a temporary reference instead.
    ///
    /// If you want to supply RGB or ARGB pixels, use [`Image::new_fn`] to supply your own pixel-swapping function.
    ///
    /// See the [`rgb`] and [`bytemuck`](https://lib.rs/bytemuck) crates for making `[RGBA]` slices from `[u8]` slices.
    ///
    /// Use `0.` for gamma if the image is sRGB (most images are).
    #[inline(always)]
    pub fn new_borrowed(attr: &Attributes, pixels: &'pixels [RGBA], width: usize, height: usize, gamma: f64) -> Result<Self, Error> {
        Self::new_stride_borrowed(attr, pixels, width, height, width, gamma)
    }

    /// Generate rows on demand using a callback function.
    ///
    /// The callback function should be cheap (e.g. just byte-swap pixels). The parameters are: line of RGBA pixels (slice's len is equal to image width), and row number (0-indexed).
    /// The callback will be called multiple times per row. May be called from multiple threads at once.
    ///
    /// Use `0.` for gamma if the image is sRGB (most images are).
    ///
    /// ## Safety
    ///
    /// This function is marked as unsafe, because the callback function MUST initialize the entire row (call `write` on every `MaybeUninit` pixel).
    ///
    pub unsafe fn new_fn<F: 'pixels + Fn(&mut [MaybeUninit<RGBA>], usize) + Send + Sync>(attr: &Attributes, convert_row_fn: F, width: usize, height: usize, gamma: f64) -> Result<Self, Error> {
        Image::new_internal(attr, PixelsSource::Callback(Box::new(convert_row_fn)), width as u32, height as u32, gamma)
    }

    pub(crate) fn free_histogram_inputs(&mut self) {
        // importance_map must stay for remapping, because remap performs kmeans on potentially-unimportant pixels
        self.px.free_histogram_inputs();
    }

    pub(crate) fn new_internal(
        attr: &Attributes,
        pixels: PixelsSource<'pixels, 'pixels>,
        width: u32,
        height: u32,
        gamma: f64,
    ) -> Result<Self, Error> {
        if !Self::check_image_size(width, height) {
            return Err(ValueOutOfRange);
        }

        if !(0. ..=1.).contains(&gamma) {
            attr.verbose_print("  error: gamma must be >= 0 and <= 1 (try 1/gamma instead)");
            return Err(ValueOutOfRange);
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

    pub(crate) fn update_dither_map(&mut self, remapped_image: &RowBitmap<'_, PalIndexRemap>, palette: &PalF, uses_background: bool) -> Result<(), Error> {
        if self.edges.is_none() {
            self.contrast_maps()?;
        }
        let mut edges = match self.edges.take() {
            Some(e) => e,
            None => return Ok(()),
        };
        let colors = palette.as_slice();

        let width = self.width();
        let mut prev_row: Option<&[_]> = None;
        let mut rows = remapped_image.rows().zip(edges.chunks_exact_mut(width)).peekable();
        while let Some((this_row, edges)) = rows.next() {
            let mut lastpixel = this_row[0];
            let mut lastcol = 0;
            for (col, px) in this_row.iter().copied().enumerate().skip(1) {
                if uses_background && (colors[px as usize]).a < MIN_OPAQUE_A {
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
                        edges[lastcol] = (f32::from(u16::from(edges[lastcol]) + 128)
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
        self.dither_map = Some(edges);
        Ok(())
    }

    /// Set which pixels are more important (and more likely to get a palette entry)
    ///
    /// The map must be `width`×`height` pixels large. Higher numbers = more important.
    pub fn set_importance_map(&mut self, map: impl Into<Box<[u8]>>) -> Result<(), Error> {
        let map = map.into();
        if map.len() != self.width() * self.height() {
            return Err(BufferTooSmall);
        }
        self.importance_map = Some(map);
        Ok(())
    }

    /// Remap pixels assuming they will be displayed on this background. This is designed for GIF's "keep" mode.
    ///
    /// Pixels that match the background color will be made transparent if there's a fully transparent color available in the palette.
    ///
    /// The background image's pixels must outlive this image.
    pub fn set_background(&mut self, background: Image<'pixels>) -> Result<(), Error> {
        if background.background.is_some() {
            return Err(Unsupported);
        }
        if self.px.width != background.px.width || self.px.height != background.px.height {
            return Err(BufferTooSmall);
        }
        self.background = Some(Box::new(background));
        Ok(())
    }

    /// Reserves a color in the output palette created from this image. It behaves as if the given color was used in the image and was very important.
    ///
    /// The RGB values are assumed to have the same gamma as the image.
    ///
    /// It must be called before the image is quantized.
    ///
    /// Returns error if more than 256 colors are added. If image is quantized to fewer colors than the number of fixed colors added, then excess fixed colors will be ignored.
    pub fn add_fixed_color(&mut self, color: RGBA) -> Result<(), Error> {
        if self.fixed_colors.len() >= MAX_COLORS { return Err(Unsupported); }
        self.fixed_colors.try_reserve(1)?;
        self.fixed_colors.push_in_cap(color);
        Ok(())
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

    #[inline(always)]
    pub(crate) fn gamma(&self) -> Option<f64> {
        if self.px.gamma > 0. { Some(self.px.gamma) } else { None }
    }

    /// Builds two maps:
    ///    `importance_map` - approximation of areas with high-frequency noise, except straight edges. 1=flat, 0=noisy.
    ///    edges - noise map including all edges
    pub(crate) fn contrast_maps(&mut self) -> Result<(), Error> {
        let width = self.width();
        let height = self.height();
        if width < 4 || height < 4 || (3 * width * height) > LIQ_HIGH_MEMORY_LIMIT {
            return Ok(()); // shrug
        }

        let noise = if let Some(n) = self.importance_map.as_deref_mut() { n } else {
            let vec = try_zero_vec(width * height)?;
            self.importance_map.get_or_insert_with(move || vec.into_boxed_slice())
        };

        let edges = if let Some(e) = self.edges.as_mut() { e } else {
            let vec = try_zero_vec(width * height)?;
            self.edges.get_or_insert_with(move || vec.into_boxed_slice())
        };

        let mut rows_iter = self.px.all_rows_f()?.chunks_exact(width);

        let mut next_row = rows_iter.next().ok_or(Error::InvalidPointer)?;
        let mut curr_row = next_row;
        let mut prev_row;

        for (noise_row, edges_row) in noise[..width * height].chunks_exact_mut(width).zip(edges[..width * height].chunks_exact_mut(width)) {
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
                let horiz = (prev + next - curr * 2.).map(f32::abs); // noise is amplified
                let prevl = prev_row[i].0;
                let nextl = next_row[i].0;
                let vert = (prevl + nextl - curr * 2.).map(f32::abs);
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
        let mut tmp = try_zero_vec(width * height)?;
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

    /// Stride is in pixels. Allows defining regions of larger images or images with padding without copying. The stride is in pixels.
    ///
    /// Otherwise the same as [`Image::new_borrowed`].
    #[inline(always)]
    pub fn new_stride_borrowed(attr: &Attributes, pixels: &'pixels [RGBA], width: usize, height: usize, stride: usize, gamma: f64) -> Result<Self, Error> {
        Self::new_stride_internal(attr, SeaCow::borrowed(pixels), width, height, stride, gamma)
    }

    /// Create new image by copying `pixels` to an internal buffer, so that it makes a self-contained type.
    ///
    /// The `pixels` argument can be `Vec<RGBA>`, or `Box<[RGBA]>` or `&[RGBA]`.
    ///
    /// Otherwise the same as [`Image::new_stride_borrowed`].
    #[inline]
    pub fn new_stride<VecRGBA>(attr: &Attributes, pixels: VecRGBA, width: usize, height: usize, stride: usize, gamma: f64) -> Result<Image<'static>, Error> where VecRGBA: Into<Box<[RGBA]>> {
        Self::new_stride_internal(attr, SeaCow::boxed(pixels.into()), width, height, stride, gamma)
    }

    fn new_stride_internal<'a>(attr: &Attributes, pixels: SeaCow<'a, RGBA>, width: usize, height: usize, stride: usize, gamma: f64) -> Result<Image<'a>, Error> {
        let slice = pixels.as_slice();
        if slice.len() < (stride * height + width - stride) {
            attr.verbose_print(format!("Buffer length is {} bytes, which is not enough for {}×{}×4 RGBA bytes", slice.len()*4, stride, height));
            return Err(BufferTooSmall);
        }

        let rows = SeaCow::boxed(slice.chunks(stride).map(|row| Pointer(row.as_ptr())).take(height).collect());
        Image::new_internal(attr, PixelsSource::Pixels { rows, pixels: Some(pixels) }, width as u32, height as u32, gamma)
    }
}

fn try_zero_vec(len: usize) -> Result<Vec<u8>, Error> {
    let mut vec = Vec::new();
    vec.try_reserve_exact(len)?;
    vec.resize(len, 0);
    Ok(vec)
}
