use crate::error::*;
use crate::image::Image;
use crate::pal::PalIndex;
use crate::pal::ARGBF;
use crate::pal::MAX_COLORS;
use crate::pal::{f_pixel, gamma_lut, RGBA};
use crate::quant::QuantizationResult;
use crate::rows::temp_buf;
use crate::rows::DynamicRows;
use crate::Attributes;
use rgb::ComponentSlice;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;

/// Number of pixels in a given color for [`Histogram::add_colors()`]
///
/// Used for building a histogram manually. Otherwise see [`Histogram::add_image()`]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct HistogramEntry {
    /// The color
    pub color: RGBA,
    /// Importance of the color (e.g. number of occurrences)
    pub count: u32,
}

/// Generate one shared palette for multiple images
///
/// If you're converting one image at a time, see [`Attributes::new_image`] instead
pub struct Histogram {
    gamma: Option<f64>,
    fixed_colors: FixedColorsSet,

    /// maps RGBA as u32 to (boosted) count
    hashmap: HashMap<u32, (u32, RGBA), U32Hasher>,

    posterize_bits: u8,
    max_histogram_entries: u32,
}

pub(crate) type FixedColorsSet = HashSet<HashColor, U32Hasher>;

#[derive(Clone)]
pub(crate) struct HistItem {
    pub color: f_pixel,
    pub adjusted_weight: f32,
    pub perceptual_weight: f32,
    /// temporary in median cut
    pub mc_color_weight: f32,
    pub tmp: HistSortTmp,
}

impl HistItem {
    // Safety: just an int, and it's been initialized when constructing the object
    #[inline(always)]
    pub fn mc_sort_value(&self) -> u32 {
        unsafe { self.tmp.mc_sort_value }
    }

    // The u32 has been initialized when constructing the object, and u8/u16 is smaller than that
    #[inline(always)]
    pub fn likely_palette_index(&self) -> PalIndex {
        assert!(std::mem::size_of::<PalIndex>() <= std::mem::size_of::<u32>());
        unsafe { self.tmp.likely_palette_index }
    }
}

impl fmt::Debug for HistItem {
    #[cold]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HistItem")
            .field("color", &self.color)
            .field("adjusted_weight", &self.adjusted_weight)
            .field("perceptual_weight", &self.perceptual_weight)
            .field("color_weight", &self.mc_color_weight)
            .finish()
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union HistSortTmp {
    pub mc_sort_value: u32,
    pub likely_palette_index: PalIndex,
}

impl Histogram {
    /// Creates histogram object that will be used to collect color statistics from multiple images.
    ///
    /// All options should be set on `attr` before the histogram object is created. Options changed later may not have effect.
    #[inline]
    #[must_use]
    pub fn new(attr: &Attributes) -> Self {
        Self {
            posterize_bits: attr.posterize_bits(),
            max_histogram_entries: attr.max_histogram_entries,
            fixed_colors: HashSet::with_hasher(U32Hasher(0)),
            hashmap: HashMap::with_hasher(U32Hasher(0)),
            gamma: None,
        }
    }

    /// "Learns" colors from the image, which will be later used to generate the palette.
    ///
    /// Fixed colors added to the image are also added to the histogram. If the total number of fixed colors exceeds 256,
    /// this function will fail with `LIQ_BUFFER_TOO_SMALL`.
    #[inline(never)]
    pub fn add_image(&mut self, attr: &Attributes, image: &mut Image) -> Result<(), Error> {
        let width = image.width();
        let height = image.height();
        if image.importance_map.is_none() && attr.use_contrast_maps {
            image.contrast_maps()?;
        }

        self.gamma = image.gamma();

        if !image.fixed_colors.is_empty() {
            let lut = gamma_lut(self.gamma.unwrap_or(0.45455));
            self.fixed_colors.extend(image.fixed_colors.iter().copied().enumerate().map(|(idx, c)| {
               HashColor { px: f_pixel::from_rgba(&lut, c), index: idx as _ }
            }));
        }

        if attr.progress(f32::from(attr.progress_stage1) * 0.40) {
            return Err(Aborted); // bow can free the RGBA source if copy has been made in f_pixels
        }

        let posterize_bits = attr.posterize_bits();
        let surface_area = height * width;
        let estimated_colors = (surface_area / (posterize_bits as usize + if surface_area > 512 * 512 { 7 } else { 5 })).min(250_000);
        self.reserve(estimated_colors);

        self.add_pixel_rows(&mut image.px, image.importance_map.as_deref(), posterize_bits)?;
        image.free_histogram_inputs();

        Ok(())
    }

    /// Alternative to `add_image()`. Intead of counting colors in an image, it directly takes an array of colors and their counts.
    ///
    /// This function is only useful if you already have a histogram of the image from another source.
    ///
    /// The gamma may be 0 to mean sRGB. All calls to `add_colors` and `add_fixed_color` should use the same gamma value.
    #[inline(never)]
    pub fn add_colors(&mut self, entries: &[HistogramEntry], gamma: f64) -> Result<(), Error> {
        if entries.is_empty() || entries.len() > 1 << 24 {
            return Err(ValueOutOfRange);
        }

        if !(0. ..1.).contains(&gamma) {
            return Err(ValueOutOfRange);
        }

        if self.gamma.is_none() && gamma > 0. {
            self.gamma = Some(gamma);
        }

        self.reserve(entries.len());

        for e in entries {
            self.add_color(e.color, e.count);
        }

        Ok(())
    }

    /// Add a color guaranteed to be in the final palette
    pub fn add_fixed_color(&mut self, color: RGBA, gamma: f64) -> Result<(), Error> {
        let lut = gamma_lut(if gamma > 0. { gamma } else { 0.45455 });
        let px = f_pixel::from_rgba(&lut, RGBA{r: color.r, g: color.g, b: color.b, a: color.a,});

        if self.fixed_colors.len() >= MAX_COLORS {
            return Err(Unsupported);
        }
        let idx = self.fixed_colors.len();
        self.fixed_colors.insert(HashColor { px, index: idx as _ });
        Ok(())
    }

    /// Generate palette for all images/colors added to the histogram.
    ///
    /// Palette generated using this function won't be improved during remapping.
    /// If you're generating palette for only one image, it's better not to use the `Histogram`.
    #[inline]
    pub fn quantize(&mut self, attr: &Attributes) -> Result<QuantizationResult, Error> {
        self.quantize_internal(attr, true)
    }

    #[inline(never)]
    pub(crate) fn quantize_internal(&mut self, attr: &Attributes, freeze_result_colors: bool) -> Result<QuantizationResult, Error> {
        if self.hashmap.is_empty() && self.fixed_colors.is_empty() {
            return Err(Unsupported);
        }

        if attr.progress(0.) { return Err(Aborted); }
        if attr.progress(f32::from(attr.progress_stage1) * 0.89) {
            return Err(Aborted);
        }

        let gamma = self.gamma.unwrap_or(0.45455);
        let (_, target_mse, _) = attr.target_mse(self.hashmap.len());
        let hist = self.finalize_builder(gamma, target_mse).map_err(|_| OutOfMemory)?;

        attr.verbose_print(format!("  made histogram...{} colors found", hist.items.len()));

        QuantizationResult::new(attr, hist, freeze_result_colors, &self.fixed_colors, gamma)
    }

    #[inline(always)]
    fn add_color(&mut self, rgba: RGBA, boost: u32) {
        if boost == 0 {
            return;
        }

        let px_int = if rgba.a != 0 {
            self.posterize_mask() & unsafe { RGBAInt { rgba }.int }
        } else { 0 };

        self.hashmap.entry(px_int)
            // it can overflow on images over 2^24 pixels large
            .and_modify(move |e| e.0 = e.0.saturating_add(boost))
            .or_insert((boost, rgba));
    }

    fn reserve(&mut self, entries: usize) {
        let new_entries = entries.saturating_sub(self.hashmap.len() / 3); // assume some will be dupes, if called multiple times
        self.hashmap.reserve(new_entries);
    }

    #[inline(always)]
    fn posterize_mask(&self) -> u32 {
        let channel_mask = 255 << self.posterize_bits;
        u32::from_ne_bytes([channel_mask, channel_mask, channel_mask, channel_mask])
    }

    /// optionallys et
    fn init_posterize_bits(&mut self, posterize_bits: u8) {
        if self.posterize_bits >= posterize_bits {
            return;
        }
        self.posterize_bits = posterize_bits;
        let new_posterize_mask = self.posterize_mask();

        let new_size = (self.hashmap.len()/3).max(self.hashmap.capacity()/5);
        let old_hashmap = std::mem::replace(&mut self.hashmap, HashMap::with_capacity_and_hasher(new_size, U32Hasher(0)));
        self.hashmap.extend(old_hashmap.into_iter().map(move |(k, v)| {
            (k & new_posterize_mask, v)
        }));
    }

    pub(crate) fn add_pixel_rows(&mut self, image: &mut DynamicRows<'_, '_>, importance_map: Option<&[u8]>, posterize_bits: u8) -> Result<(), Error> {
        let width = image.width as usize;
        let height = image.height as usize;

        let mut importance_map = importance_map.unwrap_or(&[]).chunks_exact(width).fuse();
        let image_iter = image.rgba_rows_iter()?;

        let mut temp_row = temp_buf(width)?;
        for row in 0..height {
            let pixels_row = &image_iter.row_rgba(&mut temp_row, row)[..width];
            let importance_map = importance_map.next().map(move |m| &m[..width]).unwrap_or(&[]);
            for (col, px) in pixels_row.iter().copied().enumerate() {
                let boost = importance_map.get(col).copied().unwrap_or(255);
                self.add_color(px, boost.into());
            }
        }
        self.init_posterize_bits(posterize_bits);

        if self.hashmap.len() > self.max_histogram_entries as usize && self.posterize_bits < 3 {
            self.init_posterize_bits(self.posterize_bits + 1);
        }
        Ok(())
    }

    pub(crate) fn finalize_builder(&mut self, gamma: f64, target_mse: f64) -> Result<HistogramInternal, Error> {
        debug_assert!(gamma > 0.);

        let mut counts = [0; LIQ_MAXCLUSTER];
        let mut temp = Vec::new();
        temp.try_reserve_exact(self.hashmap.len())?;

        let max_fixed_color_difference = (target_mse / 2.).max(2. / 256. / 256.) as f32;

        let lut = gamma_lut(gamma);

        temp.extend(self.hashmap.values().filter_map(|&(boost, color)| {
            let weight = boost as f32;

            let cluster_index = ((color.r >> 7) << 3) | ((color.g >> 7) << 2) | ((color.b >> 7) << 1) | (color.a >> 7);
            let color = f_pixel::from_rgba(&lut, color);

            // fixed colors are always included in the palette, so it would be wasteful to duplicate them in palette from histogram
            // FIXME: removes fixed colors from histogram (could be done better by marking them as max importance instead)
            for HashColor { px, .. } in &self.fixed_colors {
                if color.diff(px) < max_fixed_color_difference {
                    return None;
                }
            }

            counts[cluster_index as usize] += 1;

            Some(TempHistItem { color, weight, cluster_index })
        }));

        let mut clusters = [Cluster { begin: 0, end: 0 }; LIQ_MAXCLUSTER];
        let mut next_begin = 0;
        for (cluster, count) in clusters.iter_mut().zip(counts) {
            cluster.begin = next_begin;
            cluster.end = next_begin;
            next_begin += count;
        }

        let mut items = Vec::new();
        items.try_reserve_exact(temp.len())?;
        items.resize(temp.len(), HistItem {
            color: if cfg!(debug_assertions) { f_pixel( ARGBF { r:f32::NAN, g:f32::NAN, b:f32::NAN, a:f32::NAN } ) } else { f_pixel::default() },
            adjusted_weight: if cfg!(debug_assertions) { f32::NAN } else { 0. },
            perceptual_weight: if cfg!(debug_assertions) { f32::NAN } else { 0. },
            mc_color_weight: if cfg!(debug_assertions) { f32::NAN } else { 0. },
            tmp: HistSortTmp { mc_sort_value: if cfg!(debug_assertions) { !0 } else { 0 } },
        });
        let mut items = items.into_boxed_slice();

        // Limit perceptual weight to 1/10th of the image surface area to prevent
        // a single color from dominating all others.
        let max_perceptual_weight = 0.1 * (temp.iter().map(|t| f64::from(t.weight)).sum::<f64>() / 256.) as f32;

        let mut total_perceptual_weight = 0.;
        for temp_item in temp {
            let cluster = &mut clusters[temp_item.cluster_index as usize];
            let next_index = cluster.end as usize;
            cluster.end += 1;

            let weight = (temp_item.weight * (1. / 256.)).min(max_perceptual_weight);
            total_perceptual_weight += f64::from(weight);

            items[next_index].color = temp_item.color;
            items[next_index].perceptual_weight = weight;
            items[next_index].adjusted_weight = weight;
        }

        Ok(HistogramInternal { items, total_perceptual_weight, clusters })
    }
}

#[derive(Copy, Clone)]
struct TempHistItem {
    color: f_pixel, // FIXME: RGBA would be more efficient?
    weight: f32,
    cluster_index: u8,
}

#[repr(C)]
union RGBAInt {
    rgba: RGBA,
    int: u32,
}

/// Clusters form initial boxes for quantization, to ensure extreme colors are better represented
pub const LIQ_MAXCLUSTER: usize = 16;

pub(crate) struct HistogramInternal {
    pub items: Box<[HistItem]>,
    pub total_perceptual_weight: f64,
    pub clusters: [Cluster; LIQ_MAXCLUSTER],
}

// Pre-grouped colors
#[derive(Copy, Clone, Debug)]
pub(crate) struct Cluster {
    pub begin: u32,
    pub end: u32,
}

// Simple deterministic hasher for the color hashmap
impl std::hash::BuildHasher for U32Hasher {
    type Hasher = Self;
    #[inline(always)]
    fn build_hasher(&self) -> Self {
        Self(0)
    }
}

pub(crate) struct U32Hasher(pub u32);
impl std::hash::Hasher for U32Hasher {
    // magic constant from fxhash. For a single 32-bit key that's all it needs!
    #[inline(always)]
    fn finish(&self) -> u64 { (self.0 as u64).wrapping_mul(0x517cc1b727220a95) }
    #[inline(always)]
    fn write_u32(&mut self, i: u32) { self.0 = i; }

    fn write(&mut self, _bytes: &[u8]) { unimplemented!() }
    fn write_u8(&mut self, _i: u8) { unimplemented!() }
    fn write_u16(&mut self, _i: u16) { unimplemented!() }
    fn write_u64(&mut self, _i: u64) { unimplemented!() }
    fn write_u128(&mut self, _i: u128) { unimplemented!() }
    fn write_usize(&mut self, _i: usize) { unimplemented!() }
    fn write_i8(&mut self, _i: i8) { unimplemented!() }
    fn write_i16(&mut self, _i: i16) { unimplemented!() }
    fn write_i32(&mut self, _i: i32) { unimplemented!() }
    fn write_i64(&mut self, _i: i64) { unimplemented!() }
    fn write_i128(&mut self, _i: i128) { unimplemented!() }
    fn write_isize(&mut self, _i: isize) { unimplemented!() }
}

/// libstd's `HashSet` is afraid of NaN.
/// contains color + original index (since hashmap forgets order)
#[derive(PartialEq, Debug)]
pub(crate) struct HashColor { pub px: f_pixel, pub index: PalIndex }

#[allow(clippy::derived_hash_with_manual_eq)]
impl Hash for HashColor {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for c in self.px.as_slice() {
            u32::from_ne_bytes(c.to_ne_bytes()).hash(state);
        }
    }
}

impl Eq for HashColor {
    fn assert_receiver_is_total_eq(&self) {}
}
