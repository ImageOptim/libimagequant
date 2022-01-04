use crate::error::*;
use crate::ffi::MagicTag;
use crate::ffi::{LIQ_FREED_MAGIC, LIQ_HISTOGRAM_MAGIC};
use crate::image::Image;
use crate::pal::PalIndex;
use crate::pal::ARGBF;
use crate::pal::{f_pixel, gamma_lut, RGBA};
use crate::quant::QuantizationResult;
use crate::rows::temp_buf;
use crate::rows::DynamicRows;
use crate::Attributes;
use rgb::ComponentSlice;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
use std::hash::Hash;
use std::os::raw::c_uint;

/// Number of pixels in a given color
///
/// Used if you're building histogram manually. Otherwise see `add_image()`
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct HistogramEntry {
    pub color: RGBA,
    pub count: c_uint,
}

/// Generate one shared palette for multiple images.
pub struct Histogram {
    pub(crate) magic_header: MagicTag,
    gamma: Option<f64>,
    fixed_colors: FixedColorsSet,

    /// maps RGBA as u32 to (boosted) count
    hashmap: HashMap<u32, (u32, RGBA), RgbaHasher>,
    /// how many pixels were counted
    total_area: usize,

    posterize_bits: u8,
    max_histogram_entries: u32,
}

pub(crate) type FixedColorsSet = HashSet<HashColor, RgbaHasher>;

#[derive(Clone)]
pub(crate) struct HistItem {
    pub color: f_pixel,
    pub adjusted_weight: f32,
    pub perceptual_weight: f32,
    pub mc_color_weight: f32,
    pub tmp: HistSortTmp,
}

impl HistItem {
    // Safety: just an int, and it's been initialized when constructing the object
    #[inline(always)]
    pub fn mc_sort_value(&self) -> u32 {
        unsafe { self.tmp.mc_sort_value }
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
            fixed_colors: HashSet::with_hasher(RgbaHasher(0)),
            hashmap: HashMap::with_hasher(RgbaHasher(0)),
            magic_header: LIQ_HISTOGRAM_MAGIC,
            gamma: None,
            total_area: 0,
        }
    }

    /// "Learns" colors from the image, which will be later used to generate the palette.
    ///
    /// Fixed colors added to the image are also added to the histogram. If the total number of fixed colors exceeds 256,
    /// this function will fail with `LIQ_BUFFER_TOO_SMALL`.
    #[inline(never)]
    pub fn add_image(&mut self, attr: &Attributes, image: &mut Image) -> Result<(), liq_error> {
        let width = image.width();
        let height = image.height();
        if image.importance_map.is_none() && attr.use_contrast_maps {
            image.contrast_maps()?;
        }

        self.gamma = Some(image.gamma());

        for c in image.fixed_colors.iter().copied() {
            self.fixed_colors.insert(HashColor(c));
        }

        if attr.progress(attr.progress_stage1 as f32 * 0.40) {
            return Err(LIQ_ABORTED); // bow can free the RGBA source if copy has been made in f_pixels
        }

        let posterize_bits = attr.posterize_bits();
        let surface_area = height * width;
        let estimated_colors = (surface_area / (posterize_bits as usize + if surface_area > 512 * 512 { 7 } else { 5 })).min(250_000);
        self.reserve(estimated_colors);

        let importance_map = image.importance_map.as_ref().map(|m| m.as_slice());

        self.add_pixel_rows(&mut image.px, importance_map, posterize_bits)?;
        image.free_histogram_inputs();

        Ok(())
    }

    /// Alternative to `add_image()`. Intead of counting colors in an image, it directly takes an array of colors and their counts.
    ///
    /// This function is only useful if you already have a histogram of the image from another source.
    #[inline(never)]
    pub fn add_colors(&mut self, entries: &[HistogramEntry], gamma: f64) -> Result<(), liq_error> {
        if entries.is_empty() || entries.len() > 1 << 24 {
            return Err(LIQ_VALUE_OUT_OF_RANGE);
        }

        if !(0. ..1.).contains(&gamma) {
            return Err(LIQ_VALUE_OUT_OF_RANGE);
        }

        self.gamma = Some(if gamma > 0. { gamma } else { 0.45455 });
        self.reserve(entries.len());

        self.total_area += entries.len();
        for e in entries {
            self.add_color(e.color, e.count.try_into().unwrap_or(u16::MAX));
        }

        Ok(())
    }

    /// Add a color guaranteed to be in the final palette
    pub fn add_fixed_color(&mut self, color: RGBA, gamma: f64) -> liq_error {
        let lut = gamma_lut(if gamma > 0. { gamma } else { 0.45455 });
        let px = f_pixel::from_rgba(&lut, RGBA{r: color.r, g: color.g, b: color.b, a: color.a,});

        if self.fixed_colors.len() > 255 {
            return LIQ_UNSUPPORTED;
        }
        self.fixed_colors.insert(HashColor(px));
        LIQ_OK
    }

    /// Generate palette for all images/colors added to the histogram.
    ///
    /// Palette generated using this function won't be improved during remapping.
    /// If you're generating palette for only one image, it's better not to use the `Histogram`.
    #[inline]
    pub fn quantize(&mut self, attr: &Attributes) -> Result<QuantizationResult, liq_error> {
        self.quantize_internal(attr, true)
    }

    #[inline(never)]
    pub(crate) fn quantize_internal(&mut self, attr: &Attributes, freeze_result_colors: bool) -> Result<QuantizationResult, liq_error> {
        if self.hashmap.is_empty() && self.fixed_colors.is_empty() {
            return Err(LIQ_UNSUPPORTED);
        }

        if attr.progress(0.) { return Err(LIQ_ABORTED); }
        if attr.progress(attr.progress_stage1 as f32 * 0.89) {
            return Err(LIQ_ABORTED);
        }

        let gamma = self.gamma.unwrap_or(0.45455);
        let (_, target_mse, _) = attr.target_mse(self.hashmap.len());
        let hist = self.finalize_builder(gamma, target_mse);

        attr.verbose_print(format!("  made histogram...{} colors found", hist.items.len()));

        QuantizationResult::new(attr, hist, freeze_result_colors, &self.fixed_colors, gamma)
    }

    #[inline(always)]
    fn add_color(&mut self, rgba: RGBA, boost: u16) {
        let px_int = if rgba.a != 0 {
            self.posterize_mask() & unsafe { RGBAInt { rgba }.int }
        } else { 0 };

        self.hashmap.entry(px_int)
            .and_modify(move |e| e.0 += boost as u32)
            .or_insert((boost as u32, rgba));
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
        let old_hashmap = std::mem::replace(&mut self.hashmap, HashMap::with_capacity_and_hasher(new_size, RgbaHasher(0)));
        self.hashmap.extend(old_hashmap.into_iter().map(move |(k, v)| {
            (k & new_posterize_mask, v)
        }));
    }

    pub(crate) fn add_pixel_rows(&mut self, image: &mut DynamicRows<'_, '_>, importance_map: Option<&[u8]>, posterize_bits: u8) -> Result<(), liq_error> {
        let width = image.width as usize;
        let height = image.height as usize;
        self.total_area += width * height;

        let mut importance_map = importance_map.unwrap_or(&[]).chunks_exact(width).fuse();
        let image_iter = image.rgba_rows_iter()?;

        let mut temp_row = temp_buf(width);
        for row in 0..height {
            let pixels_row = &image_iter.row_rgba(&mut temp_row, row)[..width];
            let importance_map = importance_map.next().map(move |m| &m[..width]);
            for (col, px) in pixels_row.iter().copied().enumerate() {
                self.add_color(px, importance_map.map(move |map| map[col]).unwrap_or(255) as u16);
            }
        }
        self.init_posterize_bits(posterize_bits);

        if self.hashmap.len() > self.max_histogram_entries as usize && self.posterize_bits < 3 {
            self.init_posterize_bits(self.posterize_bits + 1);
        }
        Ok(())
    }

    pub(crate) fn finalize_builder(&mut self, gamma: f64, target_mse: f64) -> HistogramInternal {
        debug_assert!(gamma > 0.);

        let mut counts = [0; LIQ_MAXCLUSTER];
        let mut temp = Vec::with_capacity(self.hashmap.len());
        // Limit perceptual weight to 1/10th of the image surface area to prevent
        // a single color from dominating all others.
        let max_perceptual_weight = 0.1 * self.total_area as f32;

        let max_fixed_color_difference = (target_mse / 2.).max(2. / 256. / 256.) as f32;

        let lut = gamma_lut(gamma);

        let total_perceptual_weight = self.hashmap.values().map(|&(boost, color)| {
            if boost == 0 && !temp.is_empty() {
                return 0.;
            }
            let cluster_index = (((color.r >> 7) << 3) | ((color.g >> 7) << 2) | ((color.b >> 7) << 1) | (color.a >> 7)) as u8;

            let weight = (boost as f32 / 170.).min(max_perceptual_weight);
            if weight == 0. {
                return 0.;
            }

            let color = f_pixel::from_rgba(&lut, color);

            // fixed colors are always included in the palette, so it would be wasteful to duplicate them in palette from histogram
            // FIXME: removes fixed colors from histogram (could be done better by marking them as max importance instead)
            for HashColor(fixed) in &self.fixed_colors {
                if color.diff(fixed) < max_fixed_color_difference {
                    return 0.;
                }
            }

            counts[cluster_index as usize] += 1;

            temp.push(TempHistItem {
                color, cluster_index, weight,
            });
            weight as f64
        }).sum::<f64>();

        let mut clusters = [Cluster { begin: 0, end: 0 }; LIQ_MAXCLUSTER];
        let mut next_begin = 0;
        for (cluster, count) in clusters.iter_mut().zip(counts) {
            cluster.begin = next_begin;
            cluster.end = next_begin;
            next_begin += count;
        }

        let mut items = vec![HistItem {
            color: if cfg!(debug_assertions) { f_pixel( ARGBF { r:f32::NAN, g:f32::NAN, b:f32::NAN, a:f32::NAN } ) } else { f_pixel::default() },
            adjusted_weight: if cfg!(debug_assertions) { f32::NAN } else { 0. },
            perceptual_weight: if cfg!(debug_assertions) { f32::NAN } else { 0. },
            mc_color_weight: if cfg!(debug_assertions) { f32::NAN } else { 0. },
            tmp: HistSortTmp { mc_sort_value: 0 },
        }; temp.len()].into_boxed_slice();

        for temp_item in temp {
            let cluster = &mut clusters[temp_item.cluster_index as usize];
            let next_index = cluster.end as usize;
            cluster.end += 1;

            items[next_index].color = temp_item.color;
            items[next_index].perceptual_weight = temp_item.weight;
            items[next_index].adjusted_weight = temp_item.weight;
        }

        HistogramInternal {
            items,
            clusters,
            total_perceptual_weight,
        }
    }
}

#[derive(Copy, Clone)]
struct TempHistItem {
    color: f_pixel, // FIXME: RGBA would be more efficient?
    weight: f32,
    cluster_index: u8,
}

union RGBAInt {
    rgba: RGBA,
    int: u32,
}

impl Drop for Histogram {
    fn drop(&mut self) {
        self.magic_header = LIQ_FREED_MAGIC;
    }
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
impl std::hash::BuildHasher for RgbaHasher {
    type Hasher = Self;
    #[inline(always)]
    fn build_hasher(&self) -> Self {
        Self(0)
    }
}

pub(crate) struct RgbaHasher(pub u32);
impl std::hash::Hasher for RgbaHasher {
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

/// libstd's HashSet is afraid of NaN
#[repr(transparent)]
#[derive(PartialEq, Debug)]
pub(crate) struct HashColor(pub f_pixel);

#[allow(clippy::derive_hash_xor_eq)]
impl Hash for HashColor {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for c in self.0.as_slice() {
            u32::from_ne_bytes(c.to_ne_bytes()).hash(state);
        }
    }
}

impl Eq for HashColor {
    fn assert_receiver_is_total_eq(&self) {}
}
