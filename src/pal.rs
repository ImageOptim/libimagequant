use crate::OrdFloat;
use arrayvec::ArrayVec;
use core::iter;
use core::ops::{Deref, DerefMut};
use rgb::prelude::*;

/// 8-bit RGBA in sRGB. This is the only color format *publicly* used by the library.
pub type RGBA = rgb::Rgba<u8>;

#[allow(clippy::upper_case_acronyms)]
pub type ARGBF = rgb::Argb<f32>;

const INTERNAL_GAMMA: f64 = 0.57;
const LIQ_WEIGHT_A: f32 = 0.625;
const LIQ_WEIGHT_R: f32 = 0.5;
const LIQ_WEIGHT_G: f32 = 1.;
const LIQ_WEIGHT_B: f32 = 0.45;

/// This is a fudge factor - reminder that colors are not in 0..1 range any more
const LIQ_WEIGHT_MSE: f64 = 0.45;

/// 4xf32 color using internal gamma.
///
/// ARGB layout is important for x86 SIMD.
/// I've created the newtype wrapper to try a 16-byte alignment, but it didn't improve perf :(
#[cfg_attr(
    any(target_arch = "x86_64", all(target_feature = "neon", target_arch = "aarch64")),
    repr(C, align(16))
)]
#[derive(Debug, Copy, Clone, Default, PartialEq)]
#[allow(non_camel_case_types)]
pub struct f_pixel(pub ARGBF);

impl f_pixel {
    #[cfg(not(any(target_arch = "x86_64", all(target_feature = "neon", target_arch = "aarch64"))))]
    #[inline(always)]
    pub fn diff(&self, other: &f_pixel) -> f32 {
        let alphas = other.0.a - self.0.a;
        let black = self.0 - other.0;
        let white = ARGBF {
            a: 0.,
            r: black.r + alphas,
            g: black.g + alphas,
            b: black.b + alphas,
        };
        (black.r * black.r).max(white.r * white.r) +
        (black.g * black.g).max(white.g * white.g) +
        (black.b * black.b).max(white.b * white.b)
    }

    #[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
    #[inline(always)]
    pub fn diff(&self, other: &Self) -> f32 {
        unsafe {
            use core::arch::aarch64::*;

            let px = vld1q_f32((self as *const Self).cast::<f32>());
            let py = vld1q_f32((other as *const Self).cast::<f32>());

            // y.a - x.a
            let mut alphas = vsubq_f32(py, px);
            alphas = vdupq_laneq_f32(alphas, 0); // copy first to all four

            let mut onblack = vsubq_f32(px, py); // x - y
            let mut onwhite = vaddq_f32(onblack, alphas); // x - y + (y.a - x.a)

            onblack = vmulq_f32(onblack, onblack);
            onwhite = vmulq_f32(onwhite, onwhite);

            let max = vmaxq_f32(onwhite, onblack);

            let mut max_r = [0.; 4];
            vst1q_f32(max_r.as_mut_ptr(), max);

            let mut max_gb = [0.; 4];
            vst1q_f32(max_gb.as_mut_ptr(), vpaddq_f32(max, max));

            // add rgb, not a

            max_r[1] + max_gb[1]
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn diff(&self, other: &f_pixel) -> f32 {
        unsafe {
            use core::arch::x86_64::*;

            let px = _mm_loadu_ps(self as *const f_pixel as *const f32);
            let py = _mm_loadu_ps(other as *const f_pixel as *const f32);

            // y.a - x.a
            let mut alphas = _mm_sub_ss(py, px);
            alphas = _mm_shuffle_ps(alphas, alphas, 0); // copy first to all four

            let mut onblack = _mm_sub_ps(px, py); // x - y
            let mut onwhite = _mm_add_ps(onblack, alphas); // x - y + (y.a - x.a)

            onblack = _mm_mul_ps(onblack, onblack);
            onwhite = _mm_mul_ps(onwhite, onwhite);
            let max = _mm_max_ps(onwhite, onblack);

            // the compiler is better at horizontal add than I am
            let mut tmp = [0.; 4];
            _mm_storeu_ps(tmp.as_mut_ptr(), max);

            // add rgb, not a
            let res = tmp[1] + tmp[2] + tmp[3];
            res
        }
    }

    #[inline]
    pub(crate) fn to_rgb(self, gamma: f64) -> RGBA {
        if self.is_fully_transparent() {
            return RGBA::new(0, 0, 0, 0);
        }

        let r = (f64::from(LIQ_WEIGHT_A) / f64::from(LIQ_WEIGHT_R)) as f32 * self.r / self.a;
        let g = (f64::from(LIQ_WEIGHT_A) / f64::from(LIQ_WEIGHT_G)) as f32 * self.g / self.a;
        let b = (f64::from(LIQ_WEIGHT_A) / f64::from(LIQ_WEIGHT_B)) as f32 * self.b / self.a;

        let gamma = (gamma / INTERNAL_GAMMA) as f32;
        debug_assert!(gamma.is_finite());

        // 256, because numbers are in range 1..255.9999… rounded down
        RGBA {
            r: (r.powf(gamma) * 256.) as u8,
            g: (g.powf(gamma) * 256.) as u8,
            b: (b.powf(gamma) * 256.) as u8,
            a: (self.a * (256. / f64::from(LIQ_WEIGHT_A)) as f32) as u8,
        }
    }

    pub fn from_rgba(gamma_lut: &[f32; 256], px: RGBA) -> Self {
        let a = f32::from(px.a) / 255.;
        Self(ARGBF {
            a: a * LIQ_WEIGHT_A,
            r: gamma_lut[px.r as usize] * LIQ_WEIGHT_R * a,
            g: gamma_lut[px.g as usize] * LIQ_WEIGHT_G * a,
            b: gamma_lut[px.b as usize] * LIQ_WEIGHT_B * a,
        })
    }

    #[inline]
    pub(crate) fn is_fully_transparent(self) -> bool {
        self.a < (1. / 255. * f64::from(LIQ_WEIGHT_A)) as f32
    }

    #[inline]
    pub(crate) fn is_fully_opaque(self) -> bool {
        self.a >= (255. / 256. * f64::from(LIQ_WEIGHT_A)) as f32
    }
}

impl Deref for f_pixel {
    type Target = ARGBF;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for f_pixel {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<ARGBF> for f_pixel {
    #[inline(always)]
    fn from(x: ARGBF) -> Self {
        Self(x)
    }
}

/// To keep the data dense, `is_fixed` is stuffed into the sign bit
#[derive(Copy, Clone, Debug)]
pub(crate) struct PalPop(f32);

impl PalPop {
    #[inline(always)]
    pub fn is_fixed(self) -> bool {
        self.0 < 0.
    }

    #[must_use]
    pub fn to_fixed(self) -> Self {
        if self.0 < 0. {
            return self;
        }
        Self(if self.0 > 0. { -self.0 } else { -1. })
    }

    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn new(popularity: f32) -> Self {
        debug_assert!(popularity >= 0.);
        Self(popularity)
    }

    #[inline(always)]
    #[must_use]
    pub fn popularity(self) -> f32 {
        self.0.abs()
    }
}

#[cfg(feature = "large_palettes")]
pub type PalIndex = u16;

#[cfg(not(feature = "large_palettes"))]
pub type PalIndex = u8;

/// This could be increased to support > 256 colors in remapping too
pub type PalIndexRemap = u8;
pub type PalLen = u16;

/// Palettes are stored on the stack, and really large ones will cause stack overflows
pub(crate) const MAX_COLORS: usize = if PalIndex::MAX == 255 { 256 } else { 2048 };

/// A palette of premultiplied ARGB 4xf32 colors in internal gamma
#[derive(Clone)]
pub(crate) struct PalF {
    colors: ArrayVec<f_pixel, MAX_COLORS>,
    pops: ArrayVec<PalPop, MAX_COLORS>,
}

impl PalF {
    #[inline]
    pub fn new() -> Self {
        debug_assert!(PalIndex::MAX as usize + 1 >= MAX_COLORS);
        debug_assert!(PalLen::MAX as usize >= MAX_COLORS);
        Self {
            colors: ArrayVec::default(),
            pops: ArrayVec::default(),
        }
    }

    #[inline(always)]
    pub fn push(&mut self, color: f_pixel, popularity: PalPop) {
        self.pops.push(popularity);
        self.colors.push(color);
    }

    pub fn set(&mut self, idx: usize, color: f_pixel, popularity: PalPop) {
        debug_assert!(idx < self.colors.len() && idx < self.pops.len());

        if let Some(pops_idx) = self.pops.get_mut(idx) {
            *pops_idx = popularity;
        }
        if let Some(colors_idx) = self.colors.get_mut(idx) {
            *colors_idx = color;
        }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[f_pixel] {
        &self.colors
    }

    #[inline(always)]
    pub fn pop_as_slice(&self) -> &[PalPop] {
        &self.pops
    }

    // this is max colors allowed by the user, not just max in the current (candidate/low-quality) palette
    pub(crate) fn with_fixed_colors(mut self, max_colors: PalLen, fixed_colors: &[f_pixel]) -> Self {
        if fixed_colors.is_empty() {
            return self;
        }

        // if using low quality, there's a chance mediancut won't create enough colors in the palette
        let max_fixed_colors = fixed_colors.len().min(max_colors as usize);
        if self.len() < max_fixed_colors {
            let needs_extra = max_fixed_colors - self.len();
            self.colors.extend(fixed_colors.iter().copied().take(needs_extra));
            self.pops.extend(iter::repeat(PalPop::new(0.)).take(needs_extra));
            debug_assert_eq!(self.len(), max_fixed_colors);
        }

        // since the fixed colors were in the histogram, expect them to be in the palette,
        // and change closest existing one to be exact fixed
        for (i, fixed_color) in fixed_colors.iter().enumerate().take(self.len()) {
            let (best_idx, _) = self.colors.iter().enumerate().skip(i).min_by_key(|(_, pal_color)| {
                // not using Nearest, because creation of the index may take longer than naive search once
                OrdFloat::new(pal_color.diff(fixed_color))
            }).expect("logic bug in fixed colors, please report a bug");
            debug_assert!(best_idx >= i);
            self.swap(i, best_idx);
            self.set(i, *fixed_color, self.pops[i].to_fixed());
        }

        debug_assert!(self.colors.iter().zip(fixed_colors).all(|(p, f)| p == f));
        debug_assert!(self.pops.iter().take(fixed_colors.len()).all(|pop| pop.is_fixed()));
        self
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        debug_assert_eq!(self.colors.len(), self.pops.len());
        self.colors.len()
    }

    #[inline(always)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&mut f_pixel, &mut PalPop)> {
        let c = &mut self.colors[..];
        let pop = &mut self.pops[..c.len()];
        c.iter_mut().zip(pop)
    }

    #[cfg_attr(debug_assertions, track_caller)]
    pub(crate) fn swap(&mut self, a: usize, b: usize) {
        self.colors.swap(a, b);
        self.pops.swap(a, b);
    }

    /// Also rounds the input pal
    pub(crate) fn init_int_palette(&mut self, int_palette: &mut Palette, gamma: f64, posterize: u8) {
        let lut = gamma_lut(gamma);
        for ((f_color, f_pop), int_pal) in self.iter_mut().zip(&mut int_palette.entries) {
            let mut px = f_color.to_rgb(gamma)
                .map(move |c| posterize_channel(c, posterize));
            *f_color = f_pixel::from_rgba(&lut, px);
            if px.a == 0 && !f_pop.is_fixed() {
                px.r = 71u8;
                px.g = 112u8;
                px.b = 76u8;
            }
            *int_pal = px;
        }
        int_palette.count = self.len() as _;
    }
}

#[inline]
const fn posterize_channel(color: u8, bits: u8) -> u8 {
    if bits == 0 {
        color
    } else {
        (color & !((1 << bits) - 1)) | (color >> (8 - bits))
    }
}

#[inline(always)]
pub fn gamma_lut(gamma: f64) -> [f32; 256] {
    debug_assert!(gamma > 0.);
    let mut tmp = [0.; 256];
    for (i, t) in tmp.iter_mut().enumerate() {
        *t = ((i as f32) / 255.).powf((INTERNAL_GAMMA / gamma) as f32);
    }
    tmp
}

/// MSE that assumes 0..1 channels scaled to MSE that we have in practice
#[inline]
pub(crate) fn unit_mse_to_internal_mse(internal_mse: f64) -> f64 {
    LIQ_WEIGHT_MSE * internal_mse
}

/// Internal MSE scaled to equivalent in 0..255 pixels
pub(crate) fn internal_mse_to_standard_mse(mse: f64) -> f64 {
    (mse * 65536. / 6.) / LIQ_WEIGHT_MSE
}

/// Not used in the Rust API.
/// RGBA colors obtained from [`QuantizationResult`](crate::QuantizationResult)
#[repr(C)]
#[derive(Clone)]
pub struct Palette {
    /// Number of used colors in the `entries`
    pub count: core::ffi::c_uint,
    /// The colors, up to `count`
    pub entries: [RGBA; MAX_COLORS],
}

impl Deref for Palette {
    type Target = [RGBA];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for Palette {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Palette {
    /// Palette colors
    #[inline(always)]
    #[must_use]
    pub fn as_slice(&self) -> &[RGBA] {
        &self.entries[..self.count as usize]
    }

    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [RGBA] {
        &mut self.entries[..self.count as usize]
    }
}

#[test]
fn diff_test() {
    let a = f_pixel(ARGBF {a: 1., r: 0.2, g: 0.3, b: 0.5});
    let b = f_pixel(ARGBF {a: 1., r: 0.3, g: 0.3, b: 0.5});
    let c = f_pixel(ARGBF {a: 1., r: 1., g: 0.3, b: 0.5});
    let d = f_pixel(ARGBF {a: 0., r: 1., g: 0.3, b: 0.5});
    assert!(a.diff(&b) < b.diff(&c));
    assert!(c.diff(&b) < c.diff(&d));

    let a = f_pixel(ARGBF {a: 1., b: 0.2, r: 0.3, g: 0.5});
    let b = f_pixel(ARGBF {a: 1., b: 0.3, r: 0.3, g: 0.5});
    let c = f_pixel(ARGBF {a: 1., b: 1., r: 0.3, g: 0.5});
    let d = f_pixel(ARGBF {a: 0., b: 1., r: 0.3, g: 0.5});
    assert!(a.diff(&b) < b.diff(&c));
    assert!(c.diff(&b) < c.diff(&d));

    let a = f_pixel(ARGBF {a: 1., g: 0.2, b: 0.3, r: 0.5});
    let b = f_pixel(ARGBF {a: 1., g: 0.3, b: 0.3, r: 0.5});
    let c = f_pixel(ARGBF {a: 1., g: 1., b: 0.3, r: 0.5});
    let d = f_pixel(ARGBF {a: 0., g: 1., b: 0.3, r: 0.5});
    assert!(a.diff(&b) < b.diff(&c));
    assert!(c.diff(&b) < c.diff(&d));
}

#[test]
fn alpha_test() {
    let gamma = gamma_lut(0.45455);
    for (start, end) in [
        (RGBA::new(0,0,0,0), RGBA::new(0,0,0,2)),
        (RGBA::new(0,0,0,253), RGBA::new(0,0,0,255))
    ] {
        let start = f_pixel::from_rgba(&gamma, start).a as f64;
        let end = f_pixel::from_rgba(&gamma, end).a as f64;
        let range = end - start;
        for i in 0..1000 {
            let a = (start + ((i as f64) / 1000. * range)) as f32;
            for a in [a, a.next_up(), a.next_down(), a+1e-6, a-1e-6] {
                let px = f_pixel(ARGBF {a, g: 0., b: 0., r: 0.});
                let rgb = px.to_rgb(0.45455);
                assert_eq!(rgb.a == 0, px.is_fully_transparent(), "not trns!? {px:?}, {rgb:?} {} {}", a / LIQ_WEIGHT_A, a / LIQ_WEIGHT_A * 255.);
                assert_eq!(rgb.a == 255, px.is_fully_opaque(), "not opaque?! {px:?}, {rgb:?} {} {} {}", a / LIQ_WEIGHT_A, a / LIQ_WEIGHT_A * 255., a / LIQ_WEIGHT_A * 256.);
            }
        }
    }
}

#[test]
fn pal_test() {
    let mut p = PalF::new();
    let gamma = gamma_lut(0.45455);
    for i in 0..=255u8 {
        let rgba = RGBA::new(i, i, i, 100 + i / 2);
        p.push(f_pixel::from_rgba(&gamma, rgba), PalPop::new(1.));
        assert_eq!(i as usize + 1, p.len());
        assert_eq!(i as usize + 1, p.pop_as_slice().len());
        assert_eq!(i as usize + 1, p.as_slice().len());
        assert_eq!(i as usize + 1, p.colors.len());
        assert_eq!(i as usize + 1, p.pops.len());
        assert_eq!(i as usize + 1, p.iter_mut().count());
    }

    let mut int_pal = Palette {
        count: 0,
        entries: [RGBA::default(); MAX_COLORS],
    };
    p.init_int_palette(&mut int_pal, 0.45455, 0);

    for i in 0..=255u8 {
        let rgba = p.as_slice()[i as usize].to_rgb(0.45455);
        assert_eq!(rgba, RGBA::new(i, i, i, 100 + i / 2));
        assert_eq!(int_pal[i as usize], RGBA::new(i, i, i, 100 + i / 2));
    }
}

#[test]
#[cfg(feature = "large_palettes")]
fn largepal() {
    let gamma = gamma_lut(0.5);
    let mut p = PalF::new();
    for i in 0..1000 {
        let rgba = RGBA::new(i as u8, (i/2) as u8, (i/4) as u8, 255);
        p.push(f_pixel::from_rgba(&gamma, rgba), PalPop::new(1.));
    }
}
