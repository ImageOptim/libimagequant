use crate::hist::{FixedColorsSet, HashColor};
use arrayvec::ArrayVec;
use std::ops::{Deref, DerefMut};
use std::os::raw::c_uint;

/// 8-bit RGBA in sRGB. This is the only color format *publicly* used by the library.
pub type RGBA = rgb::RGBA<u8>;

#[allow(clippy::upper_case_acronyms)]
pub type ARGBF = rgb::alt::ARGB<f32>;

pub const INTERNAL_GAMMA: f64 = 0.57;
pub const LIQ_WEIGHT_A: f32 = 0.625;
pub const LIQ_WEIGHT_R: f32 = 0.5;
pub const LIQ_WEIGHT_G: f32 = 1.;
pub const LIQ_WEIGHT_B: f32 = 0.45;

/// This is a fudge factor - reminder that colors are not in 0..1 range any more
pub const LIQ_WEIGHT_MSE: f64 = 0.45;

pub const MIN_OPAQUE_A: f32 = 1. / 256. * LIQ_WEIGHT_A;
pub const MAX_TRANSP_A: f32 = 255. / 256. * LIQ_WEIGHT_A;

/// 4xf32 color using internal gamma.
///
/// ARGB layout is important for x86 SIMD.
/// I've created the newtype wrapper to try a 16-byte alignment, but it didn't improve perf :(
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Default, PartialEq)]
#[allow(non_camel_case_types)]
pub struct f_pixel(pub ARGBF);

impl f_pixel {
    #[cfg(not(target_arch = "x86_64"))]
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

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn diff(&self, other: &f_pixel) -> f32 {
        unsafe {
            use std::arch::x86_64::*;

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
            _mm_storeu_ps(&mut tmp as *mut _ as *mut f32, max);

            // add rgb, not a
            let res = tmp[1] + tmp[2] + tmp[3];
            res
        }
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_rgb(&self, gamma: f64) -> RGBA {
        if self.a < MIN_OPAQUE_A {
            return RGBA::new(0, 0, 0, 0);
        }

        let r = (LIQ_WEIGHT_A / LIQ_WEIGHT_R) * self.r / self.a;
        let g = (LIQ_WEIGHT_A / LIQ_WEIGHT_G) * self.g / self.a;
        let b = (LIQ_WEIGHT_A / LIQ_WEIGHT_B) * self.b / self.a;
        let a = (256. / LIQ_WEIGHT_A) * self.a;

        let gamma = (gamma / INTERNAL_GAMMA) as f32;

        // 256, because numbers are in range 1..255.9999… rounded down
        RGBA {
            r: (r.powf(gamma) * 256.) as u8,
            g: (g.powf(gamma) * 256.) as u8,
            b: (b.powf(gamma) * 256.) as u8,
            a: a as u8,
        }
    }

    pub fn from_rgba(gamma_lut: &[f32; 256], px: RGBA) -> Self {
        let a = px.a as f32 / 255.;
        Self(ARGBF {
            a: a * LIQ_WEIGHT_A,
            r: gamma_lut[px.r as usize] * LIQ_WEIGHT_R * a,
            g: gamma_lut[px.g as usize] * LIQ_WEIGHT_G * a,
            b: gamma_lut[px.b as usize] * LIQ_WEIGHT_B * a,
        })
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
    pub fn is_fixed(&self) -> bool {
        self.0 < 0.
    }

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
    pub fn popularity(&self) -> f32 {
        self.0.abs()
    }
}

/// This could be increased to support > 256 colors
pub type PalIndex = u8;
pub type PalLen = u16;

pub(crate) const MAX_COLORS: usize = 256;

/// A palette of premultiplied ARGB 4xf32 colors in internal gamma
#[derive(Clone)]
pub(crate) struct PalF {
    colors: ArrayVec<f_pixel, MAX_COLORS>,
    pops: ArrayVec<PalPop, MAX_COLORS>,
}

impl PalF {
    #[inline]
    pub fn new() -> Self {
        debug_assert!(PalLen::MAX as usize >= MAX_COLORS);
        Self {
            colors: ArrayVec::new(),
            pops: ArrayVec::new(),
        }
    }

    #[inline(always)]
    pub fn push(&mut self, color: f_pixel, popularity: PalPop) {
        self.pops.push(popularity);
        self.colors.push(color);
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[f_pixel] {
        &self.colors
    }

    #[inline(always)]
    pub fn pop_as_slice(&self) -> &[PalPop] {
        &self.pops
    }

    pub(crate) fn with_fixed_colors(self, max_colors: PalLen, fixed_colors: &FixedColorsSet) -> PalF {
        if fixed_colors.is_empty() {
            return self;
        }

        let mut new_pal = PalF::new();
        let is_fixed = &PalPop::new(1.).to_fixed();
        let new_colors = fixed_colors.iter().map(move |HashColor(color)| (color, is_fixed))
            .chain(self.iter())
            .take(max_colors as usize);

        for (c, pop) in new_colors {
            new_pal.push(*c, *pop);
        }

        new_pal
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.colors.len() as usize
    }

    #[inline(always)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&mut f_pixel, &mut PalPop)> {
        let c = &mut self.colors[..];
        let pop = &mut self.pops[..c.len()];
        c.iter_mut().zip(pop)
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = (&f_pixel, &PalPop)> + '_ {
        let c = &self.colors[..];
        let pop = &self.pops[..c.len()];
        c.iter().zip(pop)
    }

    pub(crate) fn swap(&mut self, a: usize, b: usize) {
        self.colors.swap(a, b);
        self.pops.swap(a, b);
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

/// RGBA colors obtained from [`QuantizationResult`](crate::QuantizationResult)
#[repr(C)]
pub struct Palette {
    /// Number of used colors in the `entries`
    pub count: c_uint,
    /// The colors, up to `count`
    pub entries: [RGBA; 256],
}

impl std::ops::Deref for Palette {
    type Target = [RGBA];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::ops::DerefMut for Palette {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Palette {
    /// Palette colors
    #[inline(always)]
    pub fn as_slice(&self) -> &[RGBA] {
        &self.entries[..self.count as usize]
    }

    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [RGBA] {
        &mut self.entries[..self.count as usize]
    }
}
