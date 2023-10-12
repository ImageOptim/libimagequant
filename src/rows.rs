use crate::error::Error;
use crate::pal::{f_pixel, gamma_lut, RGBA};
use crate::seacow::Pointer;
use crate::seacow::SeaCow;
use crate::LIQ_HIGH_MEMORY_LIMIT;
use std::mem::MaybeUninit;

pub(crate) type RowCallback<'a> = dyn Fn(&mut [MaybeUninit<RGBA>], usize) + Send + Sync + 'a;

pub(crate) enum PixelsSource<'pixels, 'rows> {
    /// The `pixels` field is never read, but it is used to store the rows.
    #[allow(dead_code)]
    Pixels {
        rows: SeaCow<'rows, Pointer<RGBA>>,
        pixels: Option<SeaCow<'pixels, RGBA>>,
    },
    Callback(Box<RowCallback<'rows>>),
}

pub(crate) struct DynamicRows<'pixels, 'rows> {
    pub(crate) width: u32,
    pub(crate) height: u32,
    f_pixels: Option<Box<[f_pixel]>>,
    pixels: PixelsSource<'pixels, 'rows>,
    pub(crate) gamma: f64,
}

pub(crate) struct DynamicRowsIter<'parent, 'pixels, 'rows> {
    px: &'parent DynamicRows<'pixels, 'rows>,
    temp_f_row: Option<Box<[MaybeUninit<f_pixel>]>>,
}

impl<'a, 'pixels, 'rows> DynamicRowsIter<'a, 'pixels, 'rows> {
    #[must_use]
    pub fn row_f<'px>(&'px mut self, temp_row: &mut [MaybeUninit<RGBA>], row: usize) -> &'px [f_pixel] {
        debug_assert_eq!(temp_row.len(), self.px.width as usize);
        if let Some(pixels) = self.px.f_pixels.as_ref() {
            let start = self.px.width as usize * row;
            &pixels[start..start + self.px.width as usize]
        } else {
            let lut = gamma_lut(self.px.gamma);
            let row_pixels = self.px.row_rgba(temp_row, row);

            match self.temp_f_row.as_mut() {
                Some(t) => DynamicRows::convert_row_to_f(t, row_pixels, &lut),
                None => &mut [], // this can't happen
            }
        }
    }

    #[must_use]
    pub fn row_f_shared<'px>(&'px self, temp_row: &mut [MaybeUninit<RGBA>], temp_row_f: &'px mut [MaybeUninit<f_pixel>], row: usize) -> &'px [f_pixel] {
        if let Some(pixels) = self.px.f_pixels.as_ref() {
            &pixels[self.px.width as usize * row..]
        } else {
            let lut = gamma_lut(self.px.gamma);
            let row_pixels = self.px.row_rgba(temp_row, row);

            DynamicRows::convert_row_to_f(temp_row_f, row_pixels, &lut)
        }
    }

    #[must_use]
    pub fn row_rgba<'px>(&'px self, temp_row: &'px mut [MaybeUninit<RGBA>], row: usize) -> &'px [RGBA] {
        self.px.row_rgba(temp_row, row)
    }
}

impl<'pixels,'rows> DynamicRows<'pixels,'rows> {
    #[inline]
    pub(crate) fn new(width: u32, height: u32, pixels: PixelsSource<'pixels, 'rows>, gamma: f64) -> Self {
        debug_assert!(gamma > 0.);
        Self { width, height, f_pixels: None, pixels, gamma }
    }

    fn row_rgba<'px>(&'px self, temp_row: &'px mut [MaybeUninit<RGBA>], row: usize) -> &[RGBA] {
        match &self.pixels {
            PixelsSource::Pixels { rows, .. } => unsafe {
                std::slice::from_raw_parts(rows.as_slice()[row].0, self.width())
            },
            PixelsSource::Callback(cb) => {
                cb(temp_row, row);
                // cb needs to be marked as unsafe, since it's responsible for initialization :(
                unsafe { slice_assume_init_mut(temp_row) }
            }
        }
    }

    fn convert_row_to_f<'f>(row_f_pixels: &'f mut [MaybeUninit<f_pixel>], row_pixels: &[RGBA], gamma_lut: &[f32; 256]) -> &'f mut [f_pixel] {
        assert_eq!(row_f_pixels.len(), row_pixels.len());
        for (dst, src) in row_f_pixels.iter_mut().zip(row_pixels) {
            dst.write(f_pixel::from_rgba(gamma_lut, *src));
        }
        // Safe, just initialized
        unsafe { slice_assume_init_mut(row_f_pixels) }
    }

    #[must_use]
    fn should_use_low_memory(&self) -> bool {
        self.width() * self.height() > LIQ_HIGH_MEMORY_LIMIT / std::mem::size_of::<f_pixel>()
    }

    #[inline]
    fn temp_f_row_for_iter(&self) -> Result<Option<Box<[MaybeUninit<f_pixel>]>>, Error> {
        if self.f_pixels.is_some() {
            return Ok(None);
        }
        Ok(Some(temp_buf(self.width())?))
    }

    pub fn prepare_iter(&mut self, temp_row: &mut [MaybeUninit<RGBA>], allow_steamed: bool) -> Result<(), Error> {
        debug_assert_eq!(temp_row.len(), self.width as _);

        if self.f_pixels.is_some() || (allow_steamed && self.should_use_low_memory()) {
            return Ok(());
        }

        let width = self.width();
        let lut = gamma_lut(self.gamma);
        let mut f_pixels = temp_buf(width * self.height())?;
        for (row, f_row) in f_pixels.chunks_exact_mut(width).enumerate() {
            let row_pixels = self.row_rgba(temp_row, row);
            Self::convert_row_to_f(f_row, row_pixels, &lut);
        }
        // just initialized
        self.f_pixels = Some(unsafe { box_assume_init(f_pixels) });
        Ok(())
    }

    #[inline]
    pub fn rows_iter(&mut self, temp_row: &mut [MaybeUninit<RGBA>]) -> Result<DynamicRowsIter<'_, 'pixels, 'rows>, Error> {
        self.prepare_iter(temp_row, true)?;
        Ok(DynamicRowsIter {
            temp_f_row: self.temp_f_row_for_iter()?,
            px: self,
        })
    }

    /// Call `prepare_iter()` first
    #[inline]
    pub fn rows_iter_prepared(&self) -> Result<DynamicRowsIter<'_, 'pixels, 'rows>, Error> {
        Ok(DynamicRowsIter {
            temp_f_row: self.temp_f_row_for_iter()?,
            px: self,
        })
    }

    #[inline]
    pub fn rgba_rows_iter(&self) -> Result<DynamicRowsIter<'_, 'pixels, 'rows>, Error> {
        // This happens when histogram image is recycled
        if let PixelsSource::Pixels { rows, .. } = &self.pixels {
            if rows.as_slice().is_empty() {
                return Err(Error::Unsupported);
            }
        }
        Ok(DynamicRowsIter { px: self, temp_f_row: None })
    }

    #[inline]
    pub fn all_rows_f(&mut self) -> Result<&[f_pixel], Error> {
        if self.f_pixels.is_some() {
            return Ok(self.f_pixels.as_ref().unwrap()); // borrow-checker :(
        }
        self.prepare_iter(&mut temp_buf(self.width())?, false)?;
        self.f_pixels.as_deref().ok_or(Error::Unsupported)
    }

    /// Not recommended
    #[cfg(feature = "_internal_c_ffi")]
    pub(crate) unsafe fn set_memory_ownership(&mut self, own_rows: bool, own_pixels: bool, free_fn: unsafe extern fn(*mut std::os::raw::c_void)) -> Result<(), Error> {
        if own_rows {
            match &mut self.pixels {
                PixelsSource::Pixels { rows, .. } => rows.make_owned(free_fn),
                PixelsSource::Callback(_) => return Err(Error::ValueOutOfRange),
            }
        }

        if own_pixels {
            let len = self.width() * self.height();
            match &mut self.pixels {
                PixelsSource::Pixels { pixels: Some(pixels), .. } => pixels.make_owned(free_fn),
                PixelsSource::Pixels { pixels, rows } => {
                    // the row with the lowest address is assumed to be at the start of the bitmap
                    let ptr = rows.as_slice().iter().map(|p| p.0).min().ok_or(Error::Unsupported)?;
                    *pixels = Some(SeaCow::c_owned(ptr as *mut _, len, free_fn));
                },
                PixelsSource::Callback(_) => return Err(Error::ValueOutOfRange),
            }
        }
        Ok(())
    }

    pub fn free_histogram_inputs(&mut self) {
        if self.f_pixels.is_some() {
            self.pixels = PixelsSource::Pixels { rows: SeaCow::borrowed(&[]), pixels: None };
        }
    }

    #[inline(always)]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width as usize
    }

    #[inline(always)]
    #[must_use]
    pub fn height(&self) -> usize {
        self.height as usize
    }
}

pub(crate) fn temp_buf<T>(len: usize) -> Result<Box<[MaybeUninit<T>]>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)?;
    unsafe { v.set_len(len) };
    Ok(v.into_boxed_slice())
}

#[test]
fn send() {
    fn is_send<T: Send>() {}
    fn is_sync<T: Sync>() {}
    is_send::<DynamicRows>();
    is_sync::<DynamicRows>();
    is_send::<PixelsSource>();
    is_sync::<PixelsSource>();
}

#[inline(always)]
unsafe fn box_assume_init<T>(s: Box<[MaybeUninit<T>]>) -> Box<[T]> {
    std::mem::transmute(s)
}

#[inline(always)]
unsafe fn slice_assume_init_mut<T>(s: &mut [MaybeUninit<T>]) -> &mut [T] {
    &mut *(s as *mut [MaybeUninit<T>] as *mut [T])
}
