use crate::error::Error;
use crate::image::Image;
use crate::kmeans::Kmeans;
use crate::nearest::Nearest;
use crate::pal::{f_pixel, PalF, PalIndexRemap, Palette, ARGBF, LIQ_WEIGHT_MSE, MIN_OPAQUE_A};
use crate::quant::QuantizationResult;
use crate::rayoff::*;
use crate::rows::{temp_buf, DynamicRows};
use crate::seacow::{RowBitmap, RowBitmapMut};
use std::cell::RefCell;
use std::mem::MaybeUninit;

#[repr(u8)]
#[derive(Eq, PartialEq, Clone, Copy)]
pub enum DitherMapMode {
    None = 0,
    Enabled = 1,
    Always = 2,
}

pub(crate) struct Remapped {
    pub(crate) int_palette: Palette,
    pub(crate) palette_error: Option<f64>,
}

#[inline(never)]
pub(crate) fn remap_to_palette<'x, 'b: 'x>(px: &mut DynamicRows, background: Option<&mut Image<'_>>, importance_map: Option<&[u8]>, output_pixels: &'x mut RowBitmapMut<'b, MaybeUninit<PalIndexRemap>>, palette: &mut PalF) -> Result<(f64, RowBitmap<'x, PalIndexRemap>), Error> {
    let n = Nearest::new(palette)?;
    let colors = palette.as_slice();
    let palette_len = colors.len();
    if palette_len > PalIndexRemap::MAX as usize + 1 {
        return Err(Error::Unsupported);
    }

    let tls = ThreadLocal::new();
    let width = px.width as usize;
    let per_thread_buffers = move || -> Result<_, Error> { Ok(RefCell::new((Kmeans::new(palette_len)?, temp_buf(width)?, temp_buf(width)?, temp_buf(width)?))) };

    let tls_tmp1 = tls.get_or_try(per_thread_buffers)?;
    let mut tls_tmp = tls_tmp1.borrow_mut();

    let input_rows = px.rows_iter(&mut tls_tmp.1)?;
    let (background, transparent_index) = background.map(|background| {
        (Some(background), n.search(&f_pixel::default(), 0).0 as PalIndexRemap)
    })
    .filter(|&(_, transparent_index)| colors[usize::from(transparent_index)].a < MIN_OPAQUE_A)
    .unwrap_or((None, 0));
    let background = background.map(|bg| bg.px.rows_iter(&mut tls_tmp.1)).transpose()?;

    if background.is_some() {
        tls_tmp.0.update_color(f_pixel::default(), 1., transparent_index as _);
    }

    drop(tls_tmp);

    let remapping_error = output_pixels.rows_mut().enumerate().par_bridge().map(|(row, output_pixels_row)| {
        let mut remapping_error = 0.;
        let tls_res = match tls.get_or_try(per_thread_buffers) {
            Ok(res) => res,
            Err(_) => return f64::NAN,
        };
        let (kmeans, temp_row, temp_row_f, temp_row_f_bg) = &mut *tls_res.borrow_mut();

        let output_pixels_row = &mut output_pixels_row[..width];
        let importance_map = importance_map.and_then(|m| m.get(row * width..)).unwrap_or(&[]);
        let row_pixels = &input_rows.row_f_shared(temp_row, temp_row_f, row)[..width];
        let bg_pixels = if let Some(background) = &background  {
            &background.row_f_shared(temp_row, temp_row_f_bg, row)[..width]
        } else { &[] };

        let mut last_match = 0;
        for (col, (inp, out)) in row_pixels.iter().zip(output_pixels_row).enumerate() {
            let (matched, diff) = n.search(inp, last_match as _);
            let matched = matched as PalIndexRemap;
            last_match = matched;
            if let Some(bg) = bg_pixels.get(col) {
                let bg_diff = bg.diff(inp);
                if bg_diff <= diff {
                    remapping_error += f64::from(bg_diff);
                    out.write(transparent_index);
                    continue;
                }
            }
            remapping_error += f64::from(diff);
            out.write(matched);
            let importance = f32::from(importance_map.get(col).copied().unwrap_or(1));
            kmeans.update_color(*inp, importance, matched as _);
        }
        remapping_error
    })
    .sum::<f64>();

    if remapping_error.is_nan() {
        return Err(Error::OutOfMemory);
    }

    if let Some(kmeans) = tls.into_iter()
        .map(|t| RefCell::into_inner(t).0)
        .reduce(Kmeans::merge) { kmeans.finalize(palette); }

    let remapping_error = remapping_error / f64::from(px.width * px.height);
    Ok((remapping_error, unsafe { output_pixels.assume_init() }))
}

fn get_dithered_pixel(dither_level: f32, max_dither_error: f32, thiserr: f_pixel, px: f_pixel) -> f_pixel {
    let s = thiserr.0 * dither_level;
    // This prevents gaudy green pixels popping out of the blue (or red or black! ;)
    let dither_error = s.r * s.r + s.g * s.g + s.b * s.b + s.a * s.a;
    if dither_error < 2. / 256. / 256. {
        // don't dither areas that don't have noticeable error — makes file smaller
        return px;
    }

    let mut ratio: f32 = 1.;
    const MAX_OVERFLOW: f32 = 1.1;
    const MAX_UNDERFLOW: f32 = -0.1;
    // allowing some overflow prevents undithered bands caused by clamping of all channels
    if px.r + s.r > MAX_OVERFLOW {
        ratio = ratio.min((MAX_OVERFLOW - px.r) / s.r);
    } else if px.r + s.r < MAX_UNDERFLOW {
        ratio = ratio.min((MAX_UNDERFLOW - px.r) / s.r);
    }
    if px.g + s.g > MAX_OVERFLOW {
        ratio = ratio.min((MAX_OVERFLOW - px.g) / s.g);
    } else if px.g + s.g < MAX_UNDERFLOW {
        ratio = ratio.min((MAX_UNDERFLOW - px.g) / s.g);
    }
    if px.b + s.b > MAX_OVERFLOW {
        ratio = ratio.min((MAX_OVERFLOW - px.b) / s.b);
    } else if px.b + s.b < MAX_UNDERFLOW {
        ratio = ratio.min((MAX_UNDERFLOW - px.b) / s.b);
    }
    if dither_error > max_dither_error {
        ratio *= 0.8;
    }
    f_pixel(ARGBF {
        a: (px.a + s.a).clamp(0., 1.),
        r: px.r + s.r * ratio,
        g: px.g + s.g * ratio,
        b: px.b + s.b * ratio,
    })
}

/// Uses edge/noise map to apply dithering only to flat areas. Dithering on edges creates jagged lines, and noisy areas are "naturally" dithered.
///
///  If `output_image_is_remapped` is true, only pixels noticeably changed by error diffusion will be written to output image.
#[inline(never)]
pub(crate) fn remap_to_palette_floyd(input_image: &mut Image, mut output_pixels: RowBitmapMut<'_, MaybeUninit<PalIndexRemap>>, palette: &PalF, quant: &QuantizationResult, max_dither_error: f32, output_image_is_remapped: bool) -> Result<(), Error> {
    let progress_stage1 = if quant.use_dither_map != DitherMapMode::None { 20 } else { 0 };

    let width = input_image.width();
    let height = input_image.height();

    let mut temp_row = temp_buf(width)?;

    let dither_map = if quant.use_dither_map != DitherMapMode::None {
        input_image.dither_map.as_deref().or(input_image.edges.as_deref()).unwrap_or(&[])
    } else {
        &[]
    };

    let n = Nearest::new(palette)?;
    let palette = palette.as_slice();

    let mut background = input_image.background.as_mut().map(|bg| {
        bg.px.prepare_iter(&mut temp_row, true)?;
        Ok::<_, Error>(&bg.px)
    }).transpose()?;

    let transparent_index = if background.is_some() { n.search(&f_pixel::default(), 0).0 as PalIndexRemap } else { 0 };
    if background.is_some() && palette[transparent_index as usize].a > MIN_OPAQUE_A {
        background = None;
    }
    // response to this value is non-linear and without it any value < 0.8 would give almost no dithering
    let mut base_dithering_level = (1. - (1. - quant.dither_level) * (1. - quant.dither_level)) * (15. / 16.); // prevent small errors from accumulating
    if !dither_map.is_empty() {
        base_dithering_level *= 1. / 255.; // dither_map is in 0-255 scale
    }

    // when using remapping on top of a background, lots of pixels may be transparent, making poor guesses
    // (guesses are only for speed, don't affect visuals)
    let guess_from_remapped_pixels = output_image_is_remapped && background.is_none();

    input_image.px.prepare_iter(&mut temp_row, true)?;
    let input_image_px = &input_image.px;
    let n = &n;

    // Chunks have overhead, so should be big (more than 2 bring diminishing results). Chunks risk causing seams, so should be tall.
    let num_chunks = if quant.single_threaded_dithering { 1 } else { (width * height / 524_288).min(height / 128).max(if height > 128 {2} else {1}).min(num_cpus()) };
    let chunks = output_pixels.chunks((height + num_chunks - 1) / num_chunks);
    scope(move |s| {
        let mut chunk_start_row = 0;
        for mut chunk in chunks {
            let chunk_len = chunk.len();
            let mut temp_row = temp_buf(width)?;
            let mut input_image_iter = input_image_px.rows_iter_prepared()?;
            let mut background = background.map(|bg| bg.rows_iter_prepared()).transpose()?;
            let mut diffusion = Vec::new();
            let errwidth = width + 2; // +2 saves from checking out of bounds access
            diffusion.try_reserve_exact(errwidth * 2)?;
            diffusion.resize(errwidth * 2, f_pixel::default());

            // restart of dithering creates a seam. this does redundant work to init diffusion state,
            // so that later chunks don't start from scratch
            if chunk_start_row > 2 {
                let mut discard_row = temp_buf(width)?;
                for row in (chunk_start_row - 2) .. chunk_start_row {
                    let row_pixels = input_image_iter.row_f(&mut temp_row, row as _);
                    let bg_pixels = background.as_mut().map(|b| b.row_f(&mut temp_row, row as _)).unwrap_or(&[]);
                    let dither_map = dither_map.get(row * width .. row * width + width).unwrap_or(&[]);
                    let scan_forward = row & 1 == 0;
                    dither_row(row_pixels, &mut discard_row, width as u32, dither_map, base_dithering_level, max_dither_error, n, palette, transparent_index, bg_pixels, guess_from_remapped_pixels, &mut diffusion, scan_forward);
                }
            }
            // parallel remap makes progress not very useful
            if quant.remap_progress(progress_stage1 as f32 + chunk_start_row as f32 * (100. - progress_stage1 as f32) / height as f32) {
                return Err(Error::Aborted);
            }
            s.spawn(move |_| {
                for (chunk_row, output_pixels_row) in chunk.rows_mut().enumerate() {
                    let row = chunk_start_row + chunk_row;
                    let row_pixels = input_image_iter.row_f(&mut temp_row, row as _);
                    let bg_pixels = background.as_mut().map(|b| b.row_f(&mut temp_row, row as _)).unwrap_or(&[]);
                    let dither_map = dither_map.get(row * width .. row * width + width).unwrap_or(&[]);
                    let scan_forward = row & 1 == 0;
                    dither_row(row_pixels, output_pixels_row, width as u32, dither_map, base_dithering_level, max_dither_error, n, palette, transparent_index, bg_pixels, guess_from_remapped_pixels, &mut diffusion, scan_forward);
                }
            });
            chunk_start_row += chunk_len;
        }
        Ok(())
    })
}

#[inline(never)]
fn dither_row(row_pixels: &[f_pixel], output_pixels_row: &mut [MaybeUninit<PalIndexRemap>], width: u32, dither_map: &[u8], base_dithering_level: f32, max_dither_error: f32, n: &Nearest, palette: &[f_pixel], transparent_index: PalIndexRemap, bg_pixels: &[f_pixel], guess_from_remapped_pixels: bool, diffusion: &mut [f_pixel], even_row: bool) {
    let width = width as usize;
    assert_eq!(row_pixels.len(), width);
    assert_eq!(output_pixels_row.len(), width);

    let (thiserr, nexterr) = {
        // +2 saves from checking out of bounds access
        let (d1, d2) = diffusion.split_at_mut(width + 2);
        if even_row { (d1, d2) } else { (d2, d1) }
    };

    nexterr.fill_with(f_pixel::default);

    let mut undithered_bg_used = 0u8;
    let mut last_match = 0;
    for x in 0..width {
        let col = if even_row { x } else { width - 1 - x };
        let thiserr = &mut thiserr[col .. col + 3];
        let nexterr = &mut nexterr[col .. col + 3];
        let input_px = row_pixels[col];

        let mut dither_level = base_dithering_level;
        if let Some(&l) = dither_map.get(col) {
            dither_level *= f32::from(l);
        }

        let spx = get_dithered_pixel(dither_level, max_dither_error, thiserr[1], input_px);
        let guessed_match = if guess_from_remapped_pixels {
            unsafe { output_pixels_row[col].assume_init() }
        } else {
            last_match
        };
        let (matched, dither_diff) = n.search(&spx, guessed_match as _);
        let mut matched = matched  as PalIndexRemap;
        last_match = matched as PalIndexRemap;
        let mut output_px = palette[last_match as usize];
        if let Some(bg_pixel) = bg_pixels.get(col) {
            // if the background makes better match *with* dithering, it's a definitive win
            let bg_for_dither_diff = spx.diff(bg_pixel);
            if bg_for_dither_diff <= dither_diff {
                output_px = *bg_pixel;
                matched = transparent_index;
            } else if undithered_bg_used > 1 {
                // the undithered fallback can cause artifacts when too many undithered pixels accumulate a big dithering error
                // so periodically ignore undithered fallback to prevent that
                undithered_bg_used = 0;
            } else {
                // if dithering is not applied, there's a high risk of creating artifacts (flat areas, error accumulating badly),
                // OTOH poor dithering disturbs static backgrounds and creates oscilalting frames that break backgrounds
                // back and forth in two differently bad ways
                let max_diff = input_px.diff(bg_pixel);
                let dithered_diff = input_px.diff(&output_px);
                // if dithering is worse than natural difference between frames
                // (this rule dithers moving areas, but does not dither static areas)
                if dithered_diff > max_diff {
                    // then see if an undithered color is closer to the ideal
                    let guessed_px = palette[guessed_match as usize];
                    let undithered_diff = input_px.diff(&guessed_px); // If dithering error is crazy high, don't propagate it that much
                    if undithered_diff < max_diff {
                        undithered_bg_used += 1;
                        output_px = guessed_px;
                        matched = guessed_match;
                    }
                }
            }
        }
        output_pixels_row[col].write(matched);
        let mut err = spx.0 - output_px.0;
        // This prevents weird green pixels popping out of the blue (or red or black! ;)
        if err.r * err.r + err.g * err.g + err.b * err.b + err.a * err.a > max_dither_error {
            err *= 0.75;
        }
        if even_row {
            thiserr[2].0 += err * (7. / 16.);
            nexterr[0].0 += err * (3. / 16.);
            nexterr[1].0 += err * (5. / 16.);
            nexterr[2].0 = err * (1. / 16.);
        } else {
            thiserr[0].0 += err * (7. / 16.);
            nexterr[0].0 = err * (1. / 16.);
            nexterr[1].0 += err * (5. / 16.);
            nexterr[2].0 += err * (3. / 16.);
        }
    }
}

pub(crate) fn mse_to_standard_mse(mse: f64) -> f64 {
    (mse * 65536. / 6.) / LIQ_WEIGHT_MSE // parallelized dither map might speed up floyd remapping
}

#[test]
fn send() {
    fn is_send<T: Send>() {}

    is_send::<RowBitmapMut<'_, MaybeUninit<u8>>>();
}

#[test]
fn background_to_nop() {
    use crate::RGBA;
    let pixels: Vec<_> = (0..200*200).map(|n| RGBA::new(n as u8, (n/17) as u8, (n/78) as u8, 255)).collect();

    let mut attr = crate::new();
    let mut img = attr.new_image_borrowed(&pixels, 200, 200, 0.).unwrap();
    let img2 = attr.new_image_borrowed(&pixels, 200, 200, 0.).unwrap();
    img.set_background(img2).unwrap();
    img.add_fixed_color(RGBA::new(0,0,0,0)).unwrap();
    attr.set_max_colors(3).unwrap();
    let mut res = attr.quantize(&mut img).unwrap();
    res.set_dithering_level(0.).unwrap();
    let (_, idx) = res.remapped(&mut img).unwrap();
    let first = idx[0];
    assert!(idx.iter().all(|&x| x == first));

    res.set_dithering_level(1.).unwrap();
    let (_, idx) = res.remapped(&mut img).unwrap();
    let first = idx[0];
    assert!(idx.iter().all(|&x| x == first));
}
