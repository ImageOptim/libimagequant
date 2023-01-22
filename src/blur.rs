/// Blurs image horizontally (width 2*size+1) and writes it transposed to dst (called twice gives 2d blur)
#[inline(never)]
fn transposing_1d_blur(src: &[u8], dst: &mut [u8], width: usize, height: usize, size: u16) {
    if width < 2 * size as usize + 1 || height < 2 * size as usize + 1 {
        return;
    }

    for (j, row) in src.chunks_exact(width).enumerate() {
        let mut sum = u16::from(row[0]) * size;
        for &v in &row[0..size as usize] {
            sum += u16::from(v);
        }
        for i in 0..size as usize {
            sum -= u16::from(row[0]);
            sum += u16::from(row[i + size as usize]);
            dst[i * height + j] = (sum / (size * 2)) as u8;
        }
        for i in size as usize..width - size as usize {
            sum -= u16::from(row[i - size as usize]);
            sum += u16::from(row[i + size as usize]);
            dst[i * height + j] = (sum / (size * 2)) as u8;
        }
        for i in width - size as usize..width {
            sum -= u16::from(row[i - size as usize]);
            sum += u16::from(row[width - 1]);
            dst[i * height + j] = (sum / (size * 2)) as u8;
        }
    }
}

/// Picks maximum of neighboring pixels (blur + lighten)
#[inline(never)]
pub(crate) fn liq_max3(src: &[u8], dst: &mut [u8], width: usize, height: usize) {
    liq_op3(src, dst, width, height, |a, b| a.max(b));
}

pub(crate) fn liq_op3(src: &[u8], dst: &mut [u8], width: usize, height: usize, op: impl Fn(u8, u8) -> u8) {
    for j in 0..height {
        let row = &src[j * width..][..width];
        let dst = &mut dst[j * width..][..width];
        let prevrow = &src[j.saturating_sub(1) * width..][..width];
        let nextrow = &src[(j + 1).min(height - 1) * width..][..width];
        let mut prev: u8;
        let mut curr = row[0];
        let mut next = row[0];
        for i in 0..width - 1 {
            prev = curr;
            curr = next;
            next = row[i + 1];
            let t1 = op(prev, next);
            let t2 = op(nextrow[i], prevrow[i]);
            dst[i] = op(curr, op(t1, t2));
        }
        let t1 = op(curr, next);
        let t2 = op(nextrow[width - 1], prevrow[width - 1]);
        dst[width - 1] = op(curr, op(t1, t2));
    }
}

/// Picks minimum of neighboring pixels (blur + darken)
#[inline(never)]
pub(crate) fn liq_min3(src: &[u8], dst: &mut [u8], width: usize, height: usize) {
    liq_op3(src, dst, width, height, |a, b| a.min(b));
}

/// Filters src image and saves it to dst, overwriting tmp in the process.
/// Image must be width*height pixels high. Size controls radius of box blur.
pub(crate) fn liq_blur(src_dst: &mut [u8], tmp: &mut [u8], width: usize, height: usize, size: u16) {
    transposing_1d_blur(src_dst, tmp, width, height, size);
    transposing_1d_blur(tmp, src_dst, height, width, size);
}
