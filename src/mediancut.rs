use crate::hist::{HistItem, HistogramInternal};
use crate::pal::{f_pixel, PalF, PalPop};
use crate::pal::{PalLen, ARGBF};
use crate::quant::quality_to_mse;
use crate::PushInCapacity;
use crate::{Error, OrdFloat};
use rgb::ComponentMap;
use rgb::ComponentSlice;
use std::cmp::Reverse;

struct MedianCutter<'hist> {
    boxes: Vec<MBox<'hist>>,
    hist_total_perceptual_weight: f64,
    target_colors: PalLen,
}

struct MBox<'hist> {
    /// Histogram entries that fell into this bucket
    pub colors: &'hist mut [HistItem],
    /// Center color selected to represent the colors
    pub avg_color: f_pixel,
    /// Difference from the average color, per channel, weighed using `adjusted_weight`
    pub variance: ARGBF,
    pub adjusted_weight_sum: f64,
    pub total_error: Option<f64>,
    /// max color difference between avg_color and any histogram entry
    pub max_error: f32,
}

impl<'hist> MBox<'hist> {
    pub fn new(hist: &'hist mut [HistItem]) -> Self {
        let weight_sum = hist.iter().map(|a| {
            debug_assert!(a.adjusted_weight.is_finite());
            debug_assert!(a.adjusted_weight > 0.);
            f64::from(a.adjusted_weight)
        }).sum();
        Self::new_c(hist, weight_sum, weighed_average_color(hist))
    }

    fn new_s(hist: &'hist mut [HistItem], adjusted_weight_sum: f64, other_boxes: &[MBox<'_>]) -> Self {
        debug_assert!(!hist.is_empty());
        let mut avg_color = weighed_average_color(hist);
        // It's possible that an average color will end up being bad for every entry,
        // so prefer picking actual colors so that at least one histogram entry will be satisfied.
        if (hist.len() < 500 && hist.len() > 2) || Self::is_useless_color(&avg_color, hist, other_boxes) {
            avg_color = hist.iter().min_by_key(|a| OrdFloat::new(avg_color.diff(&a.color))).map(|a| a.color).unwrap_or_default();
        }
        Self::new_c(hist, adjusted_weight_sum, avg_color)
    }

    fn new_c(hist: &'hist mut [HistItem], adjusted_weight_sum: f64, avg_color: f_pixel) -> Self {
        let (variance, max_error) = Self::box_stats(hist, &avg_color);
        Self {
            variance,
            max_error,
            avg_color,
            colors: hist,
            adjusted_weight_sum,
            total_error: None,
        }
    }

    /// It's possible that the average color is useless
    fn is_useless_color(new_avg_color: &f_pixel, colors: &[HistItem], other_boxes: &[MBox<'_>]) -> bool {
        colors.iter().all(move |c| {
            let own_box_diff = new_avg_color.diff(&c.color);
            let other_box_is_better = other_boxes.iter()
                .any(move |other| other.avg_color.diff(&c.color) < own_box_diff);

            other_box_is_better
        })
    }

    fn box_stats(hist: &[HistItem], avg_color: &f_pixel) -> (ARGBF, f32) {
        let mut variance = ARGBF::default();
        let mut max_error = 0.;
        for a in hist {
            variance += (avg_color.0 - a.color.0).map(|c| c * c) * a.adjusted_weight;
            let diff = avg_color.diff(&a.color);
            if diff > max_error {
                max_error = diff;
            }
        }
        (variance, max_error)
    }

    pub fn compute_total_error(&mut self) -> f64 {
        let avg = self.avg_color;
        let e = self.colors.iter().map(move |a| f64::from(avg.diff(&a.color)) * f64::from(a.perceptual_weight)).sum::<f64>();
        self.total_error = Some(e);
        e
    }

    pub fn prepare_sort(&mut self) {
        struct ChanVariance {
            pub chan: usize,
            pub variance: f32,
        }

        // Sort dimensions by their variance, and then sort colors first by dimension with the highest variance
        let vars = self.variance.as_slice();
        let mut channels = [
            ChanVariance { chan: 0, variance: vars[0] },
            ChanVariance { chan: 1, variance: vars[1] },
            ChanVariance { chan: 2, variance: vars[2] },
            ChanVariance { chan: 3, variance: vars[3] },
        ];
        channels.sort_by_key(|a| Reverse(OrdFloat::new(a.variance)));

        for a in self.colors.iter_mut() {
            let chans = a.color.as_slice();
            // Only the first channel really matters. But other channels are included, because when trying median cut
            // many times with different histogram weights, I don't want sort randomness to influence the outcome.
            a.tmp.mc_sort_value = (((chans[channels[0].chan] * 65535.) as u32) << 16)
                | ((chans[channels[2].chan] + chans[channels[1].chan] / 2. + chans[channels[3].chan] / 4.) * 65535.) as u32; // box will be split to make color_weight of each side even
        }
    }

    fn median_color(&mut self) -> f_pixel {
        let len = self.colors.len();
        let (_, mid_item, _) = self.colors.select_nth_unstable_by_key(len/2, |a| a.mc_sort_value());
        mid_item.color
    }

    pub fn prepare_color_weight_total(&mut self) -> f64 {
        let median = self.median_color();
        self.colors.iter_mut().map(move |a| {
            let w = median.diff(&a.color).sqrt() * (1. + a.adjusted_weight).sqrt();
            debug_assert!(w.is_finite());
            a.mc_color_weight = w;
            f64::from(w)
        })
        .sum()
    }

    #[inline]
    pub fn split(mut self, other_boxes: &[MBox<'_>]) -> [Self; 2] {
        self.prepare_sort();
        let half_weight = self.prepare_color_weight_total() / 2.;
        // yeah, there's some off-by-one error in there
        let break_at = hist_item_sort_half(self.colors, half_weight).max(1);

        let (left, right) = self.colors.split_at_mut(break_at);
        let left_sum = left.iter().map(|a| f64::from(a.adjusted_weight)).sum();
        let right_sum = self.adjusted_weight_sum - left_sum;

        [MBox::new_s(left, left_sum, other_boxes),
         MBox::new_s(right, right_sum, other_boxes)]
    }
}

#[inline]
fn qsort_pivot(base: &[HistItem]) -> usize {
    let len = base.len();
    if len < 32 {
        return len / 2;
    }
    let mut pivots = [8, len / 2, len - 1];
    // LLVM can't see it's in bounds :(
    pivots.sort_unstable_by_key(move |&idx| unsafe {
        debug_assert!(base.get(idx).is_some());
        base.get_unchecked(idx)
    }.mc_sort_value());
    pivots[1]
}

fn qsort_partition(base: &mut [HistItem]) -> usize {
    let mut r = base.len();
    base.swap(qsort_pivot(base), 0);
    let pivot_value = base[0].mc_sort_value();
    let mut l = 1;
    while l < r {
        if base[l].mc_sort_value() >= pivot_value {
            l += 1;
        } else {
            r -= 1;
            while l < r && base[r].mc_sort_value() <= pivot_value { r -= 1; }
            base.swap(l, r);
        }
    }
    l -= 1;
    base.swap(l, 0);
    l
}

/// sorts the slice to make the sum of weights lower than `weight_half_sum` one side,
/// returns index of the edge between <halfvar and >halfvar parts of the set
#[inline(never)]
fn hist_item_sort_half(mut base: &mut [HistItem], mut weight_half_sum: f64) -> usize {
    let mut base_index = 0;
    if base.is_empty() { return 0; }
    loop {
        let partition = qsort_partition(base);
        let (left, right) = base.split_at_mut(partition + 1); // +1, because pivot stays on the left side
        let left_sum = left.iter().map(|c| f64::from(c.mc_color_weight)).sum::<f64>();
        if left_sum >= weight_half_sum {
            match left.get_mut(..partition) { // trim pivot point, avoid panick branch in []
                Some(left) if !left.is_empty() => { base = left; continue; },
                _ => return base_index,
            }
        }
        weight_half_sum -= left_sum;
        base_index += left.len();
        if !right.is_empty() {
            base = right;
        } else {
            return base_index;
        }
    }
}

impl<'hist> MedianCutter<'hist> {
    fn total_box_error_below_target(&mut self, mut target_mse: f64) -> bool {
        target_mse *= self.hist_total_perceptual_weight;
        let mut total_error = self.boxes.iter().filter_map(|mb| mb.total_error).sum::<f64>();
        if total_error > target_mse {
            return false;
        }
        for mb in self.boxes.iter_mut().filter(|mb| mb.total_error.is_none()) {
            total_error += mb.compute_total_error();
            if total_error > target_mse {
                return false;
            }
        }
        true
    }

    pub fn new(hist: &'hist mut HistogramInternal, target_colors: PalLen) -> Result<Self, Error> {
        let hist_total_perceptual_weight = hist.total_perceptual_weight;

        debug_assert!(hist.clusters[0].begin == 0);
        debug_assert!(hist.clusters.last().unwrap().end as usize == hist.items.len());

        let mut hist_items = &mut hist.items[..];
        let mut boxes = Vec::new();
        boxes.try_reserve(target_colors as usize)?;

        let used_boxes = hist.clusters.iter().filter(|b| b.begin != b.end).count();
        if used_boxes <= target_colors as usize / 3 {
            // boxes are guaranteed to be sorted
            let mut prev_end = 0;
            for b in hist.clusters.iter().filter(|b| b.begin != b.end) {
                let begin = b.begin as usize;
                debug_assert_eq!(begin, prev_end);
                let end = b.end as usize;
                prev_end = end;
                let (this_box, rest) = hist_items.split_at_mut(end - begin);
                hist_items = rest;
                boxes.push_in_cap(MBox::new(this_box));
            }
        } else {
            boxes.push_in_cap(MBox::new(hist_items));
        };

        Ok(Self {
            boxes,
            hist_total_perceptual_weight,
            target_colors,
        })
    }

    fn into_palette(mut self) -> PalF {
        let mut palette = PalF::new();

        for (i, b) in self.boxes.iter_mut().enumerate() {
            b.colors.iter_mut().for_each(move |a| a.tmp.likely_palette_index = i as _);

            // store total color popularity (perceptual_weight is approximation of it)
            let pop = b.colors.iter().map(|a| f64::from(a.perceptual_weight)).sum::<f64>();
            palette.push(b.avg_color, PalPop::new(pop as f32));
        }
        palette
    }

    fn cut(mut self, target_mse: f64, max_mse: f64) -> PalF {
        let max_mse = max_mse.max(quality_to_mse(20));

        while self.boxes.len() < self.target_colors as usize {
            // first splits boxes that exceed quality limit (to have colors for things like odd green pixel),
            // later raises the limit to allow large smooth areas/gradients get colors.
            let fraction_done = self.boxes.len() as f64 / f64::from(self.target_colors);
            let current_max_mse = max_mse + fraction_done * 16. * max_mse;
            let bi = match self.take_best_splittable_box(current_max_mse) {
                Some(bi) => bi,
                None => break,
            };

            self.boxes.extend(bi.split(&self.boxes));

            if self.total_box_error_below_target(target_mse) {
                break;
            }
        }

        self.into_palette()
    }

    fn take_best_splittable_box(&mut self, max_mse: f64) -> Option<MBox<'hist>> {
        self.boxes.iter().enumerate()
            .filter(|(_, b)| b.colors.len() > 1)
            .map(move |(i, b)| {
                let cv = b.variance.r.max(b.variance.g).max(b.variance.b);
                let mut thissum = b.adjusted_weight_sum * f64::from(cv.max(b.variance.a));
                if f64::from(b.max_error) > max_mse {
                    thissum = thissum * f64::from(b.max_error) / max_mse;
                }
                (i, thissum)
            })
            .max_by_key(|&(_, thissum)| OrdFloat::new64(thissum))
            .map(|(i, _)| self.boxes.swap_remove(i))
    }
}

#[inline(never)]
pub(crate) fn mediancut(hist: &mut HistogramInternal, target_colors: PalLen, target_mse: f64, max_mse_per_color: f64) -> Result<PalF, Error> {
    Ok(MedianCutter::new(hist, target_colors)?.cut(target_mse, max_mse_per_color))
}

fn weighed_average_color(hist: &[HistItem]) -> f_pixel {
    debug_assert!(!hist.is_empty());
    let mut t = f_pixel::default();
    let mut sum = 0.;
    for c in hist {
        sum += c.adjusted_weight;
        t.0 += c.color.0 * c.adjusted_weight;
    }
    if sum != 0. {
        t.0 /= sum;
    }
    t
}
