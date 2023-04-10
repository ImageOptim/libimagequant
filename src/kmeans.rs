use crate::hist::{HistItem, HistogramInternal};
use crate::nearest::Nearest;
use crate::pal::{f_pixel, PalF, PalIndex, PalPop};
use crate::rayoff::*;
use crate::Error;
use rgb::alt::ARGB;
use rgb::ComponentMap;
use std::cell::RefCell;

pub(crate) struct Kmeans {
    averages: Vec<ColorAvg>,
    weighed_diff_sum: f64,
}

#[derive(Copy, Clone, Default)]
struct ColorAvg {
    pub sum: ARGB<f64>,
    pub total: f64,
}

/// K-Means iteration: new palette color is computed from weighted average of colors that map best to that palette entry.
impl Kmeans {
    #[inline]
    pub fn new(pal_len: usize) -> Result<Self, Error> {
        let mut averages = Vec::new();
        averages.try_reserve_exact(pal_len)?;
        averages.resize(pal_len, ColorAvg::default());
        Ok(Self {
            averages,
            weighed_diff_sum: 0.,
        })
    }

    #[inline]
    pub fn update_color(&mut self, px: f_pixel, value: f32, matched: PalIndex) {
        let c = &mut self.averages[matched as usize];
        c.sum += (px.0 * value).map(f64::from);
        c.total += f64::from(value);
    }

    pub fn finalize(self, palette: &mut PalF) -> f64 {
        for (avg, (color, pop)) in self.averages.iter().zip(palette.iter_mut()).filter(|(_, (_, pop))| !pop.is_fixed()) {
            let total = avg.total;
            *pop = PalPop::new(total as f32);
            if total > 0. && color.a != 0. {
                *color = avg.sum.map(move |c| (c / total) as f32).into();
            }
        }
        self.weighed_diff_sum
    }

    #[inline(never)]
    pub(crate) fn iteration(hist: &mut HistogramInternal, palette: &mut PalF, adjust_weight: bool) -> Result<f64, Error> {
        if hist.items.is_empty() {
            return Ok(0.);
        }

        let n = Nearest::new(palette)?;
        let colors = palette.as_slice();
        let len = colors.len();

        let tls = ThreadLocal::new();
        let total = hist.total_perceptual_weight;

        // chunk size is a trade-off between parallelization and overhead
        hist.items.par_chunks_mut(256).for_each({
            let tls = &tls; move |batch| {
            let kmeans = tls.get_or(move || RefCell::new(Kmeans::new(len)));
            if let Ok(ref mut kmeans) = *kmeans.borrow_mut() {
                kmeans.iterate_batch(batch, &n, colors, adjust_weight);
            }
        }});

        let diff = tls.into_iter()
            .map(RefCell::into_inner)
            .reduce(Kmeans::try_merge)
            .transpose()?
            .map_or(0., |kmeans| {
                kmeans.finalize(palette) / total
            });

        replace_unused_colors(palette, hist)?;
        Ok(diff)
    }

    fn iterate_batch(&mut self, batch: &mut [HistItem], n: &Nearest, colors: &[f_pixel], adjust_weight: bool) {
        self.weighed_diff_sum += batch.iter_mut().map(|item| {
            let px = item.color;
            let (matched, mut diff) = n.search(&px, item.likely_palette_index());
            item.tmp.likely_palette_index = matched;
            if adjust_weight {
                let remapped = colors[matched as usize];
                let (_, new_diff) = n.search(&f_pixel(px.0 + px.0 - remapped.0), matched);
                diff = new_diff;
                item.adjusted_weight = (item.perceptual_weight + 2. * item.adjusted_weight) * (0.5 + diff);
            }
            debug_assert!(f64::from(diff) < 1e20);
            self.update_color(px, item.adjusted_weight, matched);
            f64::from(diff * item.perceptual_weight)
        }).sum::<f64>();
    }

    #[inline]
    pub fn merge(mut self, new: Kmeans) -> Kmeans {
        self.weighed_diff_sum += new.weighed_diff_sum;
        self.averages.iter_mut().zip(new.averages).for_each(|(p, n)| {
            p.sum += n.sum;
            p.total += n.total;
        });
        self
    }

    #[inline]
    pub fn try_merge<E>(old: Result<Self, E>, new: Result<Self, E>) -> Result<Self, E> {
        match (old, new) {
            (Ok(old), Ok(new)) => Ok(Kmeans::merge(old, new)),
            (Err(e), _) | (_, Err(e)) => Err(e),
        }
    }
}

/// kmeans may have merged or obsoleted some palette entries.
/// This replaces these entries with histogram colors that are currently least-fitting the palette.
fn replace_unused_colors(palette: &mut PalF, hist: &HistogramInternal) -> Result<(), Error> {
    for pal_idx in 0..palette.len() {
        let pop = palette.pop_as_slice()[pal_idx];
        if pop.popularity() == 0. && !pop.is_fixed() {
            let n = Nearest::new(palette)?;
            let mut worst = None;
            let mut worst_diff = 0.;
            let colors = palette.as_slice();
            // the search is just for diff, ignoring adjusted_weight,
            // because the palette already optimizes for the max weight, so it'd likely find another redundant entry.
            for item in hist.items.iter() {
                // the early reject avoids running full palette search for every entry
                let may_be_worst = colors.get(item.likely_palette_index() as usize)
                    .map_or(true, |pal| pal.diff(&item.color) > worst_diff);
                if may_be_worst {
                    let diff = n.search(&item.color, item.likely_palette_index()).1;
                    if diff > worst_diff {
                        worst_diff = diff;
                        worst = Some(item);
                    }
                }
            }
            if let Some(worst) = worst {
                palette.set(pal_idx, worst.color, PalPop::new(worst.adjusted_weight));
            }
        }
    }
    Ok(())
}
