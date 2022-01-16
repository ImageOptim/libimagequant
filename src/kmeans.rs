use crate::Error;
use crate::hist::{HistItem, HistogramInternal};
use crate::nearest::Nearest;
use crate::pal::{PalF, PalIndex, PalPop, f_pixel};
use fallible_collections::FallibleVec;
use rgb::alt::ARGB;
use rgb::ComponentMap;
use std::cell::RefCell;
use crate::rayoff::*;

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
        let mut averages: Vec<_> = FallibleVec::try_with_capacity(pal_len)?;
        averages.resize(pal_len, ColorAvg::default());
        Ok(Self {
            averages,
            weighed_diff_sum: 0.,
        })
    }

    #[inline]
    pub fn update_color(&mut self, px: f_pixel, value: f32, matched: PalIndex) {
        let c = &mut self.averages[matched as usize];
        c.sum += (px.0 * value).map(|c| c as f64);
        c.total += value as f64;
    }

    pub fn finalize(self, palette: &mut PalF) -> f64 {
        for (avg, (color, pop)) in self.averages.iter().zip(palette.iter_mut()).filter(|(_, (_, pop))| !pop.is_fixed()) {
            let total = avg.total;
            *pop = PalPop::new(total as f32);
            if total > 0. {
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
        hist.items.par_chunks_mut(256).for_each(|batch| {
            let kmeans = tls.get_or(move || RefCell::new(Kmeans::new(len)));
            if let Ok(ref mut kmeans) = *kmeans.borrow_mut() {
                kmeans.iterate_batch(batch, &n, colors, adjust_weight);
            }
        });

        let diff = tls.into_iter()
            .map(RefCell::into_inner)
            .reduce(Kmeans::try_merge)
            .transpose()?
            .map(|kmeans| {
                kmeans.finalize(palette) / total
            }).unwrap_or(0.);

        // kmeans may have obsoleted some palette entries. Replace them with any entry from the histogram
        // (it happens so rarely that there's no point doing something smarter)
        palette.iter_mut().filter(|(_, p)| !p.is_fixed() && p.popularity() == 0.).zip(hist.items.iter()).for_each(|((c, _), item)| {
            *c = item.color;
        });
        Ok(diff)
    }

    fn iterate_batch(&mut self, batch: &mut [HistItem], n: &Nearest, colors: &[f_pixel], adjust_weight: bool) {
        self.weighed_diff_sum += batch.iter_mut().map(|item| {
            let px = item.color;
            let (matched, mut diff) = n.search(&px, unsafe { item.tmp.likely_palette_index });
            item.tmp.likely_palette_index = matched;
            if adjust_weight {
                let remapped = colors[matched as usize];
                let (_, new_diff) = n.search(&f_pixel(px.0 + px.0 - remapped.0), matched);
                diff = new_diff;
                item.adjusted_weight = (item.perceptual_weight + 2. * item.adjusted_weight) * (0.5 + diff);
            }
            debug_assert!((diff as f64) < 1e20);
            self.update_color(px, item.adjusted_weight, matched);
            (diff * item.perceptual_weight) as f64
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
