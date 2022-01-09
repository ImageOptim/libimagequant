use fallible_collections::FallibleVec;

use crate::{OrdFloat, Error};
use crate::pal::PalIndex;
use crate::pal::{f_pixel, PalF};

impl<'pal> Nearest<'pal> {
    #[inline(never)]
    pub fn new(palette: &'pal PalF) -> Result<Self, Error> {
        let mut indexes: Vec<_> = (0..palette.len())
            .map(|idx| MapIndex { idx: idx as _ })
            .collect();
        let mut handle = Nearest {
            root: vp_create_node(&mut indexes, palette)?,
            palette,
            nearest_other_color_dist: [0.; 256],
        };
        for (i, color) in palette.as_slice().iter().enumerate() {
            let mut best = Visitor {
                idx: 0, distance: f32::MAX, distance_squared: f32::MAX,
                exclude: i as i16,
            };
            vp_search_node(&handle.root, color, &mut best);
            handle.nearest_other_color_dist[i] = best.distance_squared / 4.;
        }
        Ok(handle)
    }
}

impl Nearest<'_> {
    #[inline]
    pub fn search(&self, px: &f_pixel, likely_colormap_index: PalIndex) -> (PalIndex, f32) {
        // The index may be invalid, so it needs to be checked
        let mut best_candidate = if let Some(pal_px) = self.palette.as_slice().get(likely_colormap_index as usize) {
            let guess_diff = px.diff(pal_px);
            if guess_diff < self.nearest_other_color_dist[likely_colormap_index as usize] {
                return (likely_colormap_index, guess_diff);
            }
            Visitor {
                distance: guess_diff.sqrt(),
                distance_squared: guess_diff,
                idx: likely_colormap_index,
                exclude: -1,
            }
        } else {
            Visitor { distance: f32::INFINITY, distance_squared: f32::INFINITY, idx: 0, exclude: -1, }
        };

        vp_search_node(&self.root, px, &mut best_candidate);
        (best_candidate.idx as PalIndex, best_candidate.distance * best_candidate.distance)
    }
}

pub(crate) struct Nearest<'pal> {
    root: Node,
    palette: &'pal PalF,
    nearest_other_color_dist: [f32; 256],
}

pub struct MapIndex {
    pub idx: PalIndex,
}

pub struct Visitor {
    pub distance: f32,
    pub distance_squared: f32,
    pub idx: PalIndex,
    pub exclude: i16,
}

impl Visitor {
    #[inline]
    fn visit(&mut self, distance: f32, distance_squared: f32, idx: PalIndex) {
        if distance_squared < self.distance_squared && self.exclude != idx as i16 {
            self.distance = distance;
            self.distance_squared = distance_squared;
            self.idx = idx;
        }
    }
}

pub struct Leaf {
    pub color: f_pixel,
    pub idx: PalIndex,
}

pub struct Node {
    pub near: Option<Box<Node>>,
    pub far: Option<Box<Node>>,
    pub vantage_point: f_pixel,
    pub radius: f32,
    pub radius_squared: f32,
    pub idx: PalIndex,
    pub rest: Box<[Leaf]>,
}

fn vp_create_node(indexes: &mut [MapIndex], items: &PalF) -> Result<Node, Error> {
    debug_assert!(!indexes.is_empty());
    let palette = items.as_slice();

    if indexes.len() <= 1 {
        return Ok(Node {
            vantage_point: palette[usize::from(indexes[0].idx)],
            radius: f32::NAN,
            radius_squared: f32::NAN,
            idx: indexes[0].idx,
            near: None,
            far: None,
            rest: [].into(),
        });
    }

    let most_popular_item = indexes.iter().enumerate().max_by_key(move |(_, i)| {
        OrdFloat::<f32>::unchecked_new(items.pop_as_slice()[usize::from(i.idx)].popularity())
    }).unwrap().0;
    indexes.swap(0, most_popular_item);
    let (ref_, indexes) = indexes.split_first_mut().unwrap();

    let vantage_point = palette[usize::from(ref_.idx)];
    indexes.sort_unstable_by_key(move |i| OrdFloat::<f32>::unchecked_new(vantage_point.diff(&palette[i.idx as usize])));

    let half_index = indexes.len() / 2;
    let num_indexes = indexes.len();
    let (near, far) = indexes.split_at_mut(half_index);

    let radius_squared = vantage_point.diff(&palette[far[0].idx as usize]);
    let radius = radius_squared.sqrt();

    let (near, far, rest) = if num_indexes < 7 {
        let mut rest: Vec<_> = FallibleVec::try_with_capacity(num_indexes)?;
        rest.extend(near.iter().chain(far.iter()).map(|i| Leaf {
            idx: i.idx,
            color: palette[usize::from(i.idx)],
        }));
        (None, None, rest.into_boxed_slice())
    } else {
        (
            if !near.is_empty() { Some(Box::new(vp_create_node(near, items)?)) } else { None },
            if !far.is_empty() { Some(Box::new(vp_create_node(far, items)?)) } else { None },
            [].into(),
        )
    };

    Ok(Node {
        vantage_point: palette[usize::from(ref_.idx)],
        radius,
        radius_squared,
        idx: ref_.idx,
        near, far, rest,
    })
}

fn vp_search_node(mut node: &Node, needle: &f_pixel, best_candidate: &mut Visitor) {
    loop {
        let distance_squared = node.vantage_point.diff(needle);
        let distance = distance_squared.sqrt();

        best_candidate.visit(distance, distance_squared, node.idx);

        if !node.rest.is_empty() {
            for r in node.rest.iter() {
                let distance_squared = r.color.diff(needle);
                best_candidate.visit(distance_squared.sqrt(), distance_squared, r.idx);
            }
            break;
        }

        // Recurse towards most likely candidate first to narrow best candidate's distance as soon as possible
        if distance_squared < node.radius_squared {
            if let Some(near) = &node.near {
                vp_search_node(near, needle, best_candidate);
            }
            // The best node (final answer) may be just ouside the radius, but not farther than
            // the best distance we know so far. The vp_search_node above should have narrowed
            // best_candidate->distance, so this path is rarely taken.
            if distance >= node.radius - best_candidate.distance {
                if let Some(far) = &node.far {
                    node = far;
                    continue;
                }
            }
        } else {
            if let Some(far) = &node.far {
                vp_search_node(far, needle, best_candidate);
            }
            if distance <= node.radius + best_candidate.distance {
                if let Some(near) = &node.near {
                    node = near;
                    continue;
                }
            }
        }
        break;
    }
}
