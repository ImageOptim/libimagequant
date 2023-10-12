use crate::pal::PalIndex;
use crate::pal::MAX_COLORS;
use crate::pal::{f_pixel, PalF};
use crate::{Error, OrdFloat};

impl<'pal> Nearest<'pal> {
    #[inline(never)]
    pub fn new(palette: &'pal PalF) -> Result<Self, Error> {
        if palette.len() > PalIndex::MAX as usize + 1 {
            return Err(Error::Unsupported);
        }
        let mut indexes: Vec<_> = (0..palette.len())
            .map(|idx| MapIndex { idx: idx as _ })
            .collect();
        if indexes.is_empty() {
            return Err(Error::Unsupported);
        }
        let mut handle = Nearest {
            root: vp_create_node(&mut indexes, palette),
            palette,
            nearest_other_color_dist: [0.; MAX_COLORS],
        };
        for (i, color) in palette.as_slice().iter().enumerate() {
            let mut best = Visitor {
                idx: 0, distance: f32::MAX, distance_squared: f32::MAX,
                exclude: Some(i as PalIndex),
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
                exclude: None,
            }
        } else {
            Visitor { distance: f32::INFINITY, distance_squared: f32::INFINITY, idx: 0, exclude: None, }
        };

        vp_search_node(&self.root, px, &mut best_candidate);
        (best_candidate.idx, best_candidate.distance_squared)
    }
}

pub(crate) struct Nearest<'pal> {
    root: Node,
    palette: &'pal PalF,
    nearest_other_color_dist: [f32; MAX_COLORS],
}

pub struct MapIndex {
    pub idx: PalIndex,
}

pub struct Visitor {
    pub distance: f32,
    pub distance_squared: f32,
    pub idx: PalIndex,
    pub exclude: Option<PalIndex>,
}

impl Visitor {
    #[inline]
    fn visit(&mut self, distance: f32, distance_squared: f32, idx: PalIndex) {
        if distance_squared < self.distance_squared && self.exclude != Some(idx) {
            self.distance = distance;
            self.distance_squared = distance_squared;
            self.idx = idx;
        }
    }
}

pub(crate) struct Node {
    vantage_point: f_pixel,
    inner: NodeInner,
    idx: PalIndex,
}

const LEAF_MAX_SIZE: usize = 6;

enum NodeInner {
    Nodes {
        radius: f32,
        radius_squared: f32,
        near: Box<Node>,
        far: Box<Node>,
    },
    Leaf {
        len: u8,
        idxs: [PalIndex; LEAF_MAX_SIZE],
        colors: Box<[f_pixel; LEAF_MAX_SIZE]>,
    },
}

#[inline(never)]
fn vp_create_node(indexes: &mut [MapIndex], items: &PalF) -> Node {
    debug_assert!(!indexes.is_empty());
    let palette = items.as_slice();

    if indexes.len() == 1 {
        return Node {
            vantage_point: palette[usize::from(indexes[0].idx)],
            idx: indexes[0].idx,
            inner: NodeInner::Leaf { len: 0, idxs: [0; LEAF_MAX_SIZE], colors: Box::new([f_pixel::default(); LEAF_MAX_SIZE]) },
        };
    }

    let most_popular_item = indexes.iter().enumerate().max_by_key(move |(_, idx)| {
        OrdFloat::new(items.pop_as_slice()[usize::from(idx.idx)].popularity())
    }).map(|(n, _)| n).unwrap_or_default();
    indexes.swap(most_popular_item, 0);
    let (ref_, indexes) = indexes.split_first_mut().unwrap();

    let vantage_point = palette[usize::from(ref_.idx)];
    indexes.sort_unstable_by_key(move |i| OrdFloat::new(vantage_point.diff(&palette[usize::from(i.idx)])));

    let num_indexes = indexes.len();

    let inner = if num_indexes <= LEAF_MAX_SIZE {
        let mut colors = [f_pixel::default(); LEAF_MAX_SIZE];
        let mut idxs = [Default::default(); LEAF_MAX_SIZE];

        indexes.iter().zip(colors.iter_mut().zip(idxs.iter_mut())).for_each(|(i, (color, idx))| {
            *idx = i.idx;
            *color = palette[usize::from(i.idx)];
        });
        NodeInner::Leaf {
            len: num_indexes as _,
            idxs,
            colors: Box::new(colors),
        }
    } else {
        let half_index = num_indexes / 2;
        let (near, far) = indexes.split_at_mut(half_index);
        debug_assert!(!near.is_empty());
        debug_assert!(!far.is_empty());
        let radius_squared = vantage_point.diff(&palette[usize::from(far[0].idx)]);
        let radius = radius_squared.sqrt();
        NodeInner::Nodes {
            radius, radius_squared,
            near: Box::new(vp_create_node(near, items)),
            far: Box::new(vp_create_node(far, items)),
        }
    };

    Node {
        inner,
        vantage_point: palette[usize::from(ref_.idx)],
        idx: ref_.idx,
    }
}

#[inline(never)]
fn vp_search_node(mut node: &Node, needle: &f_pixel, best_candidate: &mut Visitor) {
    loop {
        let distance_squared = node.vantage_point.diff(needle);
        let distance = distance_squared.sqrt();

        best_candidate.visit(distance, distance_squared, node.idx);

        match node.inner {
            NodeInner::Nodes { radius, radius_squared, ref near, ref far } => {
                // Recurse towards most likely candidate first to narrow best candidate's distance as soon as possible
                if distance_squared < radius_squared {
                    vp_search_node(near, needle, best_candidate);
                    // The best node (final answer) may be just ouside the radius, but not farther than
                    // the best distance we know so far. The vp_search_node above should have narrowed
                    // best_candidate->distance, so this path is rarely taken.
                    if distance >= radius - best_candidate.distance {
                        node = far;
                        continue;
                    }
                } else {
                    vp_search_node(far, needle, best_candidate);
                    if distance <= radius + best_candidate.distance {
                        node = near;
                        continue;
                    }
                }
                break;
            },
            NodeInner::Leaf { len: num, ref idxs, ref colors } => {
                colors.iter().zip(idxs.iter().copied()).take(num as usize).for_each(|(color, idx)| {
                    let distance_squared = color.diff(needle);
                    best_candidate.visit(distance_squared.sqrt(), distance_squared, idx);
                });
                break;
            },
        }
    }
}
