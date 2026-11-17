// Coding exercises for "Connectivity and labelling". Replace each `todo!()` with your code,
// then check it with `cargo test -p labelling label_connectivity::` and look at `cargo run -p labelling -- label-connectivity`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;
use std::collections::VecDeque;

/// The 4 neighbours `(dx, dy)` of a pixel: up, left, right, down.
pub const N4: [(isize, isize); 4] = [(0, -1), (-1, 0), (1, 0), (0, 1)];
/// The 8 neighbours `(dx, dy)` of a pixel, in reading order.
pub const N8: [(isize, isize); 8] = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

/// The neighbour offsets for 8-connectivity (`eight = true`) or 4-connectivity.
pub fn neighbours(eight: bool) -> &'static [(isize, isize)] {
    if eight {
        &N8
    } else {
        &N4
    }
}

/// C1. Flood fill with a queue: give `label` to the seed and to every object pixel connected
/// to it that has no label yet (label 0). Returns the pixels in the order they were taken
/// out of the queue.
///
/// Hint: a `VecDeque` is a queue (`push_back`, `pop_front`). Give a pixel its label when it
/// enters the queue, so that it enters only once. `mask.contains(nx, ny)` tells whether a
/// neighbour position (signed numbers) is inside the image.
pub fn flood_fill(mask: &Image<bool>, labels: &mut Image<u32>, seed: (usize, usize), label: u32, eight: bool) -> Vec<(usize, usize)> {
    todo!("C1: flood_fill")
}

/// C2. Label all components: scan the image row by row; at every object pixel without a
/// label, start a flood fill with the next label 1, 2, 3, ... Returns the label image and
/// the number of components.
pub fn label_flood_fill(mask: &Image<bool>, eight: bool) -> (Image<u32>, u32) {
    todo!("C2: label_flood_fill")
}

/// C3. Union–find (a disjoint-set forest) on the elements `0..n`. Every element points to a
/// parent; a root points to itself. `rank` is only used by `union_by_rank`.
#[derive(Clone, Debug)]
pub struct UnionFind {
    pub parent: Vec<usize>,
    pub rank: Vec<usize>,
}

impl UnionFind {
    /// `n` elements, each its own set.
    pub fn new(n: usize) -> Self {
        UnionFind { parent: (0..n).collect(), rank: vec![0; n] }
    }

    /// C3a. The root of the set of `x`: follow the parents until an element is its own parent.
    pub fn find(&self, x: usize) -> usize {
        todo!("C3a: find")
    }

    /// C3b. Join the sets of `a` and `b` without heuristics: the root of `a` goes under the
    /// root of `b`.
    pub fn union(&mut self, a: usize, b: usize) {
        todo!("C3b: union")
    }

    /// C3c. Union by rank: the root of lower rank goes under the other; on a tie the root of
    /// `a` goes under the root of `b`, and the rank of `b`'s root grows by one.
    pub fn union_by_rank(&mut self, a: usize, b: usize) {
        todo!("C3c: union_by_rank")
    }
}

/// C4. The two-pass algorithm. First pass, in raster order: at each object pixel look at the
/// neighbours already visited (8: up-left, up, up-right, left; 4: up, left); none labelled:
/// a new provisional label; otherwise copy the smallest, and record with `union` that all
/// labels seen are equivalent. Second pass: replace every provisional label by its class
/// number, the classes numbered 1, 2, 3, ... in order of their smallest provisional label.
/// Returns (provisional labels, final labels, number of components).
///
/// Hint: `UnionFind::new(width * height + 1)` has room for every possible provisional label.
/// For the numbering, go through the provisional labels 1, 2, 3, ... and give each new root
/// the next class number.
pub fn label_two_pass(mask: &Image<bool>, eight: bool) -> (Image<u32>, Image<u32>, u32) {
    todo!("C4: label_two_pass")
}
