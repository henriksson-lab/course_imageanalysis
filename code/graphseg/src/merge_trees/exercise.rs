// Coding exercises for "Hierarchical segmentation and merge trees".
// Replace each `todo!()` with your code, then check it with `cargo test -p graphseg merge_trees::`
// and look at `cargo run -p graphseg -- merge-trees`.
#![allow(unused_variables, unused_imports)]

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use crate::common::{find, sorted};

/// An edge `(a, b, d)` between leaves (initial regions) `a` and `b`: the pass between them.
pub use crate::common::Edge;

/// One merge of a merge tree: the tree nodes `a` and `b` join at `level`. Leaves are the nodes
/// `0 .. n`; merge number `k` (in the list) creates the node `n + k`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Merge {
    pub a: usize,
    pub b: usize,
    pub level: f32,
}

/// Which attribute orders an attribute hierarchy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Attribute {
    /// depth: pass height minus the minimum of the region
    Dynamics,
    /// number of pixels (or the given leaf areas, summed)
    Area,
    /// area × depth
    Volume,
}

/// C1. The altitude hierarchy: Kruskal on the passes. Every edge that joins two different
/// components becomes a merge of their two tree nodes at the level of the pass.
///
/// Hint: union–find with `find` (given, in `src/common.rs`); also keep, for every root, the tree node that
/// stands for its component (a leaf at first, then `n + merges.len() − 1` after a merge).
pub fn altitude_tree(n: usize, edges: &[Edge]) -> Vec<Merge> {
    todo!("C1: altitude_tree")
}

/// C2. An attribute hierarchy. Flood by altitude (Kruskal on the passes, as in C1); each time
/// two components meet at pass `h`, the merge gets the attribute of the one with the smaller
/// attribute: dynamics `h − min`, area `|C|`, or volume `|C| · (h − min)`. Then build the tree
/// again from the same joining edges, sorted by these values (equal values in flooding order).
/// `minimum` and `area` are given per leaf.
///
/// Hint: the same loop as C1, but keep the minimum and the area of every component at its
/// root; collect the joining edges with their new values, then call `altitude_tree` on them.
pub fn attribute_tree(n: usize, edges: &[Edge], minimum: &[f32], area: &[f32], attribute: Attribute) -> Vec<Merge> {
    todo!("C2: attribute_tree")
}

/// C3. Apply the first `k` merges of a merge tree. Returns a region label `0, 1, 2, …` for
/// every leaf (numbered in the order of their first leaf).
///
/// Hint: every tree node needs one of its leaves: leaves are their own, and merge `k`'s node
/// can use a leaf of its child `a`. Join those leaves with union–find.
pub fn cut_first(n: usize, merges: &[Merge], k: usize) -> Vec<usize> {
    todo!("C3: cut_first")
}

/// Cut a merge tree at `level`: apply every merge with level ≤ `level` (for trees whose
/// levels never decrease, such as C1 and C2).
pub fn cut(n: usize, merges: &[Merge], level: f32) -> Vec<usize> {
    cut_first(n, merges, merges.iter().filter(|m| m.level <= level).count())
}

/// Cut a merge tree so that `regions` regions remain: apply the first `n − regions` merges.
pub fn cut_regions(n: usize, merges: &[Merge], regions: usize) -> Vec<usize> {
    cut_first(n, merges, n.saturating_sub(regions))
}

/// C4. The ultrametric contour map: for every edge `(a, b)`, the level at which leaves `a` and
/// `b` first belong to one region. Replay the merges with union–find (no shortcuts, the
/// smaller tree under the larger) and note the level of each link; then `u(a, b)` is the
/// largest link level on the way from `a` and from `b` up to their first common ancestor.
///
/// Hint: here `find` must not take shortcuts (no path halving): walk up `parent` plainly.
/// Collect the ancestors of `a` with the largest link level so far, then walk up from `b`.
pub fn ucm(n: usize, merges: &[Merge], edges: &[Edge]) -> Vec<f32> {
    todo!("C4: ucm")
}

/// An entry of the merge queue of C5 (given): a pair of regions `a < b`, their difference, and
/// the version numbers of `a` and `b` when it was queued. In a `BinaryHeap` the smallest
/// difference pops first.
pub struct Candidate {
    pub diff: f32,
    pub a: usize,
    pub b: usize,
    pub stamp: (usize, usize),
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.diff.total_cmp(&self.diff).then(other.a.cmp(&self.a)).then(other.b.cmp(&self.b))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}

/// C5. Greedy merging on a region adjacency graph: always merge the adjacent pair with the
/// smallest difference of means; the new region has the area-weighted mean. `pairs` lists the
/// adjacent leaves. A priority queue with lazy updates: every region has a version number,
/// raised when it changes, and a popped pair whose versions are old is skipped. The merge
/// levels are the differences (so they can go down: this is not single linkage).
///
/// Hint: push `Candidate`s (given) into a `std::collections::BinaryHeap`; give every region a
/// version number, raised when it changes, and compare it with the entry's `stamp`.
pub fn rag_merge(means: &[f32], areas: &[f32], pairs: &[(usize, usize)]) -> Vec<Merge> {
    todo!("C5: rag_merge")
}
