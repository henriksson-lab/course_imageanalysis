// Model solution for the coding exercises of "Hierarchical segmentation and merge trees".

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
pub fn altitude_tree(n: usize, edges: &[Edge]) -> Vec<Merge> {
    let mut parent: Vec<usize> = (0..n).collect();
    let mut node: Vec<usize> = (0..n).collect(); // tree node of the component, at its root
    let mut merges = Vec::new();
    for (a, b, d) in sorted(edges) {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra == rb {
            continue;
        }
        merges.push(Merge { a: node[ra], b: node[rb], level: d });
        parent[rb] = ra;
        node[ra] = n + merges.len() - 1;
    }
    merges
}

/// C2. An attribute hierarchy. Flood by altitude (Kruskal on the passes, as in C1); each time
/// two components meet at pass `h`, the merge gets the attribute of the one with the smaller
/// attribute: dynamics `h − min`, area `|C|`, or volume `|C| · (h − min)`. Then build the tree
/// again from the same joining edges, sorted by these values (equal values in flooding order).
/// `minimum` and `area` are given per leaf.
pub fn attribute_tree(n: usize, edges: &[Edge], minimum: &[f32], area: &[f32], attribute: Attribute) -> Vec<Merge> {
    let mut parent: Vec<usize> = (0..n).collect();
    let mut min = minimum.to_vec(); // per component, at its root
    let mut size = area.to_vec();
    let mut joins: Vec<Edge> = Vec::new(); // the joining edges, re-weighted
    for (a, b, h) in sorted(edges) {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra == rb {
            continue;
        }
        let value = |r: usize| match attribute {
            Attribute::Dynamics => h - min[r],
            Attribute::Area => size[r],
            Attribute::Volume => size[r] * (h - min[r]),
        };
        joins.push((a, b, value(ra).min(value(rb))));
        parent[rb] = ra;
        min[ra] = min[ra].min(min[rb]);
        size[ra] += size[rb];
    }
    altitude_tree(n, &joins)
}

/// C3. Apply the first `k` merges of a merge tree. Returns a region label `0, 1, 2, …` for
/// every leaf (numbered in the order of their first leaf).
pub fn cut_first(n: usize, merges: &[Merge], k: usize) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..n).collect();
    let mut leaf_of: Vec<usize> = (0..n).collect(); // one leaf of every tree node
    for m in &merges[..k.min(merges.len())] {
        let (la, lb) = (leaf_of[m.a], leaf_of[m.b]);
        leaf_of.push(la);
        let (ra, rb) = (find(&mut parent, la), find(&mut parent, lb));
        parent[rb] = ra;
    }
    let mut label_of_root = HashMap::new();
    (0..n)
        .map(|i| {
            let r = find(&mut parent, i);
            let next = label_of_root.len();
            *label_of_root.entry(r).or_insert(next)
        })
        .collect()
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
pub fn ucm(n: usize, merges: &[Merge], edges: &[Edge]) -> Vec<f32> {
    let mut parent: Vec<usize> = (0..n).collect();
    let mut size = vec![1usize; n];
    let mut link_level = vec![0.0f32; n];
    let mut leaf_of: Vec<usize> = (0..n).collect();
    let root = |parent: &[usize], mut a: usize| {
        while parent[a] != a {
            a = parent[a];
        }
        a
    };
    for m in merges {
        let (la, lb) = (leaf_of[m.a], leaf_of[m.b]);
        leaf_of.push(la);
        let (mut ra, mut rb) = (root(&parent, la), root(&parent, lb));
        if size[ra] < size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        parent[rb] = ra;
        size[ra] += size[rb];
        link_level[rb] = m.level;
    }
    edges
        .iter()
        .map(|&(a, b, _)| {
            // the ancestors of a, with the largest link level on the way to each
            let mut up_a = Vec::new();
            let (mut p, mut high) = (a, f32::NEG_INFINITY);
            loop {
                up_a.push((p, high));
                if parent[p] == p {
                    break;
                }
                high = high.max(link_level[p]);
                p = parent[p];
            }
            let (mut q, mut high_b) = (b, f32::NEG_INFINITY);
            loop {
                if let Some(&(_, high_a)) = up_a.iter().find(|(node, _)| *node == q) {
                    return high_a.max(high_b);
                }
                if parent[q] == q {
                    return f32::INFINITY; // never joined
                }
                high_b = high_b.max(link_level[q]);
                q = parent[q];
            }
        })
        .collect()
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
pub fn rag_merge(means: &[f32], areas: &[f32], pairs: &[(usize, usize)]) -> Vec<Merge> {
    let n = means.len();
    let mut mean = means.to_vec();
    let mut area = areas.to_vec();
    let mut node: Vec<usize> = (0..n).collect(); // tree node of each region (indexed by its id)
    let mut version = vec![0usize; n];
    let mut alive = vec![true; n];
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(a, b) in pairs {
        if a != b && !neighbours[a].contains(&b) {
            neighbours[a].push(b);
            neighbours[b].push(a);
        }
    }
    let mut queue = BinaryHeap::new();
    for a in 0..n {
        for &b in &neighbours[a] {
            if a < b {
                queue.push(Candidate { diff: (mean[a] - mean[b]).abs(), a, b, stamp: (0, 0) });
            }
        }
    }
    let mut merges = Vec::new();
    while let Some(c) = queue.pop() {
        if !alive[c.a] || !alive[c.b] || c.stamp != (version[c.a], version[c.b]) {
            continue; // out of date
        }
        // merge b into a
        let (a, b) = (c.a, c.b);
        merges.push(Merge { a: node[a], b: node[b], level: c.diff });
        node[a] = n + merges.len() - 1;
        mean[a] = (mean[a] * area[a] + mean[b] * area[b]) / (area[a] + area[b]);
        area[a] += area[b];
        alive[b] = false;
        version[a] += 1;
        let nb = std::mem::take(&mut neighbours[b]);
        for q in nb {
            neighbours[q].retain(|&x| x != b);
            if q != a && !neighbours[a].contains(&q) {
                neighbours[a].push(q);
                neighbours[q].push(a);
            }
        }
        for &q in &neighbours[a] {
            let (x, y) = if a < q { (a, q) } else { (q, a) };
            queue.push(Candidate { diff: (mean[a] - mean[q]).abs(), a: x, b: y, stamp: (version[x], version[y]) });
        }
    }
    merges
}
