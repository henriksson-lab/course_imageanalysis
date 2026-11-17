// Model solution for the coding exercises of "Connectivity and labelling".

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
pub fn flood_fill(mask: &Image<bool>, labels: &mut Image<u32>, seed: (usize, usize), label: u32, eight: bool) -> Vec<(usize, usize)> {
    let mut order = Vec::new();
    let mut queue = VecDeque::new();
    // Label a pixel when it enters the queue, so that it enters only once.
    labels[seed] = label;
    queue.push_back(seed);
    while let Some((x, y)) = queue.pop_front() {
        order.push((x, y));
        for &(dx, dy) in neighbours(eight) {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if !mask.contains(nx, ny) {
                continue;
            }
            let q = (nx as usize, ny as usize);
            if mask[q] && labels[q] == 0 {
                labels[q] = label;
                queue.push_back(q);
            }
        }
    }
    order
}

/// C2. Label all components: scan the image row by row; at every object pixel without a
/// label, start a flood fill with the next label 1, 2, 3, ... Returns the label image and
/// the number of components.
pub fn label_flood_fill(mask: &Image<bool>, eight: bool) -> (Image<u32>, u32) {
    let mut labels = Image::new(mask.width(), mask.height(), 0u32);
    let mut count = 0;
    for (x, y) in mask.coords() {
        if mask[(x, y)] && labels[(x, y)] == 0 {
            count += 1;
            flood_fill(mask, &mut labels, (x, y), count, eight);
        }
    }
    (labels, count)
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
        let mut r = x;
        while self.parent[r] != r {
            r = self.parent[r];
        }
        r
    }

    /// C3b. Join the sets of `a` and `b` without heuristics: the root of `a` goes under the
    /// root of `b`.
    pub fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }

    /// C3c. Union by rank: the root of lower rank goes under the other; on a tie the root of
    /// `a` goes under the root of `b`, and the rank of `b`'s root grows by one.
    pub fn union_by_rank(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[ra] = rb;
            if self.rank[ra] == self.rank[rb] {
                self.rank[rb] += 1;
            }
        }
    }
}

/// C4. The two-pass algorithm. First pass, in raster order: at each object pixel look at the
/// neighbours already visited (8: up-left, up, up-right, left; 4: up, left); none labelled:
/// a new provisional label; otherwise copy the smallest, and record with `union` that all
/// labels seen are equivalent. Second pass: replace every provisional label by its class
/// number, the classes numbered 1, 2, 3, ... in order of their smallest provisional label.
/// Returns (provisional labels, final labels, number of components).
pub fn label_two_pass(mask: &Image<bool>, eight: bool) -> (Image<u32>, Image<u32>, u32) {
    let visited: &[(isize, isize)] = if eight { &[(-1, -1), (0, -1), (1, -1), (-1, 0)] } else { &[(0, -1), (-1, 0)] };
    let mut prov = Image::new(mask.width(), mask.height(), 0u32);
    // at most one new label per pixel, plus the unused 0
    let mut uf = UnionFind::new(mask.width() * mask.height() + 1);
    let mut next = 0u32;

    // First pass
    for (x, y) in mask.coords() {
        if !mask[(x, y)] {
            continue;
        }
        let mut seen = Vec::new();
        for &(dx, dy) in visited {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if mask.contains(nx, ny) {
                let l = prov[(nx as usize, ny as usize)];
                if l > 0 {
                    seen.push(l);
                }
            }
        }
        match seen.iter().min() {
            None => {
                next += 1;
                prov[(x, y)] = next;
            }
            Some(&smallest) => {
                prov[(x, y)] = smallest;
                for &l in &seen {
                    uf.union(l as usize, smallest as usize);
                }
            }
        }
    }

    // Number the classes in order of their smallest provisional label.
    let mut class_of_root = vec![0u32; next as usize + 1];
    let mut final_label = vec![0u32; next as usize + 1];
    let mut count = 0;
    for l in 1..=next as usize {
        let r = uf.find(l);
        if class_of_root[r] == 0 {
            count += 1;
            class_of_root[r] = count;
        }
        final_label[l] = class_of_root[r];
    }

    // Second pass
    let labels = prov.map(|l| final_label[l as usize]);
    (prov, labels, count)
}
