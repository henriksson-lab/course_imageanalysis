// Coding exercises for "Watershed segmentation of a real image". Replace each `todo!()` with your
// code, then check it with `cargo test -p ws-gradient` and look at `cargo run -p ws-gradient`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};

// ---------------------------------------------------------------------------------------------
// Given (from earlier units, not part of this exercise): smoothing, regional minima, flooding.
// ---------------------------------------------------------------------------------------------

/// The 8 neighbours `(dx, dy)` in reading order.
pub const N8: [(isize, isize); 8] = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

/// Given: Gaussian smoothing with standard deviation `sigma` (in pixels), in two passes, with the
/// edge pixels repeated outside the image. `sigma = 0` returns a copy.
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    if sigma <= 0.0 {
        return img.clone();
    }
    let r = (3.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut s = 0.0;
            for i in -r..=r {
                let (xx, yy) = if along_x { (x as isize + i, y as isize) } else { (x as isize, y as isize + i) };
                s += w[(i + r) as usize] * src.get_border(xx, yy, Border::Clamp);
            }
            s / total
        })
    };
    pass(&pass(img, true), false)
}

/// Given (the flooding unit's C1): the regional minima with 8 neighbours, labelled 1, 2, 3, ...
/// in raster order. Returns the labels and their number.
pub fn regional_minima(f: &Image<f32>) -> (Image<u32>, u32) {
    let mut seen = Image::new(f.width(), f.height(), false);
    let mut labels = Image::new(f.width(), f.height(), 0u32);
    let mut count = 0;
    for start in f.coords() {
        if seen[start] {
            continue;
        }
        let v = f[start];
        let (mut plateau, mut queue, mut is_minimum) = (vec![start], VecDeque::from([start]), true);
        seen[start] = true;
        while let Some((x, y)) = queue.pop_front() {
            for &(dx, dy) in &N8 {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if f.contains(nx, ny) {
                    let q = (nx as usize, ny as usize);
                    if f[q] < v {
                        is_minimum = false;
                    } else if f[q] == v && !seen[q] {
                        seen[q] = true;
                        plateau.push(q);
                        queue.push_back(q);
                    }
                }
            }
        }
        if is_minimum {
            count += 1;
            for p in plateau {
                labels[p] = count;
            }
        }
    }
    (labels, count)
}

#[derive(PartialEq)]
struct Entry(f32, u64, (usize, usize));
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

/// Given (the flooding unit's C2): Meyer's flooding of `f` from the `markers` with 8 neighbours,
/// every pixel labelled (each takes the label of the pixel that queued it; no watershed line).
pub fn watershed(f: &Image<f32>, markers: &Image<u32>) -> Image<u32> {
    let mut labels = markers.clone();
    let mut queued = markers.map(|m| m > 0);
    let mut heap = BinaryHeap::new();
    let mut arrivals = 0u64;
    let mut stack: Vec<(usize, usize)> = f.coords().filter(|&p| markers[p] > 0).collect();
    loop {
        // label the pixels in `stack` are done; queue their neighbours
        for p in stack.drain(..) {
            for &(dx, dy) in &N8 {
                let (nx, ny) = (p.0 as isize + dx, p.1 as isize + dy);
                if f.contains(nx, ny) {
                    let q = (nx as usize, ny as usize);
                    if !queued[q] {
                        queued[q] = true;
                        labels[q] = labels[p];
                        heap.push(Reverse(Entry(f[q], arrivals, q)));
                        arrivals += 1;
                    }
                }
            }
        }
        match heap.pop() {
            Some(Reverse(Entry(_, _, p))) => stack.push(p),
            None => return labels,
        }
    }
}

/// Given: the boundary pixels of a label image (a pixel with a 4-neighbour of another label).
pub fn boundaries(labels: &Image<u32>) -> Image<bool> {
    labels.map_xy(|x, y, l| {
        [(1isize, 0isize), (0, 1), (-1, 0), (0, -1)].iter().any(|&(dx, dy)| {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            labels.contains(nx, ny) && labels[(nx as usize, ny as usize)] != l
        })
    })
}

// ---------------------------------------------------------------------------------------------
// The exercises
// ---------------------------------------------------------------------------------------------

/// C1. The gradient landscape as in the exercises: g = |gx| + |gy| with the Sobel kernels
/// (gx = right column − left column, gy = bottom row − top row, weights 1, 2, 1), reading the
/// edge pixel outside the image.
///
/// Hint: `img.get_border(x as isize + dx, y as isize + dy, Border::Clamp)` reads a neighbour.
pub fn gradient_l1(img: &Image<f32>) -> Image<f32> {
    todo!("C1: gradient_l1")
}

/// C2. The h-minima transform with 8 neighbours: start with g + h and erode it, never going
/// below g: m(p) = max(g(p), min of m over p and its 8 neighbours), until nothing changes.
///
/// Hint: repeating "compute all new values from the old ones" until nothing changes works, but
/// needs many rounds on a large image. Faster: update `m` in place, sweeping forwards (row by row)
/// and then backwards, until a pair of sweeps changes nothing.
pub fn h_minima(g: &Image<f32>, h: f32) -> Image<f32> {
    todo!("C2: h_minima")
}

/// C3. The dams between neighbouring regions: for every pair of labels a < b that touch
/// (4-neighbours), the lowest dam height, where a dam between pixels p and q is max(g(p), g(q)).
/// Returns (a, b, height), sorted by a, then b.
///
/// Hint: compare every pixel with its right and its lower neighbour. A
/// `std::collections::BTreeMap` from `(a, b)` to the lowest height so far keeps the pairs sorted.
pub fn dam_heights(labels: &Image<u32>, g: &Image<f32>) -> Vec<(u32, u32, f32)> {
    todo!("C3: dam_heights")
}

/// C4. Merge the regions across every dam of height ≤ t (union–find on the labels), and number
/// the merged regions 1, 2, 3, ... in order of their smallest old label. Returns the new label
/// image and the number of regions.
///
/// Hint: a `parent` vector over the labels 0..=max, as in the labelling unit; hang the root
/// with the larger label under the other, so that each root is the smallest label of its region.
pub fn merge_below(labels: &Image<u32>, dams: &[(u32, u32, f32)], t: f32) -> (Image<u32>, u32) {
    todo!("C4: merge_below")
}
