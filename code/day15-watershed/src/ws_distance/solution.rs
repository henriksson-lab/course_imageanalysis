// Model solution for the coding exercises of "Splitting touching objects with the distance transform".

use imgcourse::Image;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};

// ---------------------------------------------------------------------------------------------
// Given (from earlier units, not part of this exercise)
// ---------------------------------------------------------------------------------------------

/// The 8 neighbours `(dx, dy)` in reading order.
pub const N8: [(isize, isize); 8] = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

/// Given: label the 8-connected components of a mask 1, 2, 3, ... Returns the labels and their number.
pub fn label_components(mask: &Image<bool>) -> (Image<u32>, u32) {
    let mut labels = Image::new(mask.width(), mask.height(), 0u32);
    let mut n = 0;
    for start in mask.coords() {
        if !mask[start] || labels[start] != 0 {
            continue;
        }
        n += 1;
        labels[start] = n;
        let mut queue = VecDeque::from([start]);
        while let Some((x, y)) = queue.pop_front() {
            for &(dx, dy) in &N8 {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if mask.contains(nx, ny) && mask[(nx as usize, ny as usize)] && labels[(nx as usize, ny as usize)] == 0 {
                    labels[(nx as usize, ny as usize)] = n;
                    queue.push_back((nx as usize, ny as usize));
                }
            }
        }
    }
    (labels, n)
}

/// Given (the shape-features unit): the exact Euclidean distance transform. For every object
/// pixel, the distance to the nearest background pixel centre; outside the image is background.
/// Only background pixels next to the object can be the nearest, so only those are tried.
pub fn distance_euclidean(mask: &Image<bool>) -> Image<f32> {
    let (w, h) = (mask.width(), mask.height());
    let candidates: Vec<(usize, usize)> = mask
        .coords()
        .filter(|&(x, y)| {
            !mask[(x, y)]
                && N8.iter().any(|&(dx, dy)| {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    mask.contains(nx, ny) && mask[(nx as usize, ny as usize)]
                })
        })
        .collect();
    mask.map_xy(|x, y, inside| {
        if !inside {
            return 0.0;
        }
        let mut best = (x + 1).min(y + 1).min(w - x).min(h - y) as f32;
        for &(bx, by) in &candidates {
            best = best.min(((x as f32 - bx as f32).powi(2) + (y as f32 - by as f32).powi(2)).sqrt());
        }
        best
    })
}

/// Given (the flooding unit): the regional maxima with 8 neighbours among the pixels of `mask`:
/// plateaus with no higher neighbour. Labelled 1, 2, 3, ... in raster order; returns the number.
pub fn regional_maxima(f: &Image<f32>, mask: &Image<bool>) -> (Image<u32>, u32) {
    let mut seen = Image::new(f.width(), f.height(), false);
    let mut labels = Image::new(f.width(), f.height(), 0u32);
    let mut count = 0;
    for start in f.coords() {
        if seen[start] || !mask[start] {
            continue;
        }
        let v = f[start];
        let (mut plateau, mut queue, mut is_maximum) = (vec![start], VecDeque::from([start]), true);
        seen[start] = true;
        while let Some((x, y)) = queue.pop_front() {
            for &(dx, dy) in &N8 {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if f.contains(nx, ny) {
                    let q = (nx as usize, ny as usize);
                    if f[q] > v {
                        is_maximum = false;
                    } else if f[q] == v && mask[q] && !seen[q] {
                        seen[q] = true;
                        plateau.push(q);
                        queue.push_back(q);
                    }
                }
            }
        }
        if is_maximum {
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

/// Given (the flooding unit): Meyer's flooding of `f` from the `markers`, 8 neighbours, only
/// inside `mask` (the background is not flooded and stays 0). With `line`, a pixel whose
/// labelled neighbours carry different labels becomes a line pixel (0); without, it takes the
/// label of the pixel that queued it.
pub fn watershed(f: &Image<f32>, markers: &Image<u32>, mask: &Image<bool>, line: bool) -> Image<u32> {
    const LINE: u32 = u32::MAX;
    let mut labels = markers.clone();
    let mut queued = markers.map(|m| m > 0);
    let mut by = Image::new(f.width(), f.height(), 0u32);
    let mut heap = BinaryHeap::new();
    let mut arrivals = 0u64;
    let mut fresh: Vec<(usize, usize)> = f.coords().filter(|&p| markers[p] > 0).collect();
    loop {
        for p in fresh.drain(..) {
            for &(dx, dy) in &N8 {
                let (nx, ny) = (p.0 as isize + dx, p.1 as isize + dy);
                if f.contains(nx, ny) {
                    let q = (nx as usize, ny as usize);
                    if mask[q] && labels[q] == 0 && !queued[q] {
                        queued[q] = true;
                        by[q] = labels[p];
                        heap.push(Reverse(Entry(f[q], arrivals, q)));
                        arrivals += 1;
                    }
                }
            }
        }
        let Some(Reverse(Entry(_, _, p))) = heap.pop() else { break };
        let mut seen = 0;
        let mut different = false;
        for &(dx, dy) in &N8 {
            let (nx, ny) = (p.0 as isize + dx, p.1 as isize + dy);
            if f.contains(nx, ny) {
                let l = labels[(nx as usize, ny as usize)];
                if l != 0 && l != LINE {
                    different |= seen != 0 && l != seen;
                    seen = l;
                }
            }
        }
        if line && different {
            labels[p] = LINE;
        } else {
            labels[p] = by[p];
            fresh.push(p);
        }
    }
    labels.map(|l| if l == LINE { 0 } else { l })
}

// ---------------------------------------------------------------------------------------------
// The exercises
// ---------------------------------------------------------------------------------------------

/// Read the distance image at a position; outside the image the distance is 0 (background).
/// (A helper for C1 and C2.)
pub fn at(d: &Image<f32>, x: isize, y: isize) -> f32 {
    if d.contains(x, y) {
        d[(x as usize, y as usize)]
    } else {
        0.0
    }
}

/// Chamfer distance in two raster passes with step lengths `a` (to the 4 edge neighbours) and
/// `b` (to the 4 corner neighbours); outside the image is background.
fn chamfer(mask: &Image<bool>, a: f32, b: f32) -> Image<f32> {
    let (w, h) = (mask.width() as isize, mask.height() as isize);
    let mut d = mask.map(|m| if m { f32::INFINITY } else { 0.0 });
    // forward: the neighbours already visited (up-left, up, up-right, left)
    for y in 0..h {
        for x in 0..w {
            if mask[(x as usize, y as usize)] {
                let v = (at(&d, x - 1, y - 1) + b).min(at(&d, x, y - 1) + a).min(at(&d, x + 1, y - 1) + b).min(at(&d, x - 1, y) + a);
                d[(x as usize, y as usize)] = d[(x as usize, y as usize)].min(v);
            }
        }
    }
    // backward: the other four (right, down-left, down, down-right)
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            if mask[(x as usize, y as usize)] {
                let v = (at(&d, x + 1, y) + a).min(at(&d, x - 1, y + 1) + b).min(at(&d, x, y + 1) + a).min(at(&d, x + 1, y + 1) + b);
                d[(x as usize, y as usize)] = d[(x as usize, y as usize)].min(v);
            }
        }
    }
    d
}

/// C1. The city-block distance transform in two raster passes. Forward (row by row, left to
/// right): d(p) = min(d(p), d(up) + 1, d(left) + 1). Backward (from the bottom right): the same
/// with down and right. Start with 0 on the background and ∞ on the object; outside the image
/// is background.
pub fn distance_city_block(mask: &Image<bool>) -> Image<f32> {
    // a chamfer with an infinite diagonal step uses only the 4 edge neighbours
    chamfer(mask, 1.0, f32::INFINITY)
}

/// C2. The chamfer 3-4 distance: the same two passes with all 8 neighbours, steps of 3 to the
/// edge neighbours and 4 to the corner neighbours; divide the result by 3.
pub fn distance_chamfer34(mask: &Image<bool>) -> Image<f32> {
    chamfer(mask, 3.0, 4.0).map(|v| v / 3.0)
}

/// C3. The h-maxima transform: start with dt − h and dilate it, never going above dt:
/// m(p) = min(dt(p), max of m over p and its 8 neighbours), until nothing changes. Its
/// regional maxima are the peaks of dt that rise more than h above the saddle to a higher peak.
pub fn h_maxima(dt: &Image<f32>, h: f32) -> Image<f32> {
    let mut m = dt.map(|v| v - h);
    let (w, hh) = (dt.width() as isize, dt.height() as isize);
    let mut changed = true;
    while changed {
        changed = false;
        // forward and backward sweeps, in place
        for pass in 0..2 {
            for i in 0..(w * hh) {
                let k = if pass == 0 { i } else { w * hh - 1 - i };
                let (x, y) = (k % w, k / w);
                let mut hi = m[(x as usize, y as usize)];
                for &(dx, dy) in &N8 {
                    if m.contains(x + dx, y + dy) {
                        hi = hi.max(m[((x + dx) as usize, (y + dy) as usize)]);
                    }
                }
                let new = hi.min(dt[(x as usize, y as usize)]);
                if new > m[(x as usize, y as usize)] {
                    m[(x as usize, y as usize)] = new;
                    changed = true;
                }
            }
        }
    }
    m
}

/// C4. Split touching objects: markers = the regional maxima of h_maxima(dt, h) inside the mask;
/// flood −dt from them, inside the mask (`line`: with a watershed line). Returns the labels and
/// the number of objects.
pub fn split(dt: &Image<f32>, mask: &Image<bool>, h: f32, line: bool) -> (Image<u32>, u32) {
    let (markers, n) = regional_maxima(&h_maxima(dt, h), mask);
    let labels = watershed(&dt.map(|v| -v), &markers, mask, line);
    (labels, n)
}
