// Coding exercises for "Splitting touching objects with the distance transform". Replace each
// `todo!()` with your code, then check it with `cargo test -p day15 ws_distance::` and look at
// `cargo run -p day15 -- ws-distance`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;
use crate::common::{Entry, N8};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

// ---------------------------------------------------------------------------------------------
// Given (from earlier units, not part of this exercise)
// ---------------------------------------------------------------------------------------------

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
    let mut arrivals = 0i64;
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

/// C1. The city-block distance transform in two raster passes. Forward (row by row, left to
/// right): d(p) = min(d(p), d(up) + 1, d(left) + 1). Backward (from the bottom right): the same
/// with down and right. Start with 0 on the background and ∞ on the object; outside the image
/// is background.
///
/// Hint: `let mut d = mask.map(|m| if m { f32::INFINITY } else { 0.0 });`, then loop over y and x
/// (as `isize`, so that x − 1 is allowed) and read neighbours with `at(&d, x - 1, y)`.
pub fn distance_city_block(mask: &Image<bool>) -> Image<f32> {
    todo!("C1: distance_city_block")
}

/// C2. The chamfer 3-4 distance: the same two passes with all 8 neighbours, steps of 3 to the
/// edge neighbours and 4 to the corner neighbours; divide the result by 3.
///
/// Hint: as C1, with the four visited neighbours up-left, up, up-right, left in the forward pass
/// and right, down-left, down, down-right in the backward pass.
pub fn distance_chamfer34(mask: &Image<bool>) -> Image<f32> {
    todo!("C2: distance_chamfer34")
}

/// C3. The h-maxima transform: start with dt − h and dilate it, never going above dt:
/// m(p) = min(dt(p), max of m over p and its 8 neighbours), until nothing changes. Its
/// regional maxima are the peaks of dt that rise more than h above the saddle to a higher peak.
///
/// Hint: update `m` in place, sweeping forwards and then backwards, until a pair of sweeps
/// changes nothing (as h-minima in the gradient unit, with max and min swapped).
pub fn h_maxima(dt: &Image<f32>, h: f32) -> Image<f32> {
    todo!("C3: h_maxima")
}

/// C4. Split touching objects: markers = the regional maxima of h_maxima(dt, h) inside the mask;
/// flood −dt from them, inside the mask (`line`: with a watershed line). Returns the labels and
/// the number of objects.
///
/// Hint: `regional_maxima(&h_maxima(dt, h), mask)` gives the markers; flood `dt.map(|v| -v)`.
pub fn split(dt: &Image<f32>, mask: &Image<bool>, h: f32, line: bool) -> (Image<u32>, u32) {
    todo!("C4: split")
}
