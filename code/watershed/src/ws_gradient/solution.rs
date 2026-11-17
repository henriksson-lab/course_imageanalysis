// Model solution for the coding exercises of "Watershed segmentation of a real image".

use imgcourse::{Border, Image};
use crate::common::{Entry, N8};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

// ---------------------------------------------------------------------------------------------
// Given (from earlier units, not part of this exercise): smoothing, regional minima, flooding.
// ---------------------------------------------------------------------------------------------

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

/// Given (the flooding unit's C2): Meyer's flooding of `f` from the `markers` with 8 neighbours,
/// every pixel labelled (each takes the label of the pixel that queued it; no watershed line).
pub fn watershed(f: &Image<f32>, markers: &Image<u32>) -> Image<u32> {
    let mut labels = markers.clone();
    let mut queued = markers.map(|m| m > 0);
    let mut heap = BinaryHeap::new();
    let mut arrivals = 0i64;
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
pub fn gradient_l1(img: &Image<f32>) -> Image<f32> {
    let at = |x: usize, y: usize, dx: isize, dy: isize| img.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
    img.map_xy(|x, y, _| {
        let gx = at(x, y, 1, -1) + 2.0 * at(x, y, 1, 0) + at(x, y, 1, 1) - at(x, y, -1, -1) - 2.0 * at(x, y, -1, 0) - at(x, y, -1, 1);
        let gy = at(x, y, -1, 1) + 2.0 * at(x, y, 0, 1) + at(x, y, 1, 1) - at(x, y, -1, -1) - 2.0 * at(x, y, 0, -1) - at(x, y, 1, -1);
        gx.abs() + gy.abs()
    })
}

/// C2. The h-minima transform with 8 neighbours: start with g + h and erode it, never going
/// below g: m(p) = max(g(p), min of m over p and its 8 neighbours), until nothing changes.
pub fn h_minima(g: &Image<f32>, h: f32) -> Image<f32> {
    let mut m = g.map(|v| v + h);
    let (w, hh) = (g.width() as isize, g.height() as isize);
    let mut changed = true;
    while changed {
        changed = false;
        // a forward and a backward sweep, updating in place: the minimum travels far in one sweep
        for pass in 0..2 {
            for i in 0..(w * hh) {
                let k = if pass == 0 { i } else { w * hh - 1 - i };
                let (x, y) = (k % w, k / w);
                let mut lo = m[(x as usize, y as usize)];
                for &(dx, dy) in &N8 {
                    if m.contains(x + dx, y + dy) {
                        lo = lo.min(m[((x + dx) as usize, (y + dy) as usize)]);
                    }
                }
                let new = lo.max(g[(x as usize, y as usize)]);
                if new < m[(x as usize, y as usize)] {
                    m[(x as usize, y as usize)] = new;
                    changed = true;
                }
            }
        }
    }
    m
}

/// C3. The dams between neighbouring regions: for every pair of labels a < b that touch
/// (4-neighbours), the lowest dam height, where a dam between pixels p and q is max(g(p), g(q)).
/// Returns (a, b, height), sorted by a, then b.
pub fn dam_heights(labels: &Image<u32>, g: &Image<f32>) -> Vec<(u32, u32, f32)> {
    let mut best = std::collections::BTreeMap::new();
    for (x, y) in labels.coords() {
        for (dx, dy) in [(1, 0), (0, 1)] {
            let (nx, ny) = (x + dx, y + dy);
            if nx >= labels.width() || ny >= labels.height() {
                continue;
            }
            let (a, b) = (labels[(x, y)], labels[(nx, ny)]);
            if a == b || a == 0 || b == 0 {
                continue;
            }
            let key = (a.min(b), a.max(b));
            let height = g[(x, y)].max(g[(nx, ny)]);
            let e = best.entry(key).or_insert(height);
            *e = f32::min(*e, height);
        }
    }
    best.into_iter().map(|((a, b), h)| (a, b, h)).collect()
}

/// C4. Merge the regions across every dam of height ≤ t (union–find on the labels), and number
/// the merged regions 1, 2, 3, ... in order of their smallest old label. Returns the new label
/// image and the number of regions.
pub fn merge_below(labels: &Image<u32>, dams: &[(u32, u32, f32)], t: f32) -> (Image<u32>, u32) {
    let n = labels.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut parent: Vec<usize> = (0..=n).collect();
    fn find(parent: &[usize], mut x: usize) -> usize {
        while parent[x] != x {
            x = parent[x];
        }
        x
    }
    for &(a, b, h) in dams {
        if h <= t {
            let (ra, rb) = (find(&parent, a as usize), find(&parent, b as usize));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb); // the smaller label is the root
            }
        }
    }
    let mut new_label = vec![0u32; n + 1];
    let mut count = 0;
    for l in 1..=n {
        let r = find(&parent, l);
        if new_label[r] == 0 {
            count += 1;
            new_label[r] = count;
        }
        new_label[l] = new_label[r];
    }
    (labels.map(|l| new_label[l as usize]), count)
}
