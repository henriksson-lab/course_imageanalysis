// Model solution for the coding exercises of "Superpixels: SLIC and smaller graphs".

use imgcourse::{Border, Image};

/// A SLIC centre: position `(x, y)` and grey value `v`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Centre {
    pub x: f32,
    pub y: f32,
    pub v: f32,
}

/// C1. The SLIC distance (squared) between a pixel and a centre that differ by `dv` in grey
/// value and by `(dx, dy)` in position: `D² = dv² + (m · d_s / S)²`, `d_s² = dx² + dy²`.
pub fn slic_distance2(dv: f32, dx: f32, dy: f32, m: f32, s: f32) -> f32 {
    dv * dv + (m / s).powi(2) * (dx * dx + dy * dy)
}

/// C2. The start: centres on a regular grid with spacing `S = sqrt(N/K)`, at
/// `((i + ½)S, (j + ½)S)` (rounded down to a pixel), each moved to the pixel of lowest
/// gradient `(I(x+1, y) − I(x−1, y))² + (I(x, y+1) − I(x, y−1))²` in its 3 × 3 neighbourhood
/// (only if strictly lower than at the grid point). Returns the centres and `S`.
pub fn slic_init(img: &Image<f32>, k: usize) -> (Vec<Centre>, f32) {
    let (w, h) = (img.width(), img.height());
    let s = ((w * h) as f32 / k as f32).sqrt();
    let grad = |x: isize, y: isize| {
        let v = |x: isize, y: isize| img.get_border(x, y, Border::Clamp);
        (v(x + 1, y) - v(x - 1, y)).powi(2) + (v(x, y + 1) - v(x, y - 1)).powi(2)
    };
    let mut centres = Vec::new();
    let mut gy = 0.5 * s;
    while gy < h as f32 {
        let mut gx = 0.5 * s;
        while gx < w as f32 {
            let (mut bx, mut by) = (gx as isize, gy as isize);
            let mut best = grad(bx, by);
            let (cx, cy) = (bx, by);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (x, y) = (cx + dx, cy + dy);
                    if img.contains(x, y) && grad(x, y) < best {
                        best = grad(x, y);
                        (bx, by) = (x, y);
                    }
                }
            }
            centres.push(Centre { x: bx as f32, y: by as f32, v: img[(bx as usize, by as usize)] });
            gx += s;
        }
        gy += s;
    }
    (centres, s)
}

/// C3. One SLIC iteration. Assignment: every centre `k` visits the pixels of its window
/// `[x − S, x + S) × [y − S, y + S)` (inside the image) and takes each pixel that is nearer to
/// it (C1) than to the centre that holds it so far (`labels[p] = k + 1`; pixels no window
/// reaches keep their old label). Update: every centre moves to the mean `(x, y, v)` of its
/// pixels. Returns the number of distances computed.
pub fn slic_step(img: &Image<f32>, centres: &mut [Centre], labels: &mut Image<u32>, s: f32, m: f32) -> usize {
    let (w, h) = (img.width() as isize, img.height() as isize);
    let mut dist = Image::new(img.width(), img.height(), f32::INFINITY);
    let mut computed = 0;
    for (k, c) in centres.iter().enumerate() {
        let (x0, x1) = (((c.x - s).ceil() as isize).max(0), ((c.x + s).ceil() as isize).min(w));
        let (y0, y1) = (((c.y - s).ceil() as isize).max(0), ((c.y + s).ceil() as isize).min(h));
        for y in y0..y1 {
            for x in x0..x1 {
                let p = (x as usize, y as usize);
                let d = slic_distance2(img[p] - c.v, x as f32 - c.x, y as f32 - c.y, m, s);
                computed += 1;
                if d < dist[p] {
                    dist[p] = d;
                    labels[p] = k as u32 + 1;
                }
            }
        }
    }
    // update: the mean of every centre's pixels
    let mut sum = vec![(0.0f32, 0.0f32, 0.0f32, 0usize); centres.len()];
    for (x, y) in img.coords() {
        let l = labels[(x, y)];
        if l > 0 {
            let e = &mut sum[l as usize - 1];
            *e = (e.0 + x as f32, e.1 + y as f32, e.2 + img[(x, y)], e.3 + 1);
        }
    }
    for (c, &(sx, sy, sv, n)) in centres.iter_mut().zip(&sum) {
        if n > 0 {
            *c = Centre { x: sx / n as f32, y: sy / n as f32, v: sv / n as f32 };
        }
    }
    computed
}

/// SLIC: `slic_init`, then `iterations` times `slic_step`. Labels `1 ..= K'` (K' centres).
pub fn slic(img: &Image<f32>, k: usize, m: f32, iterations: usize) -> Image<u32> {
    let (mut centres, s) = slic_init(img, k);
    let mut labels = Image::new(img.width(), img.height(), 0u32);
    for _ in 0..iterations {
        slic_step(img, &mut centres, &mut labels, s, m);
    }
    labels
}

/// The 4-connected pieces of a label image: a piece number for every pixel (0, 1, 2, … in
/// scan order of their first pixel), and the size of every piece.
pub fn pieces(labels: &Image<u32>) -> (Image<usize>, Vec<usize>) {
    let mut piece = Image::new(labels.width(), labels.height(), usize::MAX);
    let mut sizes = Vec::new();
    for start in labels.coords() {
        if piece[start] != usize::MAX {
            continue;
        }
        let id = sizes.len();
        let mut stack = vec![start];
        piece[start] = id;
        let mut size = 0;
        while let Some((x, y)) = stack.pop() {
            size += 1;
            for (dx, dy) in [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
                let (qx, qy) = (x as isize + dx, y as isize + dy);
                if labels.contains(qx, qy) {
                    let q = (qx as usize, qy as usize);
                    if piece[q] == usize::MAX && labels[q] == labels[start] {
                        piece[q] = id;
                        stack.push(q);
                    }
                }
            }
        }
        sizes.push(size);
    }
    (piece, sizes)
}

/// C4. Connectivity clean-up: every label keeps its largest 4-connected piece; every other
/// piece (an orphan) takes the label of the pixel to the left of its first pixel in scan
/// order (above it, in the first column). Then the labels are renumbered `1, 2, …`.
pub fn enforce_connectivity(labels: &Image<u32>) -> Image<u32> {
    let (piece, sizes) = pieces(labels);
    // the largest piece of every label
    let mut largest: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for (p, &id) in piece.pixels().iter().enumerate() {
        let l = labels.pixels()[p];
        let e = largest.entry(l).or_insert(id);
        if sizes[id] > sizes[*e] {
            *e = id;
        }
    }
    // new label of every piece, decided in scan order (earlier pieces are already final)
    let mut new_label = vec![u32::MAX; sizes.len()];
    let mut out = labels.clone();
    for (x, y) in labels.coords() {
        let id = piece[(x, y)];
        if new_label[id] == u32::MAX {
            new_label[id] = if largest[&labels[(x, y)]] == id {
                labels[(x, y)]
            } else if x > 0 {
                out[(x - 1, y)]
            } else if y > 0 {
                out[(x, y - 1)]
            } else {
                labels[(x, y)] // the very first pixel: keep
            };
        }
        out[(x, y)] = new_label[id];
    }
    // renumber 1, 2, … in scan order
    let mut renumber = std::collections::HashMap::new();
    Image::from_fn(out.width(), out.height(), |x, y| {
        let next = renumber.len() as u32 + 1;
        *renumber.entry(out[(x, y)]).or_insert(next)
    })
}

/// For every pair (superpixel `s`, true region `r`) that meet: the number of pixels in both.
pub fn overlaps(sp: &Image<u32>, truth: &Image<u32>) -> std::collections::HashMap<(u32, u32), usize> {
    let mut o = std::collections::HashMap::new();
    for (&s, &r) in sp.pixels().iter().zip(truth.pixels()) {
        *o.entry((s, r)).or_insert(0) += 1;
    }
    o
}

/// C5a. Achievable segmentation accuracy: give every superpixel the true region it overlaps
/// most; the fraction of pixels that are then right, `(1/N) Σ_s max_r |s ∩ r|`.
pub fn achievable_accuracy(sp: &Image<u32>, truth: &Image<u32>) -> f32 {
    let mut best: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for ((s, _), n) in overlaps(sp, truth) {
        let e = best.entry(s).or_insert(0);
        *e = (*e).max(n);
    }
    best.values().sum::<usize>() as f32 / sp.pixels().len() as f32
}

/// C5b. Under-segmentation error: `(1/N) Σ_s Σ_r min(|s ∩ r|, |s \ r|)` over the superpixels and
/// the true regions they overlap.
pub fn undersegmentation_error(sp: &Image<u32>, truth: &Image<u32>) -> f32 {
    let mut size: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for &s in sp.pixels() {
        *size.entry(s).or_insert(0) += 1;
    }
    let total: usize = overlaps(sp, truth).iter().map(|(&(s, _), &inside)| inside.min(size[&s] - inside)).sum();
    total as f32 / sp.pixels().len() as f32
}

/// C6. Boundary recall: the fraction of the true boundary edges (pairs of 4-neighbours with
/// different true labels) that have a superpixel boundary edge of the same direction (left–right
/// or up–down) whose first pixel lies within `tol` pixels (in x and in y) of theirs.
pub fn boundary_recall(sp: &Image<u32>, truth: &Image<u32>, tol: usize) -> f32 {
    let (w, h) = (sp.width(), sp.height());
    let edge = |img: &Image<u32>, x: usize, y: usize, right: bool| {
        if right {
            x + 1 < w && img[(x, y)] != img[(x + 1, y)]
        } else {
            y + 1 < h && img[(x, y)] != img[(x, y + 1)]
        }
    };
    let t = tol as isize;
    let (mut found, mut all) = (0, 0);
    for (x, y) in truth.coords() {
        for right in [true, false] {
            if !edge(truth, x, y, right) {
                continue;
            }
            all += 1;
            let near = (-t..=t).any(|dy| {
                (-t..=t).any(|dx| {
                    let (qx, qy) = (x as isize + dx, y as isize + dy);
                    sp.contains(qx, qy) && edge(sp, qx as usize, qy as usize, right)
                })
            });
            if near {
                found += 1;
            }
        }
    }
    if all == 0 {
        1.0
    } else {
        found as f32 / all as f32
    }
}
