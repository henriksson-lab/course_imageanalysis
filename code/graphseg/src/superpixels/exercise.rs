// Coding exercises for "Superpixels: SLIC and smaller graphs".
// Replace each `todo!()` with your code, then check it with `cargo test -p graphseg superpixels::`
// and look at `cargo run -p graphseg -- superpixels`.
#![allow(unused_variables, unused_imports)]

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
    todo!("C1: slic_distance2")
}

/// C2. The start: centres on a regular grid with spacing `S = sqrt(N/K)`, at
/// `((i + ½)S, (j + ½)S)` (rounded down to a pixel), each moved to the pixel of lowest
/// gradient `(I(x+1, y) − I(x−1, y))² + (I(x, y+1) − I(x, y−1))²` in its 3 × 3 neighbourhood
/// (only if strictly lower than at the grid point). Returns the centres and `S`.
///
/// Hint: walk `gy` and `gx` from `0.5 * s` in steps of `s` while inside the image; compute the
/// gradient with `img.get_border(.., Border::Clamp)` and keep the lowest of the 3 × 3 pixels.
pub fn slic_init(img: &Image<f32>, k: usize) -> (Vec<Centre>, f32) {
    todo!("C2: slic_init")
}

/// C3. One SLIC iteration. Assignment: every centre `k` visits the pixels of its window
/// `[x − S, x + S) × [y − S, y + S)` (inside the image) and takes each pixel that is nearer to
/// it (C1) than to the centre that holds it so far (`labels[p] = k + 1`; pixels no window
/// reaches keep their old label). Update: every centre moves to the mean `(x, y, v)` of its
/// pixels. Returns the number of distances computed.
///
/// Hint: keep a distance image (start: `f32::INFINITY`). Clip the window to the image with
/// `.max(0)` and `.min(w)`. For the update, sum x, y, v and the count per label.
pub fn slic_step(img: &Image<f32>, centres: &mut [Centre], labels: &mut Image<u32>, s: f32, m: f32) -> usize {
    todo!("C3: slic_step")
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
///
/// Hint: `pieces` (given) numbers the 4-connected pieces and gives their sizes. Go through
/// the pixels in scan order: the first pixel of a piece decides its new label, and the pixel to
/// its left (or above) belongs to a piece that is already decided.
pub fn enforce_connectivity(labels: &Image<u32>) -> Image<u32> {
    todo!("C4: enforce_connectivity")
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
///
/// Hint: `overlaps` (given) counts the pixels of every (superpixel, region) pair.
pub fn achievable_accuracy(sp: &Image<u32>, truth: &Image<u32>) -> f32 {
    todo!("C5a: achievable_accuracy")
}

/// C5b. Under-segmentation error: `(1/N) Σ_s Σ_r min(|s ∩ r|, |s \ r|)` over the superpixels and
/// the true regions they overlap.
pub fn undersegmentation_error(sp: &Image<u32>, truth: &Image<u32>) -> f32 {
    todo!("C5b: undersegmentation_error")
}

/// C6. Boundary recall: the fraction of the true boundary edges (pairs of 4-neighbours with
/// different true labels) that have a superpixel boundary edge of the same direction (left–right
/// or up–down) whose first pixel lies within `tol` pixels (in x and in y) of theirs.
///
/// Hint: a pair (p, p + right) is a boundary edge of an image if the two labels differ;
/// the same for (p, p + down).
pub fn boundary_recall(sp: &Image<u32>, truth: &Image<u32>, tol: usize) -> f32 {
    todo!("C6: boundary_recall")
}
