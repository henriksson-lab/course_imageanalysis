// Coding exercises for "Filling holes". Replace each `todo!()` with your
// code, then check it with `cargo test -p day10 inpainting::` and look at `cargo run -p day10 -- inpainting`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// Not an exercise: the mean of the 4 neighbours of `(x, y)` that lie inside the image
/// (in a single row, the 2 neighbours in that row: 1D inpainting).
pub fn neighbour_mean(img: &Image<f32>, x: usize, y: usize) -> f32 {
    let (mut sum, mut n) = (0.0, 0.0);
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let (qx, qy) = (x as isize + dx, y as isize + dy);
        if img.contains(qx, qy) {
            sum += img[(qx as usize, qy as usize)];
            n += 1.0;
        }
    }
    sum / n
}

/// C1. One Jacobi sweep of harmonic inpainting: every missing pixel (`missing` true) becomes
/// the mean of its neighbours inside the image, all computed from the **old** values; the
/// known pixels stay as they are.
///
/// Hint: start from `img.clone()` and write into the copy, reading only `img`; `neighbour_mean`
/// (above) gives the mean of the neighbours inside the image.
pub fn jacobi_sweep(img: &Image<f32>, missing: &Image<bool>) -> Image<f32> {
    todo!("C1: jacobi_sweep")
}

/// C2. One Gauss–Seidel sweep, row by row from the top left: the same update, but done in
/// place, so that new values are used as soon as they are computed.
///
/// Hint: the same as C1, but read and write `img` itself, in the order `for y { for x { ... } }`.
pub fn gauss_seidel_sweep(img: &mut Image<f32>, missing: &Image<bool>) {
    todo!("C2: gauss_seidel_sweep")
}

/// C3. Harmonic inpainting: Gauss–Seidel sweeps until no pixel changes by more than `tol`.
/// The missing pixels start at 0. Returns the filled image and the number of sweeps.
///
/// Hint: `img.zip_map(missing, |v, m| if m { 0.0 } else { v })` sets the hole to 0. In a loop, keep
/// a copy of the image before each sweep to measure the largest change.
pub fn harmonic_inpaint(img: &Image<f32>, missing: &Image<bool>, tol: f32) -> (Image<f32>, usize) {
    todo!("C3: harmonic_inpaint")
}

/// C4. The masked patch distance of exemplar inpainting (slide "Exemplar inpainting"): the sum
/// of squared differences between the `(2 half + 1)²` patches around `target` and `source`,
/// over the offsets where the target pixel is known (not missing). Both patches must lie
/// inside the image.
pub fn masked_ssd(img: &Image<f32>, missing: &Image<bool>, target: (usize, usize), source: (usize, usize), half: usize) -> f32 {
    todo!("C4: masked_ssd")
}

/// C5. One exemplar step: among all source centres within `search` pixels of `target` (in x
/// and in y) whose whole patch is inside the image and known, find the one with the smallest
/// `masked_ssd`; copy its values into the missing pixels of the target patch and mark them as
/// known. Returns the chosen source centre. The target patch must lie inside the image.
///
/// Hint: loop over the candidate centres `(sx, sy)` with the whole patch inside the image
/// (`half <= sx < width - half`, the same for `sy`) and within `search` of the target; skip a
/// candidate if any pixel of its patch is missing. Keep the smallest `masked_ssd`, then copy.
pub fn fill_best_patch(img: &mut Image<f32>, missing: &mut Image<bool>, target: (usize, usize), half: usize, search: usize) -> (usize, usize) {
    todo!("C5: fill_best_patch")
}
