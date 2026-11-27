// Coding exercises for "Docking by correlation". Replace each `todo!()` with your code,
// then check it with `cargo test -p shapematch fft_docking::` and look at
// `cargo run -p shapematch -- fft-docking`.
#![allow(unused_variables, unused_imports)]

use imgcourse::fft::fft2;
use imgcourse::{Complex, Image};

/// A ligand pose: the angle (degrees, clockwise on screen), the cell of the ligand's centre in
/// the receptor grid, and the score there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub angle: f32,
    pub x: usize,
    pub y: usize,
    pub score: f32,
}

/// C1. The docking score in 1D, directly: `S[d] = sum over x of a[x] · b[(x − d) mod N]` for
/// every shift `d = 0 .. N−1`: the ligand `b` (cells at 0, 1, …) moved `d` cells to the right
/// and laid on the receptor `a` (slide "By hand: a receptor of eight cells").
///
/// Hint: two loops; `(x + n - d) % n` wraps around without going negative.
pub fn score_1d(a: &[f32], b: &[f32]) -> Vec<f32> {
    todo!("C1: score_1d")
}

/// C2. The receptor grid: `rho` on the receptor's own cells (`mask > 0.5`, the core), 1 on the
/// surface layer (cells outside the mask with a mask cell at distance at most `thickness`,
/// i.e. `dx² + dy² <= thickness²`: dilation by a disk minus the object), 0 elsewhere.
///
/// Hint: `mask.map_xy(|x, y, v| ...)`; for a cell outside the mask, look at the offsets
/// `dx, dy` in `-t..=t` inside the disk; `mask.contains(u, w)` takes `isize` coordinates.
pub fn receptor_grid(mask: &Image<f32>, thickness: usize, rho: f32) -> Image<f32> {
    todo!("C2: receptor_grid")
}

/// C3. The ligand (a small image whose centre cell is `(w/2, h/2)`) as an `m × m` grid with
/// its centre cell at `(0, 0)`: cell `(i, j)` of the ligand goes to `((i − w/2) mod m,
/// (j − h/2) mod m)`. Then `S(d) = Σ a(x) b(x − d)` is the score with the ligand's centre at `d`.
///
/// Hint: `Image::new(m, m, 0.0)`, then copy every cell; `(i + m - cx) % m` wraps around.
pub fn ligand_grid(ligand: &Image<f32>, m: usize) -> Image<f32> {
    todo!("C3: ligand_grid")
}

/// C3. The score of one pose, directly: the sum of the receptor grid under the ligand's cells
/// with the ligand's centre cell at `(x, y)` (wrapping around the receptor grid).
///
/// Hint: loop over the ligand's cells; skip the zeros; `rem_euclid` wraps negative indices.
pub fn score_at(receptor: &Image<f32>, ligand: &Image<f32>, x: usize, y: usize) -> f32 {
    todo!("C3: score_at")
}

/// C3. The score map of one ligand orientation for every shift, with the FFT: `fa` is the
/// receptor's transform (computed once), the ligand goes onto an `m × m` grid with
/// `ligand_grid`, then `S = inverse FFT of fa · conj(FFT(b))`, real part.
///
/// Hint: as `xcorr2` on the FFT registration day, without subtracting means:
/// `fft2(&b.map(Complex::from), false)`, `zip_map` for the product, `fft2(.., true)`, `.re`.
pub fn score_map(fa: &Image<Complex>, ligand: &Image<f32>) -> Image<f32> {
    todo!("C3: score_map")
}

/// C4. The ligand turned by `degrees` clockwise on screen about its centre cell `(w/2, h/2)`,
/// same size: inverse mapping with the nearest cell, as on the images day.
///
/// Hint: for every output cell, `(u, v)` = its offset from the centre; the source cell is
/// `(c·u + s·v, −s·u + c·v)` + centre, rounded (`c`, `s` = cos and sin of the angle); 0 if
/// outside.
pub fn rotate_ligand(ligand: &Image<f32>, degrees: f32) -> Image<f32> {
    todo!("C4: rotate_ligand")
}

/// C5. Docking: for every angle `0, step, 2·step, … < 360`, turn the ligand, compute its score
/// map with one correlation, and keep the best shift. The receptor's FFT is computed once.
/// Returns the best pose of every angle, in the order of the angles.
///
/// Hint: `fa = fft2(&receptor.map(Complex::from), false)` before the loop; the number of
/// angles is `(360.0 / step).ceil()` (minus a hair, so that 360/step = 72 gives 72).
pub fn dock(receptor: &Image<f32>, ligand: &Image<f32>, step: f32) -> Vec<Pose> {
    todo!("C5: dock")
}
