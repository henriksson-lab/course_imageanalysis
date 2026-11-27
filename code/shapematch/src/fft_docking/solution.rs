// Model solution for the coding exercises of "Docking by correlation".

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
pub fn score_1d(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut s = vec![0.0; n];
    for d in 0..n {
        for x in 0..n {
            s[d] += a[x] * b[(x + n - d) % n];
        }
    }
    s
}

/// C2. The receptor grid: `rho` on the receptor's own cells (`mask > 0.5`, the core), 1 on the
/// surface layer (cells outside the mask with a mask cell at distance at most `thickness`,
/// i.e. `dx² + dy² <= thickness²`: dilation by a disk minus the object), 0 elsewhere.
pub fn receptor_grid(mask: &Image<f32>, thickness: usize, rho: f32) -> Image<f32> {
    let t = thickness as isize;
    mask.map_xy(|x, y, v| {
        if v > 0.5 {
            return rho;
        }
        for dy in -t..=t {
            for dx in -t..=t {
                if dx * dx + dy * dy > t * t {
                    continue;
                }
                let (u, w) = (x as isize + dx, y as isize + dy);
                if mask.contains(u, w) && mask[(u as usize, w as usize)] > 0.5 {
                    return 1.0;
                }
            }
        }
        0.0
    })
}

/// C3. The ligand (a small image whose centre cell is `(w/2, h/2)`) as an `m × m` grid with
/// its centre cell at `(0, 0)`: cell `(i, j)` of the ligand goes to `((i − w/2) mod m,
/// (j − h/2) mod m)`. Then `S(d) = Σ a(x) b(x − d)` is the score with the ligand's centre at `d`.
pub fn ligand_grid(ligand: &Image<f32>, m: usize) -> Image<f32> {
    let (cx, cy) = (ligand.width() / 2, ligand.height() / 2);
    let mut b = Image::new(m, m, 0.0);
    for (i, j) in ligand.coords() {
        let x = (i + m - cx) % m;
        let y = (j + m - cy) % m;
        b[(x, y)] = ligand[(i, j)];
    }
    b
}

/// C3. The score of one pose, directly: the sum of the receptor grid under the ligand's cells
/// with the ligand's centre cell at `(x, y)` (wrapping around the receptor grid).
pub fn score_at(receptor: &Image<f32>, ligand: &Image<f32>, x: usize, y: usize) -> f32 {
    let (m, n) = (receptor.width() as isize, receptor.height() as isize);
    let (cx, cy) = ((ligand.width() / 2) as isize, (ligand.height() / 2) as isize);
    let mut s = 0.0;
    for (i, j) in ligand.coords() {
        let v = ligand[(i, j)];
        if v != 0.0 {
            let u = (x as isize + i as isize - cx).rem_euclid(m) as usize;
            let w = (y as isize + j as isize - cy).rem_euclid(n) as usize;
            s += v * receptor[(u, w)];
        }
    }
    s
}

/// C3. The score map of one ligand orientation for every shift, with the FFT: `fa` is the
/// receptor's transform (computed once), the ligand goes onto an `m × m` grid with
/// `ligand_grid`, then `S = inverse FFT of fa · conj(FFT(b))`, real part.
pub fn score_map(fa: &Image<Complex>, ligand: &Image<f32>) -> Image<f32> {
    let b = ligand_grid(ligand, fa.width());
    let fb = fft2(&b.map(Complex::from), false);
    let product = fa.zip_map(&fb, |p, q| p * q.conj());
    fft2(&product, true).map(|z| z.re)
}

/// C4. The ligand turned by `degrees` clockwise on screen about its centre cell `(w/2, h/2)`,
/// same size: inverse mapping with the nearest cell, as on the images day.
pub fn rotate_ligand(ligand: &Image<f32>, degrees: f32) -> Image<f32> {
    let (c, s) = (degrees.to_radians().cos(), degrees.to_radians().sin());
    let (cx, cy) = ((ligand.width() / 2) as f32, (ligand.height() / 2) as f32);
    Image::from_fn(ligand.width(), ligand.height(), |x, y| {
        // the output cell (x, y) comes from the input cell turned back by −degrees
        let (u, v) = (x as f32 - cx, y as f32 - cy);
        let xs = (c * u + s * v + cx).round() as isize;
        let ys = (-s * u + c * v + cy).round() as isize;
        if ligand.contains(xs, ys) { ligand[(xs as usize, ys as usize)] } else { 0.0 }
    })
}

/// C5. Docking: for every angle `0, step, 2·step, … < 360`, turn the ligand, compute its score
/// map with one correlation, and keep the best shift. The receptor's FFT is computed once.
/// Returns the best pose of every angle, in the order of the angles.
pub fn dock(receptor: &Image<f32>, ligand: &Image<f32>, step: f32) -> Vec<Pose> {
    let fa = fft2(&receptor.map(Complex::from), false);
    let count = (360.0 / step - 1e-4).ceil() as usize;
    (0..count)
        .map(|k| {
            let angle = k as f32 * step;
            let map = score_map(&fa, &rotate_ligand(ligand, angle));
            let (mut bx, mut by) = (0, 0);
            for (x, y) in map.coords() {
                if map[(x, y)] > map[(bx, by)] {
                    (bx, by) = (x, y);
                }
            }
            Pose { angle, x: bx, y: by, score: map[(bx, by)] }
        })
        .collect()
}
