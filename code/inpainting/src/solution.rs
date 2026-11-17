// Model solution for the coding exercises of "Filling holes".

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
pub fn jacobi_sweep(img: &Image<f32>, missing: &Image<bool>) -> Image<f32> {
    let mut out = img.clone();
    for (x, y) in img.coords() {
        if missing[(x, y)] {
            out[(x, y)] = neighbour_mean(img, x, y);
        }
    }
    out
}

/// C2. One Gauss–Seidel sweep, row by row from the top left: the same update, but done in
/// place, so that new values are used as soon as they are computed.
pub fn gauss_seidel_sweep(img: &mut Image<f32>, missing: &Image<bool>) {
    for y in 0..img.height() {
        for x in 0..img.width() {
            if missing[(x, y)] {
                img[(x, y)] = neighbour_mean(img, x, y);
            }
        }
    }
}

/// C3. Harmonic inpainting: Gauss–Seidel sweeps until no pixel changes by more than `tol`.
/// The missing pixels start at 0. Returns the filled image and the number of sweeps.
pub fn harmonic_inpaint(img: &Image<f32>, missing: &Image<bool>, tol: f32) -> (Image<f32>, usize) {
    let mut u = img.zip_map(missing, |v, m| if m { 0.0 } else { v });
    let mut sweeps = 0;
    loop {
        let before = u.clone();
        gauss_seidel_sweep(&mut u, missing);
        sweeps += 1;
        let change = u.zip_map(&before, |a, b| (a - b).abs()).max();
        if change <= tol {
            return (u, sweeps);
        }
    }
}

/// C4. The masked patch distance of exemplar inpainting (slide "Exemplar inpainting"): the sum
/// of squared differences between the `(2 half + 1)²` patches around `target` and `source`,
/// over the offsets where the target pixel is known (not missing). Both patches must lie
/// inside the image.
pub fn masked_ssd(img: &Image<f32>, missing: &Image<bool>, target: (usize, usize), source: (usize, usize), half: usize) -> f32 {
    let h = half as isize;
    let mut sum = 0.0;
    for dy in -h..=h {
        for dx in -h..=h {
            let t = ((target.0 as isize + dx) as usize, (target.1 as isize + dy) as usize);
            let s = ((source.0 as isize + dx) as usize, (source.1 as isize + dy) as usize);
            if !missing[t] {
                sum += (img[t] - img[s]).powi(2);
            }
        }
    }
    sum
}

/// C5. One exemplar step: among all source centres within `search` pixels of `target` (in x
/// and in y) whose whole patch is inside the image and known, find the one with the smallest
/// `masked_ssd`; copy its values into the missing pixels of the target patch and mark them as
/// known. Returns the chosen source centre. The target patch must lie inside the image.
pub fn fill_best_patch(img: &mut Image<f32>, missing: &mut Image<bool>, target: (usize, usize), half: usize, search: usize) -> (usize, usize) {
    let (w, h) = (img.width(), img.height());
    let mut best = (f32::INFINITY, target);
    let y0 = target.1.saturating_sub(search).max(half);
    let x0 = target.0.saturating_sub(search).max(half);
    for sy in y0..=(target.1 + search).min(h - 1 - half) {
        for sx in x0..=(target.0 + search).min(w - 1 - half) {
            // the source patch must be completely known
            let known = (sy - half..=sy + half).all(|y| (sx - half..=sx + half).all(|x| !missing[(x, y)]));
            if !known {
                continue;
            }
            let d = masked_ssd(img, missing, target, (sx, sy), half);
            if d < best.0 {
                best = (d, (sx, sy));
            }
        }
    }
    let (sx, sy) = best.1;
    for k in 0..=2 * half {
        for l in 0..=2 * half {
            let t = (target.0 + l - half, target.1 + k - half);
            if missing[t] {
                img[t] = img[(sx + l - half, sy + k - half)];
                missing[t] = false;
            }
        }
    }
    best.1
}
