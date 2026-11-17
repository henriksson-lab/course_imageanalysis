// Model solution for the coding exercises of "Sparse coding and dictionary learning".
// A patch is a vector (`Vec<f64>`, the pixels row by row); a dictionary is a list of atoms,
// each a unit vector of the same length.

use imgcourse::{linalg, Image};

/// The dot product.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// The reconstruction `Σ c_j d_j` of a code.
pub fn rebuild(dict: &[Vec<f64>], code: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; dict[0].len()];
    for (atom, &c) in dict.iter().zip(code) {
        if c != 0.0 {
            for (o, a) in out.iter_mut().zip(atom) {
                *o += c * a;
            }
        }
    }
    out
}

/// C1. Sparse coding by orthogonal matching pursuit: pick the atom with the largest
/// `|d_jᵀ r|`, refit all chosen atoms by least squares, update the residual `r`; stop after
/// `max_atoms` atoms or when `‖r‖² ≤ max_error`. Returns one coefficient per atom.
pub fn omp(dict: &[Vec<f64>], y: &[f64], max_atoms: usize, max_error: f64) -> Vec<f64> {
    let mut code = vec![0.0; dict.len()];
    let mut chosen: Vec<usize> = Vec::new();
    let mut r = y.to_vec();
    while chosen.len() < max_atoms && dot(&r, &r) > max_error {
        // 1. the atom most correlated with the residual (atoms have length 1)
        let corr: Vec<f64> = dict.iter().map(|atom| dot(atom, &r).abs()).collect();
        let mut best = 0;
        for j in 1..dict.len() {
            if corr[j] > corr[best] {
                best = j;
            }
        }
        if chosen.contains(&best) {
            break; // nothing new correlates with the residual
        }
        chosen.push(best);
        // 2. refit: least squares with the chosen atoms as columns (pixels × atoms)
        let a: Vec<Vec<f64>> = (0..y.len()).map(|i| chosen.iter().map(|&j| dict[j][i]).collect()).collect();
        let Some(c) = linalg::least_squares(&a, y) else { break };
        code = vec![0.0; dict.len()];
        for (k, &j) in chosen.iter().enumerate() {
            code[j] = c[k];
        }
        // 3. the new residual
        let approx = rebuild(dict, &code);
        r = y.iter().zip(&approx).map(|(p, q)| p - q).collect();
    }
    code
}

/// C2. The K-SVD update of one atom: the unit vector `d` that best fits the `residuals` of
/// the patches that use it, with free coefficients. `d` is the top eigenvector of
/// `Σ r rᵀ` (found by power iteration, starting from `start`); the coefficients are the
/// projections `rᵀd`. Returns `(d, coefficients)`.
pub fn update_atom(residuals: &[Vec<f64>], start: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut d = start.to_vec();
    for _ in 0..50 {
        // (Σ r rᵀ) d = Σ r (rᵀd), without building the matrix
        let mut v = vec![0.0; d.len()];
        for r in residuals {
            let p = dot(r, &d);
            for (vi, ri) in v.iter_mut().zip(r) {
                *vi += p * ri;
            }
        }
        let len = dot(&v, &v).sqrt();
        if len == 0.0 {
            break; // all residuals are 0: keep the atom
        }
        d = v.iter().map(|x| x / len).collect();
    }
    let coefs = residuals.iter().map(|r| dot(r, &d)).collect();
    (d, coefs)
}

/// C3. K-SVD: `iterations` times, code every patch with OMP (`k` atoms each), then update
/// every atom in turn: collect the patches that use it, subtract the contributions of their
/// other atoms, and replace the atom and its coefficients by `update_atom`. Returns the
/// dictionary and the mean squared error per pixel after each iteration.
pub fn ksvd(patches: &[Vec<f64>], start: &[Vec<f64>], k: usize, iterations: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut dict = start.to_vec();
    let mut errors = Vec::new();
    for _ in 0..iterations {
        // 1. sparse coding
        let mut codes: Vec<Vec<f64>> = patches.iter().map(|p| omp(&dict, p, k, 0.0)).collect();
        // 2. atom updates, one at a time, each with the fresh codes
        for j in 0..dict.len() {
            let users: Vec<usize> = (0..patches.len()).filter(|&i| codes[i][j] != 0.0).collect();
            if users.is_empty() {
                continue;
            }
            let residuals: Vec<Vec<f64>> = users
                .iter()
                .map(|&i| {
                    let approx = rebuild(&dict, &codes[i]);
                    // the patch minus all its atoms except atom j
                    (0..approx.len()).map(|p| patches[i][p] - approx[p] + codes[i][j] * dict[j][p]).collect()
                })
                .collect();
            let (atom, coefs) = update_atom(&residuals, &dict[j]);
            dict[j] = atom;
            for (u, &i) in users.iter().enumerate() {
                codes[i][j] = coefs[u];
            }
        }
        let mut err = 0.0;
        for (p, c) in patches.iter().zip(&codes) {
            err += p.iter().zip(rebuild(&dict, c)).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
        }
        errors.push(err / (patches.len() * patches[0].len()) as f64);
    }
    (dict, errors)
}

/// C4. Denoise by coding every overlapping patch (step 1) with OMP until the residual is
/// at the noise level, `d (1.15 σ)²` for patches of `d` pixels (at most `d` atoms), then
/// average: every pixel becomes `(λ y + Σ of the patch reconstructions that cover it) /
/// (λ + the number of patches that cover it)`. The patch size comes from the atoms (`d` =
/// size²).
pub fn denoise(img: &Image<f32>, dict: &[Vec<f64>], sigma: f64, lambda: f64) -> Image<f32> {
    let d = dict[0].len();
    let size = (d as f64).sqrt().round() as usize;
    let (w, h) = (img.width(), img.height());
    let mut sum = Image::new(w, h, 0.0f64);
    let mut count = Image::new(w, h, 0.0f64);
    let target = d as f64 * (1.15 * sigma).powi(2);
    for y0 in 0..=h - size {
        for x0 in 0..=w - size {
            let patch: Vec<f64> = img.crop(x0, y0, size, size).pixels().iter().map(|&v| v as f64).collect();
            let approx = rebuild(dict, &omp(dict, &patch, d, target));
            for (i, v) in approx.iter().enumerate() {
                let (x, y) = (x0 + i % size, y0 + i / size);
                sum[(x, y)] += v;
                count[(x, y)] += 1.0;
            }
        }
    }
    Image::from_fn(w, h, |x, y| ((lambda * img[(x, y)] as f64 + sum[(x, y)]) / (lambda + count[(x, y)])) as f32)
}
