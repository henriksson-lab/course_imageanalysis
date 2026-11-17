// Model solution for the coding exercises of "Localised basis functions and the Haar transform".

/// The scale of one Haar step: `1/2` by hand (averages and half-differences), `1/√2` for the
/// orthonormal transform.
pub fn scale(orthonormal: bool) -> f32 {
    if orthonormal {
        std::f32::consts::FRAC_1_SQRT_2
    } else {
        0.5
    }
}

/// C1a. One Haar step: every pair `(a, b)` = `(x[2k], x[2k+1])` becomes the average
/// `s = (a + b)·scale` and the difference `d = (a − b)·scale` (slide "The Haar transform by hand").
/// Returns `(s, d)`, each half as long as `x`.
pub fn haar_step(x: &[f32], orthonormal: bool) -> (Vec<f32>, Vec<f32>) {
    let f = scale(orthonormal);
    let mut s = Vec::new();
    let mut d = Vec::new();
    for k in 0..x.len() / 2 {
        let (a, b) = (x[2 * k], x[2 * k + 1]);
        s.push((a + b) * f);
        d.push((a - b) * f);
    }
    (s, d)
}

/// C1b. The full Haar transform of `N = 2^J` samples: repeat the step on the averages until one
/// is left. The result is `(s | d_J | ... | d_2 | d_1)`, coarse to fine: the overall average
/// first, the finest differences in the second half.
pub fn haar_forward(x: &[f32], orthonormal: bool) -> Vec<f32> {
    let mut c = x.to_vec();
    let mut len = c.len();
    while len > 1 {
        let (s, d) = haar_step(&c[..len], orthonormal);
        let half = len / 2;
        c[..half].copy_from_slice(&s);
        c[half..len].copy_from_slice(&d);
        len = half;
    }
    c
}

/// C2. The inverse transform: from the coarsest level up, every `(s, d)` becomes the pair
/// `a = s + d`, `b = s − d` (by hand), or `(s + d)/√2`, `(s − d)/√2` (orthonormal)
/// (slide "Going back: nothing is lost").
pub fn haar_inverse(c: &[f32], orthonormal: bool) -> Vec<f32> {
    // By hand the step divided by 2, so going back multiplies by 1; orthonormal: 1/√2 both ways.
    let f = if orthonormal { std::f32::consts::FRAC_1_SQRT_2 } else { 1.0 };
    let mut x = c.to_vec();
    let mut len = 1;
    while len < x.len() {
        let mut pairs = Vec::new();
        for k in 0..len {
            let (s, d) = (x[k], x[len + k]);
            pairs.push((s + d) * f);
            pairs.push((s - d) * f);
        }
        x[..2 * len].copy_from_slice(&pairs);
        len *= 2;
    }
    x
}

/// C3. Keep the `k` coefficients of largest magnitude and set the others to 0. Returns the kept
/// vector and the sum of the squares of the dropped coefficients (in an orthonormal basis: the
/// squared error of the reconstruction, slide "The math: error of keeping K coefficients").
pub fn keep_largest(c: &[f32], k: usize) -> (Vec<f32>, f32) {
    let mut order: Vec<usize> = (0..c.len()).collect();
    order.sort_by(|&i, &j| c[j].abs().total_cmp(&c[i].abs())); // largest first
    let mut kept = vec![0.0; c.len()];
    let mut dropped = 0.0;
    for (rank, &i) in order.iter().enumerate() {
        if rank < k {
            kept[i] = c[i];
        } else {
            dropped += c[i] * c[i];
        }
    }
    (kept, dropped)
}

/// C4. The orthonormal Haar basis vectors of length `n`, as the rows of a matrix: row `i` is the
/// signal whose coefficients are 1 at position `i` and 0 elsewhere (slide "The Haar basis vectors").
pub fn haar_basis(n: usize) -> Vec<Vec<f32>> {
    let mut rows = Vec::new();
    for i in 0..n {
        let mut e = vec![0.0; n];
        e[i] = 1.0;
        rows.push(haar_inverse(&e, true));
    }
    rows
}
