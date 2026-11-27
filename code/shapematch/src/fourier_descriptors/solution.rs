// Model solution for "Outlines as periodic signals: Fourier descriptors".

use super::given::{coef, freq, Match};
use imgcourse::fft::fft;
use imgcourse::Complex;

/// C1. The Fourier descriptors of an outline `z_n = x_n + i·y_n`:
/// `c_k = (1/N) Σ_n z_n e^{−2πi k n/N}`. `out[j]` is `c_k` with `k = freq(j, N)`.
pub fn descriptors(z: &[Complex]) -> Vec<Complex> {
    let n = z.len() as f32;
    fft(z, false).into_iter().map(|c| c / n).collect()
}

/// C2. The outline rebuilt from the descriptors with `|k| ≤ k_max` only (the others set to 0):
/// `z_n = Σ_{|k| ≤ k_max} c_k e^{2πi k n/N}`. With `k_max ≥ N/2` it gives back every sample.
pub fn reconstruct(c: &[Complex], k_max: usize) -> Vec<Complex> {
    let n = c.len();
    let kept: Vec<Complex> = (0..n)
        .map(|j| if freq(j, n).unsigned_abs() <= k_max { c[j] } else { Complex::ZERO })
        .collect();
    // the inverse FFT divides by N; the descriptors already carry the 1/N
    fft(&kept, true).into_iter().map(|v| v * n as f32).collect()
}

/// C3. `n` points equally spaced in arc length along the closed polygon `poly`, starting at
/// vertex 0: walk along the edges and drop a point every `L/n`.
pub fn resample_arc_length(poly: &[Complex], n: usize) -> Vec<Complex> {
    let m = poly.len();
    let mut cum = vec![0.0f32; m + 1];
    for i in 0..m {
        cum[i + 1] = cum[i] + (poly[(i + 1) % m] - poly[i]).abs();
    }
    let total = cum[m];
    let mut out = Vec::with_capacity(n);
    let mut seg = 0;
    for i in 0..n {
        let t = total * i as f32 / n as f32;
        while seg + 1 < m && cum[seg + 1] <= t {
            seg += 1;
        }
        let len = (cum[seg + 1] - cum[seg]).max(1e-12);
        let f = (t - cum[seg]) / len;
        out.push(poly[seg] * (1.0 - f) + poly[(seg + 1) % m] * f);
    }
    out
}

/// C4. The magnitude signature `|c_k| / |c_1|` for `k = −k_max … −1, 1 … k_max` (in that
/// order): unchanged by translation, scale, rotation and starting point.
pub fn magnitude_signature(c: &[Complex], k_max: usize) -> Vec<f32> {
    let m1 = coef(c, 1).abs();
    let k = k_max as isize;
    (-k..=k).filter(|&k| k != 0).map(|k| coef(c, k).abs() / m1).collect()
}

/// C5. Match outline B to outline A (same number of samples) over rotation, scale, translation
/// and starting point. For every start `m` at once, with one FFT:
/// `q(m) = Σ_{k≠0} conj(c^A_k)·c^B_k·e^{−2πi k m/N}`; the best `m` has the largest `|q(m)|`,
/// and there `λ = q(m) / Σ_{k≠0}|c^A_k|²`. With `allow_mirror`, also try the mirror image of A
/// (its descriptors are `conj(c^A_k)`) and keep the better fit.
pub fn match_outlines(a: &[Complex], b: &[Complex], allow_mirror: bool) -> Match {
    let ca = descriptors(a);
    let cb = descriptors(b);
    let fit = |ca: &[Complex], mirrored: bool| -> Match {
        let n = ca.len();
        let mut p = vec![Complex::ZERO; n];
        let (mut ea, mut eb) = (0.0f32, 0.0f32);
        for j in 1..n {
            p[j] = ca[j].conj() * cb[j];
            ea += ca[j].norm_sqr();
            eb += cb[j].norm_sqr();
        }
        let q = fft(&p, false); // forward: Σ_k P_k e^{−2πi k m/N}
        let m = (0..n).max_by(|&i, &j| q[i].abs().total_cmp(&q[j].abs())).unwrap();
        let lambda = q[m] / ea;
        Match { m, lambda, t: cb[0] - lambda * ca[0], score: q[m].abs() / (ea * eb).sqrt(), mirrored }
    };
    let plain = fit(&ca, false);
    if allow_mirror {
        let cm: Vec<Complex> = ca.iter().map(|c| c.conj()).collect();
        let mirror = fit(&cm, true);
        if mirror.score > plain.score {
            return mirror;
        }
    }
    plain
}
