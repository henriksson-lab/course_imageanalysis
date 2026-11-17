// Model solution for the coding exercises of "Stain separation: colour deconvolution".

use imgcourse::{linalg, Image};

/// A colour image of **linear** intensities: `[red, green, blue]` per pixel, in [0, 1].
pub type Rgb = Image<[f32; 3]>;
/// A 3 × 3 matrix, row by row: `m[row][column]`. A stain matrix has the stains' unit OD
/// vectors (red, green, blue) as its columns.
pub type Mat3 = [[f32; 3]; 3];

/// Published stain vectors (Ruifrok & Johnston), (red, green, blue), not yet of length 1.
pub const HEMATOXYLIN: [f32; 3] = [0.65, 0.70, 0.29];
pub const EOSIN: [f32; 3] = [0.07, 0.99, 0.11];
pub const DAB: [f32; 3] = [0.27, 0.57, 0.78];

/// C1. The optical density of every pixel, `OD_c = −log10(I_c / I0_c)` per channel. An
/// intensity below 1/255 is first raised to 1/255 (a black pixel has no finite OD).
pub fn optical_density(img: &Rgb, i0: [f32; 3]) -> Rgb {
    img.map(|p| {
        let mut od = [0.0f32; 3];
        for c in 0..3 {
            od[c] = -(p[c].max(1.0 / 255.0) / i0[c]).log10();
        }
        od
    })
}

/// C2. Beer–Lambert: the transmitted intensities of a pixel with stain amounts `amounts`
/// (one per column of `s`): `I_c = I0_c · 10^(−Σ_k s[c][k] · amounts[k])`.
pub fn transmit(amounts: [f32; 3], s: &Mat3, i0: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let od: f32 = (0..3).map(|k| s[c][k] * amounts[k]).sum();
        out[c] = i0[c] * 10f32.powf(-od);
    }
    out
}

/// C3a. The stain matrix of two stains (slide "The stain matrix and colour deconvolution"):
/// columns = the two vectors scaled to length 1, and their cross product, scaled to length 1
/// (the residual direction).
pub fn stain_matrix(s1: [f32; 3], s2: [f32; 3]) -> Mat3 {
    let unit = |v: [f32; 3]| {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        v.map(|x| x / n)
    };
    let (a, b) = (unit(s1), unit(s2));
    let r = unit([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]);
    let mut s = [[0.0; 3]; 3];
    for c in 0..3 {
        s[c] = [a[c], b[c], r[c]];
    }
    s
}

/// C3b. Colour deconvolution: the three concentration images `C = S⁻¹ OD`, pixel by pixel.
pub fn deconvolve(od: &Rgb, s: &Mat3) -> [Image<f32>; 3] {
    let a: Vec<Vec<f64>> = s.iter().map(|row| row.iter().map(|&v| v as f64).collect()).collect();
    let mut inv = [[0.0f32; 3]; 3];
    for j in 0..3 {
        let e: Vec<f64> = (0..3).map(|i| if i == j { 1.0 } else { 0.0 }).collect();
        let col = linalg::solve(&a, &e).expect("the stain vectors must be independent");
        for i in 0..3 {
            inv[i][j] = col[i] as f32;
        }
    }
    [0, 1, 2].map(|k| od.map(|p| inv[k][0] * p[0] + inv[k][1] * p[1] + inv[k][2] * p[2]))
}

/// C4. Macenko's estimate of two stain vectors from the OD of the pixels
/// (slide "The math: Macenko in formulas"):
/// 1. keep the pixels with |od| > `beta`;
/// 2. v1, v2 = the two leading eigenvectors of Σ od odᵀ (no centring);
/// 3. every kept pixel's angle φ = atan2(od · v2, od · v1);
/// 4. the stain vectors are cos φ v1 + sin φ v2 at the `alpha` and the `100 − alpha`
///    percentiles of the angles.
///
/// Returns the two unit vectors, the one with the larger red OD (hematoxylin) first. (An
/// eigenvector may come out pointing either way: flip v1 if its entries sum to less than 0,
/// so that it points into the cloud of positive ODs.)
pub fn macenko(od: &[[f32; 3]], beta: f32, alpha: f32) -> [[f32; 3]; 2] {
    let kept: Vec<[f32; 3]> = od.iter().copied().filter(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() > beta).collect();
    let mut s = vec![vec![0.0f64; 3]; 3];
    for p in &kept {
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += p[i] as f64 * p[j] as f64;
            }
        }
    }
    let (_, u) = linalg::sym_eigen(&s);
    let mut v1 = [u[0][0] as f32, u[0][1] as f32, u[0][2] as f32];
    let v2 = [u[1][0] as f32, u[1][1] as f32, u[1][2] as f32];
    if v1[0] + v1[1] + v1[2] < 0.0 {
        v1 = v1.map(|x| -x); // point into the cloud (OD is positive)
    }
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut angles: Vec<f32> = kept.iter().map(|&p| dot(p, v2).atan2(dot(p, v1))).collect();
    angles.sort_by(f32::total_cmp);
    let at = |pct: f32| angles[(pct / 100.0 * (angles.len() - 1) as f32).round() as usize];
    let vec_at = |phi: f32| {
        let v = [0, 1, 2].map(|c| phi.cos() * v1[c] + phi.sin() * v2[c]);
        let n = dot(v, v).sqrt();
        v.map(|x| x / n)
    };
    let (a, b) = (vec_at(at(alpha)), vec_at(at(100.0 - alpha)));
    if a[0] >= b[0] { [a, b] } else { [b, a] }
}
