// Model solution for the coding exercises of "Filtered back-projection and sampling".
// Coordinates as in the lecture: origin at the image centre, y up, θ counter-clockwise
// (see `tomo.rs`: `projection`, `sinogram`, `backproject`, `angles`).

use crate::fbp::tomo::backproject;
use imgcourse::Image;
use std::f32::consts::PI;

/// C1a. The ramp filter as a kernel, the Ram–Lak kernel (slide "By hand: what the filter does
/// to a box"): h[0] = 1/4, h[k] = −1/(π k)² for odd k, 0 for even k ≠ 0.
pub fn ram_lak(k: isize) -> f32 {
    if k == 0 {
        0.25
    } else if k % 2 != 0 {
        -1.0 / (PI * k as f32).powi(2)
    } else {
        0.0
    }
}

/// C1b. The Ram–Lak kernel for k = −K, ..., K as a list of 2K + 1 weights (centre in the middle).
pub fn ramp_kernel(k_max: usize) -> Vec<f32> {
    let k_max = k_max as isize;
    (-k_max..=k_max).map(ram_lak).collect()
}

/// C2. Filter a projection with a kernel of odd length 2K + 1 (centre in the middle):
/// out[i] = Σ_j h[j] p[i + j − K], with zeros outside the projection (that is the zero padding).
pub fn filter(p: &[f32], h: &[f32]) -> Vec<f32> {
    let k = (h.len() / 2) as isize;
    let mut out = vec![0.0; p.len()];
    for i in 0..p.len() {
        let mut sum = 0.0;
        for (j, &w) in h.iter().enumerate() {
            let idx = i as isize + j as isize - k;
            if idx >= 0 && (idx as usize) < p.len() {
                sum += w * p[idx as usize];
            }
        }
        out[i] = sum;
    }
    out
}

/// C3. Filtered back-projection: filter every projection (every column of the sinogram) with
/// the kernel `h`, then back-project the filtered sinogram onto an n × n image.
pub fn fbp(sino: &Image<f32>, thetas: &[f32], n: usize, h: &[f32]) -> Image<f32> {
    let mut filtered = Image::new(sino.width(), sino.height(), 0.0);
    for a in 0..sino.width() {
        let column: Vec<f32> = (0..sino.height()).map(|k| sino[(a, k)]).collect();
        for (k, v) in filter(&column, h).into_iter().enumerate() {
            filtered[(a, k)] = v;
        }
    }
    backproject(&filtered, thetas, n)
}

/// C4. A tapered ramp: the Ram–Lak kernel smoothed with the weights (1/4, 1/2, 1/4). In
/// frequency this multiplies the ramp by cos²(π f), the Hann taper with cutoff 1/2:
/// h_Hann[k] = h[k − 1]/4 + h[k]/2 + h[k + 1]/4, for k = −K, ..., K.
pub fn hann_kernel(k_max: usize) -> Vec<f32> {
    let k_max = k_max as isize;
    (-k_max..=k_max).map(|k| 0.25 * ram_lak(k - 1) + 0.5 * ram_lak(k) + 0.25 * ram_lak(k + 1)).collect()
}
