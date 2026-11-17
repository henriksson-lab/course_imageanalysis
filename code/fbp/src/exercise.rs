// Coding exercises for "Filtered back-projection and sampling". Replace each `todo!()` with
// your code, then check it with `cargo test -p fbp` and look at `cargo run -p fbp`.
// Coordinates as in the lecture: origin at the image centre, y up, θ counter-clockwise
// (see `tomo.rs`: `projection`, `sinogram`, `backproject`, `angles`).
#![allow(unused_variables, unused_imports)]

use crate::tomo::backproject;
use imgcourse::Image;
use std::f32::consts::PI;

/// C1a. The ramp filter as a kernel, the Ram–Lak kernel (slide "By hand: what the filter does
/// to a box"): h[0] = 1/4, h[k] = −1/(π k)² for odd k, 0 for even k ≠ 0.
///
/// Hint: `k % 2 != 0` is true for odd k, also negative ones.
pub fn ram_lak(k: isize) -> f32 {
    todo!("C1a: ram_lak")
}

/// C1b. The Ram–Lak kernel for k = −K, ..., K as a list of 2K + 1 weights (centre in the middle).
pub fn ramp_kernel(k_max: usize) -> Vec<f32> {
    todo!("C1b: ramp_kernel")
}

/// C2. Filter a projection with a kernel of odd length 2K + 1 (centre in the middle):
/// out[i] = Σ_j h[j] p[i + j − K], with zeros outside the projection (that is the zero padding).
///
/// Hint: use `isize` for the index `i + j − K`, and skip it when it falls outside `p`.
pub fn filter(p: &[f32], h: &[f32]) -> Vec<f32> {
    todo!("C2: filter")
}

/// C3. Filtered back-projection: filter every projection (every column of the sinogram) with
/// the kernel `h`, then back-project the filtered sinogram onto an n × n image.
///
/// Hint: collect column `a` of the sinogram into a `Vec`, filter it, write it back into a new
/// sinogram; then `backproject(&filtered, thetas, n)` from `tomo.rs`.
pub fn fbp(sino: &Image<f32>, thetas: &[f32], n: usize, h: &[f32]) -> Image<f32> {
    todo!("C3: fbp")
}

/// C4. A tapered ramp: the Ram–Lak kernel smoothed with the weights (1/4, 1/2, 1/4). In
/// frequency this multiplies the ramp by cos²(π f), the Hann taper with cutoff 1/2:
/// h_Hann[k] = h[k − 1]/4 + h[k]/2 + h[k + 1]/4, for k = −K, ..., K.
pub fn hann_kernel(k_max: usize) -> Vec<f32> {
    todo!("C4: hann_kernel")
}
