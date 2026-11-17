//! Given: the gradient of the edges unit (canny/), and the Gaussian blur of `imgcourse::filters`.
//! Not an exercise here.

pub use imgcourse::filters::gaussian_blur;
use imgcourse::filters::{filter_1d, gaussian_kernel};
use imgcourse::Image;

/// The derivative of a Gaussian as a 1D kernel, scaled so that a ramp of slope 1 gives 1.
pub fn gaussian_derivative_kernel(sigma: f32) -> Vec<f32> {
    let r = (4.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|x| x as f32 * (-((x * x) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let slope: f32 = (-r..=r).zip(&w).map(|(x, v)| x as f32 * v).sum();
    w.iter().map(|v| v / slope).collect()
}

/// The gradient (Lx, Ly) of the image smoothed at scale `sigma` (y down).
pub fn gradient(img: &Image<f32>, sigma: f32) -> (Image<f32>, Image<f32>) {
    let (g, dg) = (gaussian_kernel(sigma), gaussian_derivative_kernel(sigma));
    let lx = filter_1d(&filter_1d(img, &dg, true), &g, false);
    let ly = filter_1d(&filter_1d(img, &g, true), &dg, false);
    (lx, ly)
}
