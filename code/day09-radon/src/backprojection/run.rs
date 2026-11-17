// Back-projection of a point (the star and the halo) and of the Shepp–Logan phantom, and the
// Fourier slice theorem checked on the phantom. Pictures go to out/backprojection/.
//   cargo run -p day09 -- backprojection                      (your code)
//   cargo run -p day09 --features solution -- backprojection  (the model solution)

use crate::common::{angles, centre, projection, shepp_logan, sinogram};
use crate::backprojection::*;
use imgcourse::*;

pub fn main() {
    let dir = "out/backprojection";
    let n = 129; // odd, so that the centre is a pixel
    let c = centre(n) as usize;

    // 1. A point at (20, 10) pixels from the centre, back-projected from 1, 4, 16 and 180 angles.
    let mut point = Image::new(n, n, 0.0f32);
    point[(c + 20, c - 10)] = 1.0;
    for count in [1, 4, 16, 180] {
        let thetas = angles(count);
        let b = backproject(&sinogram(&point, &thetas), &thetas, n);
        save_gray_stretched(&b, format!("{dir}/point-{count}-angles.png"));
    }

    // 2. The halo of a point at the centre (180 angles), averaged over rings of radius r.
    let mut centred = Image::new(n, n, 0.0f32);
    centred[(c, c)] = 1.0;
    let thetas = angles(180);
    let halo = backproject(&sinogram(&centred, &thetas), &thetas, n);
    print!("halo of a point (180 angles), ring averages:");
    for r in [2.0f32, 5.0, 10.0, 20.0, 40.0] {
        let ring: Vec<f32> = halo.coords().filter(|&(x, y)| (((x as f32 - c as f32).powi(2) + (y as f32 - c as f32).powi(2)).sqrt() - r).abs() < 0.5).map(|p| halo[p]).collect();
        let mean = ring.iter().sum::<f32>() / ring.len() as f32;
        print!("  r = {r}: {mean:.4} (r · value = {:.3})", r * mean);
    }
    println!();

    // 3. Plain back-projection of the phantom: the right shape, but everything glows.
    let phantom = shepp_logan(n);
    save_gray(&phantom, format!("{dir}/phantom.png"));
    let b = backproject(&sinogram(&phantom, &thetas), &thetas, n);
    save_gray_stretched(&b, format!("{dir}/phantom-backprojected.png"));
    println!("phantom: true values 0 to {:.1}; back-projected values {:.1} to {:.1}", phantom.max(), b.min(), b.max());

    // 4. The Fourier slice theorem on the phantom: projection spectrum against the slice.
    let spectrum = fft::fft2_real(&shepp_logan(128));
    for deg in [0.0f32, 30.0, 90.0] {
        let p = projection_spectrum(&projection(&shepp_logan(128), deg.to_radians()));
        let s = spectrum_slice(&spectrum, deg.to_radians());
        // relative difference over all frequencies, and over the low ones (|ω| ≤ 16)
        let rel = |range: std::ops::Range<usize>| {
            let diff: f32 = range.clone().map(|i| (p[i] - s[i]).abs()).sum();
            100.0 * diff / range.map(|i| s[i]).sum::<f32>()
        };
        println!("Fourier slice at {deg:>2}°: projection spectrum and 2D slice differ by {:.2} % (all ω), {:.2} % (|ω| ≤ 16)", rel(0..128), rel(48..81));
    }
    println!("wrote pictures to {dir}/");
}
