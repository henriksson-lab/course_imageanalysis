// Runs your filters on the course micrograph and writes the results to out/kernels/.
//   cargo run -p images -- kernels                      (your code)
//   cargo run -p images --features solution -- kernels  (the model solution)

use imgcourse::*;
use crate::kernels::*;

pub fn main() {
    let img = samples::cell();
    let dir = "out/kernels";
    save_gray(&img, format!("{dir}/input.png"));

    save_gray(&filter3x3(&img, &GAUSSIAN, Border::Mirror), format!("{dir}/gaussian.png"));
    save_gray(&filter3x3(&img, &SHARPEN, Border::Mirror), format!("{dir}/sharpen.png"));
    // Edge filters give negative values: show 0 as mid-grey.
    save_gray_signed(&filter3x3(&img, &SOBEL_X, Border::Mirror), format!("{dir}/sobel-x.png"));
    save_gray_signed(&filter3x3(&img, &LAPLACIAN, Border::Mirror), format!("{dir}/laplacian.png"));
    save_gray_stretched(&gradient_magnitude(&img, Border::Mirror), format!("{dir}/gradient.png"));

    // Noise and edges: the Laplacian of a noisy image, without and with blurring first.
    let noisy = samples::add_noise(&img, 30.0 / 255.0, 1);
    save_gray(&noisy, format!("{dir}/noisy.png"));
    save_gray_signed(&filter3x3(&noisy, &LAPLACIAN, Border::Mirror), format!("{dir}/noisy-laplacian.png"));
    let blurred = filter3x3(&filter3x3(&noisy, &GAUSSIAN, Border::Mirror), &GAUSSIAN, Border::Mirror);
    save_gray_signed(&filter3x3(&blurred, &LAPLACIAN, Border::Mirror), format!("{dir}/noisy-blur-laplacian.png"));

    // The border: blur 20 times with zeros outside, and with the edge copied.
    for (name, border) in [("zero", Border::Zero), ("clamp", Border::Clamp)] {
        let mut b = img.clone();
        for _ in 0..20 {
            b = filter_separable(&b, [1. / 4., 2. / 4., 1. / 4.], [1. / 4., 2. / 4., 1. / 4.], border);
        }
        save_gray(&b, format!("{dir}/blur20-{name}.png"));
    }
    println!("wrote the results to {dir}/");
}
