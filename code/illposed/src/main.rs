// Runs your inverse filter on the blurred course micrograph, without and with a little noise,
// and shows an invisible object. Results in out/illposed/.
//   cargo run -p illposed                      (your code)
//   cargo run -p illposed --features solution  (the model solution)

use illposed::*;
use imgcourse::*;

fn gaussian_psf(sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as usize;
    let g = Image::from_fn(2 * r + 1, 2 * r + 1, |x, y| {
        let (dx, dy) = (x as f32 - r as f32, y as f32 - r as f32);
        (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
    });
    let total = g.sum();
    g.map(|v| v / total)
}

fn rmse(a: &Image<f32>, b: &Image<f32>) -> f32 {
    let s: f32 = a.pixels().iter().zip(b.pixels()).map(|(p, q)| (p - q).powi(2)).sum();
    (s / a.pixels().len() as f32).sqrt()
}

fn main() {
    let dir = "out/illposed";
    let f = samples::cell();
    let (w, h) = (f.width(), f.height());

    // The small examples of the exercises.
    println!("(1, 2, 1)/4 on 4 pixels: gains {:?}, condition number {}", transfer(&[0.25, 0.5, 0.25], 4), condition_number(&[0.25, 0.5, 0.25], 4));
    println!("(0.2, 0.6, 0.2) on 4 pixels: gains {:?}, condition number {}", transfer(&[0.2, 0.6, 0.2], 4), condition_number(&[0.2, 0.6, 0.2], 4));

    // Gaussian blurs of the micrograph, then noise of a given standard deviation (in grey
    // values, 1 = white), then the inverse filter.
    println!("\nthe micrograph, blurred, noise added, inverse filtered (RMSE against the object):");
    println!("  blur σ   largest gain   noise     observed   restored");
    for sigma in [0.5, 1.0] {
        let otf = otf2(&gaussian_psf(sigma), w, h);
        let gain = otf.pixels().iter().map(|z| 1.0 / z.abs()).fold(0.0f32, f32::max);
        let g = blur_periodic(&f, &otf);
        for noise in [0.0, 0.0001, 0.001, 0.01] {
            let observed = samples::add_noise(&g, noise, 1);
            let restored = inverse_filter(&observed, &otf);
            println!("  {sigma:5}   {gain:12.0}   {noise:6}   {:8.4}   {:8.4}", rmse(&observed, &f), rmse(&restored, &f));
            if sigma == 1.0 && (noise == 0.0 || noise == 0.001) {
                let name = if noise == 0.0 { "no-noise" } else { "noise-0.001" };
                save_gray(&observed, format!("{dir}/observed-{name}.png"));
                save_gray(&restored, format!("{dir}/restored-{name}.png"));
                save_gray_signed(&restored.zip_map(&f, |a, b| a - b), format!("{dir}/error-{name}.png"));
            }
        }
    }

    // An invisible object: the blur (1, 2, 1)/4 along x and y kills the checkerboard.
    let psf = Image::from_rows(&[[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]).map(|v| v / 16.0);
    let otf = otf2(&psf, w, h);
    let f2 = f.map_xy(|x, y, v| v + if (x + y) % 2 == 0 { 0.1 } else { -0.1 });
    let (g, g2) = (blur_periodic(&f, &otf), blur_periodic(&f2, &otf));
    let diff = g.zip_map(&g2, |a, b| (a - b).abs()).max();
    println!("\nobject + checkerboard (±0.1): the largest difference between the two images is {diff:.1e}");
    save_gray(&f2, format!("{dir}/object-with-checkerboard.png"));
    save_gray(&g2, format!("{dir}/image-with-checkerboard.png"));
    println!("\nwrote the pictures to {dir}/");
}
