// Image-to-image translation reduced to one pixel: what translators trained with the squared
// error, the absolute error, as a conditional GAN, or without pairs output, scored against
// held-out truth. 400 training and 400 test pairs per scenario, kernel bandwidth 0.04.
// Pictures in out/image-translation/.
//   cargo run -p gans -- image-translation                      (your code)
//   cargo run -p gans --features solution -- image-translation  (the model solution)

use crate::common::{plot, BLUE, GREEN, GREY, ORANGE};
use crate::image_translation::*;
use imgcourse::*;

const H: f64 = 0.04;

pub fn main() {
    let dir = "out/image-translation";
    println!("scenario                         RMSE: mean  median  sample  unpaired | realistic: mean  median  sample | marker-on share: truth  samples");
    for (name, gap, q_train, q_test) in [
        ("target visible (gap 0)", 0.0, 0.5, 0.5),
        ("hidden marker (gap 0.3)", 0.3, 0.5, 0.5),
        ("treated test cells", 0.3, 0.2, 0.8),
    ] {
        let mut rng = Rng::new(1);
        let train = pairs(400, gap, q_train, &mut rng);
        let test = pairs(400, gap, q_test, &mut rng);
        let xs: Vec<f64> = train.iter().map(|p| p.x).collect();
        let ys: Vec<f64> = train.iter().map(|p| p.y).collect();
        let truth: Vec<f64> = test.iter().map(|p| p.y).collect();
        let mean: Vec<f64> = test.iter().map(|p| cond_mean(&train, p.x, H)).collect();
        let median: Vec<f64> = test.iter().map(|p| cond_median(&train, p.x, H)).collect();
        let sample: Vec<f64> = test.iter().map(|p| cond_sample(&train, p.x, H, &mut rng)).collect();
        let unpaired: Vec<f64> = test.iter().map(|p| quantile_map(&xs, &ys, p.x, false)).collect();
        let real = |pred: &[f64]| test.iter().zip(pred).filter(|(p, &y)| is_real(p.x, y, gap)).count() as f64 / test.len() as f64;
        let high: Vec<usize> = (0..test.len()).filter(|&i| test[i].x > 0.5).collect();
        let on_truth = high.iter().filter(|&&i| test[i].on).count() as f64 / high.len() as f64;
        let on_sample = high.iter().filter(|&&i| sample[i] - base(test[i].x) > gap / 2.0).count() as f64 / high.len() as f64;
        println!(
            "{name:32} {:.3}   {:.3}   {:.3}   {:.3}    |   {:4.0} %  {:4.0} %  {:4.0} % |  {:4.0} %  {:4.0} %",
            rmse(&mean, &truth), rmse(&median, &truth), rmse(&sample, &truth), rmse(&unpaired, &truth),
            100.0 * real(&mean), 100.0 * real(&median), 100.0 * real(&sample), 100.0 * on_truth, 100.0 * on_sample
        );
        // the picture: training pairs (grey), mean (orange), median (green), samples (blue)
        let gx: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
        let mut img = plot(&[
            (gx.iter().map(|&x| (x, cond_mean(&train, x, H))).collect(), ORANGE),
            (gx.iter().map(|&x| (x, cond_median(&train, x, H))).collect(), GREEN),
        ], (0.0, 1.0), (0.0, 1.1), 400, 400);
        let mut dot = |x: f64, y: f64, c: [f32; 3]| {
            let (px, py) = ((x * 399.0) as isize, ((1.1 - y) / 1.1 * 399.0) as isize);
            for d in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                if img.contains(px + d.0, py + d.1) {
                    img[((px + d.0) as usize, (py + d.1) as usize)] = c;
                }
            }
        };
        for p in &train {
            dot(p.x, p.y, GREY);
        }
        for (p, &y) in test.iter().zip(&sample) {
            dot(p.x, y, BLUE);
        }
        save_rgb(&img, format!("{dir}/{}.png", name.split(' ').next().unwrap()));
    }
    println!("\nThe pictures (grey: training pairs, orange: mean, green: median, blue: samples) are in {dir}/");
}
