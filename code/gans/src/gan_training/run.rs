// What goes wrong when training GANs, on the smallest GAN of the unit gan (its complete version
// is in src/common.rs). Part 1: Jensen–Shannon divergence and Wasserstein distance between the
// data N(1, 0.5²) and a generator N(b, 0.5²) as b moves away. Part 2: five training runs, with
// precision and recall of the samples. Pictures in out/gan-training/.
//   cargo run -p gans -- gan-training                      (your code)
//   cargo run -p gans --features solution -- gan-training  (the model solution)

use crate::common::{normal_pdf, plot, BLUE, GREY, ORANGE};
use crate::gan_training::*;
use imgcourse::*;

pub fn main() {
    let dir = "out/gan-training";
    // Part 1: the two distances as the generator slides away from the data
    let dx = 0.01;
    let grid: Vec<f64> = (0..2000).map(|i| -10.0 + (i as f64 + 0.5) * dx).collect();
    let p: Vec<f64> = grid.iter().map(|&x| normal_pdf(x, 1.0, 0.5)).collect();
    println!("Data N(1, 0.5^2) against a generator N(b, 0.5^2):");
    println!("     b    distance   JS     W1 (samples)");
    let mut rng = Rng::new(3);
    let z: Vec<f64> = (0..4000).map(|_| rng.normal() as f64).collect();
    let real: Vec<f64> = z.iter().map(|v| 1.0 + 0.5 * v).collect();
    let mut js_curve = Vec::new();
    let mut w_curve = Vec::new();
    for i in 0..=16 {
        let b = 1.0 - 0.5 * i as f64;
        let q: Vec<f64> = grid.iter().map(|&x| normal_pdf(x, b, 0.5)).collect();
        let js = js_divergence(&p, &q, dx);
        let fake: Vec<f64> = (0..4000).map(|_| b + 0.5 * rng.normal() as f64).collect();
        let w = wasserstein1(&real, &fake);
        js_curve.push((1.0 - b, js));
        w_curve.push((1.0 - b, w));
        if i % 2 == 0 {
            println!("  {b:5.1}   {:5.1}   {js:.4}   {w:.3}", 1.0 - b);
        }
    }
    println!("  (ln 2 = {:.4})", std::f64::consts::LN_2);
    save_rgb(&plot(&[(js_curve, BLUE), (w_curve, ORANGE)], (0.0, 8.0), (-0.2, 8.0), 400, 200), format!("{dir}/js-vs-wasserstein.png"));

    // Part 2: training runs
    let runs: [(&str, Mixture, Settings); 5] = [
        ("balanced", Mixture::one_bump(), Settings { lr_g: 0.03, lr_d: 0.5, k_d: 5, batch: 128, loss: GenLoss::NonSaturating, optimal: false }),
        ("discriminator too slow", Mixture::one_bump(), Settings { lr_g: 0.03, lr_d: 0.05, k_d: 1, batch: 128, loss: GenLoss::NonSaturating, optimal: false }),
        ("generator too fast", Mixture::one_bump(), Settings { lr_g: 0.3, lr_d: 0.5, k_d: 1, batch: 128, loss: GenLoss::NonSaturating, optimal: false }),
        ("two bumps, best discriminator", Mixture::two_bumps(), Settings { lr_g: 0.03, lr_d: 0.5, k_d: 5, batch: 128, loss: GenLoss::NonSaturating, optimal: true }),
        ("two bumps, learned discriminator", Mixture::two_bumps(), Settings { lr_g: 0.03, lr_d: 0.5, k_d: 5, batch: 128, loss: GenLoss::NonSaturating, optimal: false }),
    ];
    for (name, data, s) in runs {
        let mut gan = LineGan::start();
        let mut rng = Rng::new(1);
        println!("\n{name}:  step: spread a, centre b, precision, recall (k = 3, 500 points each)");
        let mut line = String::new();
        for step in 0..=2000 {
            if [25, 50, 100, 200, 500, 2000].contains(&step) {
                let mut r2 = Rng::new(7);
                let real: Vec<f64> = (0..500).map(|_| data.sample(&mut r2)).collect();
                let fake = gan.samples(500, &mut r2);
                let (pr, rc) = precision_recall(&real, &fake, 3);
                line += &format!("\n  {step:4}: a {:.2}, b {:.2}, P {pr:.2}, R {rc:.2}", gan.a.abs(), gan.b);
            }
            if step < 2000 {
                gan.train_step(&data, &s, &mut rng);
            }
        }
        println!("{}", line.trim_start_matches('\n'));
        let xs: Vec<f64> = (0..=300).map(|i| -3.0 + 7.0 * i as f64 / 300.0).collect();
        let dens = xs.iter().map(|&x| (x, data.pdf(x))).collect();
        let gen = xs.iter().map(|&x| (x, normal_pdf(x, gan.b, gan.a.abs()).min(1.6))).collect();
        save_rgb(&plot(&[(dens, GREY), (gen, BLUE)], (-3.0, 4.0), (-0.05, 1.6), 350, 160), format!("{dir}/{}.png", name.replace([' ', ','], "-").replace("--", "-")));
    }
    println!("\nThe pictures are in {dir}/");
}
