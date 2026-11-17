// Trains the smallest GAN (a line generator against a quadratic logistic discriminator) on the
// data N(1, 0.5²), with the non-saturating and the original (minimax) generator loss. Prints the
// path of the generator's (a, b) and the discriminator's loss; pictures in out/gan/.
//   cargo run -p gans -- gan                      (your code)
//   cargo run -p gans --features solution -- gan  (the model solution)

use crate::common::{normal_pdf, plot, BLUE, GREY, RED};
use crate::gan::*;
use imgcourse::*;

pub fn main() {
    let dir = "out/gan";
    // the by-hand numbers of exercise B3
    let c0 = [0.0; 3];
    let g = grad_d(&c0, &[1.0], &[-1.0]);
    let c1 = [-0.5 * g[0], -0.5 * g[1], -0.5 * g[2]];
    println!("B3: gradient of the discriminator's loss at c = 0: ({:.3}, {:.3}, {:.3}); after one step (rate 0.5): c = ({:.3}, {:.3}, {:.3})", g[0], g[1], g[2], c1[0], c1[1], c1[2]);
    println!("    D(1) = {:.3}, D(-1) = {:.3}, loss {:.3} -> {:.3}", discriminator(&c1, 1.0), discriminator(&c1, -1.0), loss_d(&c0, &[1.0], &[-1.0]), loss_d(&c1, &[1.0], &[-1.0]));
    let gan = LineGan { a: 1.0, b: -1.0, c: c1 };
    for loss in [GenLoss::NonSaturating, GenLoss::Minimax] {
        let (ga, gb) = grad_g(&gan, &[0.0], loss);
        println!("    generator gradient at z = 0 ({loss:?}): d/da {ga:.3}, d/db {gb:.3}");
    }

    let data = Mixture::one_bump();
    for loss in [GenLoss::NonSaturating, GenLoss::Minimax] {
        let s = Settings { lr_g: 0.03, lr_d: 0.5, k_d: 5, batch: 128, loss, optimal: false };
        let mut gan = LineGan::start();
        let mut rng = Rng::new(1);
        println!("\n{loss:?} loss, learning rates G {} and D {}, {} D steps per G step, batches of {}:", s.lr_g, s.lr_d, s.k_d, s.batch);
        println!("  step      a       b    D loss   D(x=1)");
        let mut pics = Vec::new();
        for step in 0..=2000 {
            if [0, 25, 50, 100, 200, 500, 1000, 2000].contains(&step) {
                let mut r2 = Rng::new(99);
                let real: Vec<f64> = (0..2000).map(|_| data.sample(&mut r2)).collect();
                let fake = gan.samples(2000, &mut r2);
                println!("  {step:4}  {:6.3}  {:6.3}  {:6.3}  {:6.3}", gan.a.abs(), gan.b, loss_d(&gan.c, &real, &fake), discriminator(&gan.c, 1.0));
                if [0, 50, 200, 2000].contains(&step) {
                    let xs: Vec<f64> = (0..=300).map(|i| -3.0 + 7.0 * i as f64 / 300.0).collect();
                    let dens = xs.iter().map(|&x| (x, data.pdf(x))).collect();
                    let gen = xs.iter().map(|&x| (x, normal_pdf(x, gan.b, gan.a.abs()).min(1.6))).collect();
                    let disc = xs.iter().map(|&x| (x, discriminator(&gan.c, x))).collect();
                    pics.push((step, plot(&[(dens, GREY), (gen, BLUE), (disc, RED)], (-3.0, 4.0), (-0.05, 1.6), 350, 160)));
                }
            }
            if step < 2000 {
                train_step(&mut gan, &data, &s, &mut rng);
            }
        }
        for (step, img) in pics {
            save_rgb(&img, format!("{dir}/{loss:?}-step{step}.png").to_lowercase());
        }
    }
    println!("\nThe pictures (grey: data, blue: generator, red: D(x)) are in {dir}/");
}
