// Checks for the coding exercises of "Generative adversarial networks", with the numbers of the
// pen-and-paper exercises. Run: cargo test -p gans gan::

use gans::gan::*;
use imgcourse::Rng;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_discriminator() {
    assert!(close(discriminator(&[0.0; 3], 3.0), 0.5, 1e-12), "D with c = 0 must be 1/2 everywhere");
    let d = discriminator(&[0.0, 0.5, 0.0], 1.0);
    assert!(close(d, 0.6225, 1e-4), "B3: D(1) with c = (0, 0.5, 0): got {d}, expected 0.6225");
    let d = discriminator(&[1.0, -1.0, 0.5], 2.0);
    assert!(close(d, 0.7311, 1e-4), "sigma(1 - 2 + 2) = sigma(1): got {d}, expected 0.7311");
}

#[test]
fn c2_optimal_discriminator_b1() {
    for (p, q, e) in [(0.4, 0.1, 0.8), (0.2, 0.2, 0.5), (0.0, 0.3, 0.0), (0.0, 0.0, 0.5)] {
        let d = optimal_discriminator(p, q);
        assert!(close(d, e, 1e-12), "B1: p_data = {p}, p_gen = {q}: got {d}, expected {e}");
    }
}

#[test]
fn c3_loss_and_gradient_b3() {
    let l = loss_d(&[0.0; 3], &[1.0], &[-1.0]);
    assert!(close(l, 2.0 * 2f64.ln(), 1e-9), "B3: the loss at c = 0 is 2 ln 2 = 1.386; got {l}");
    let g = grad_d(&[0.0; 3], &[1.0], &[-1.0]);
    for (j, e) in [0.0, -1.0, 0.0].iter().enumerate() {
        assert!(close(g[j], *e, 1e-9), "B3: gradient at c = 0 should be (0, -1, 0); got {g:?}");
    }
    let l = loss_d(&[0.0, 0.5, 0.0], &[1.0], &[-1.0]);
    assert!(close(l, 0.948, 1e-3), "B3: the loss after the step is 0.948; got {l}");
    // against finite differences at another point
    let c = [0.3, -0.4, 0.2];
    let (real, fake) = ([0.5, 1.2, 2.0], [-0.7, 0.1]);
    let g = grad_d(&c, &real, &fake);
    for j in 0..3 {
        let (mut cp, mut cm) = (c, c);
        cp[j] += 1e-6;
        cm[j] -= 1e-6;
        let num = (loss_d(&cp, &real, &fake) - loss_d(&cm, &real, &fake)) / 2e-6;
        assert!(close(g[j], num, 1e-5), "grad_d disagrees with finite differences in c{j}: {} against {num}", g[j]);
    }
}

#[test]
fn c4_generator_gradient_b4() {
    let gan = LineGan { a: 1.0, b: -1.0, c: [0.0, 0.5, 0.0] };
    let (ga, gb) = grad_g(&gan, &[0.0], GenLoss::NonSaturating);
    assert!(close(ga, 0.0, 1e-9) && close(gb, -0.3112, 1e-4), "B4, non-saturating at z = 0: got ({ga}, {gb}), expected (0, -0.311)");
    let (_, gb) = grad_g(&gan, &[0.0], GenLoss::Minimax);
    assert!(close(gb, -0.1888, 1e-4), "B4, minimax at z = 0: got {gb}, expected -0.189");
    let (ga, gb) = grad_g(&gan, &[1.0, -1.0], GenLoss::NonSaturating);
    assert!(close(ga, 0.0578, 1e-4) && close(gb, -0.3078, 1e-4), "B4c, z = 1 and z = -1: got ({ga}, {gb}), expected (0.058, -0.308)");
}

#[test]
fn c5_training_converges() {
    let data = Mixture::one_bump();
    let s = Settings { lr_g: 0.03, lr_d: 0.5, k_d: 5, batch: 128, loss: GenLoss::NonSaturating, optimal: false };
    let mut gan = LineGan::start();
    let mut rng = Rng::new(1);
    for _ in 0..1000 {
        train_step(&mut gan, &data, &s, &mut rng);
    }
    assert!(close(gan.a.abs(), 0.5, 0.1) && close(gan.b, 1.0, 0.1), "after 1000 steps the generator should be close to N(1, 0.5^2); got a = {}, b = {}", gan.a, gan.b);
}
