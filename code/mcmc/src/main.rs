// Samples large Ising lattices with your samplers: three temperatures, the phase transition,
// critical slowing down, and quench against slow cooling. Pictures in out/mcmc/.
//   cargo run -p mcmc                      (your code)
//   cargo run -p mcmc --features solution  (the model solution)

use imgcourse::*;
use mcmc::*;

/// Spins as a picture, orange +1 and blue −1, each spin `zoom × zoom` pixels.
fn save_spins(s: &Spins, zoom: usize, path: String) {
    let img = Image::from_fn(s.width() * zoom, s.height() * zoom, |x, y| if s[(x / zoom, y / zoom)] > 0 { [0.93, 0.55, 0.13] } else { [0.20, 0.42, 0.80] });
    save_rgb(&img, path);
}

fn main() {
    let dir = "out/mcmc";
    let (j, size) = (1.0, 128);

    // Samples at three temperatures: 300 Metropolis sweeps from random spins.
    println!("{size} × {size}, 300 Metropolis sweeps from random spins:");
    for (name, t) in [("cold", 1.5), ("critical", 2.27), ("hot", 5.0)] {
        let mut rng = Rng::new(1);
        let mut s = random_spins(size, size, &mut rng);
        for _ in 0..300 {
            metropolis_sweep(&mut s, j, 0.0, t, &mut rng);
        }
        let (e, m) = energy_and_magnetisation(&s, j, 0.0);
        println!("  T = {t:4}: E/n = {e:6.3}, m = {m:6.3}");
        save_spins(&s, 2, format!("{dir}/sample-{name}.png"));
    }

    // Gibbs and Metropolis on the same lattice: how often does a pixel change per sweep?
    let mut rng = Rng::new(2);
    let mut a = random_spins(64, 64, &mut rng);
    let mut b = a.clone();
    let (mut changed_gibbs, mut changed_metro) = (0, 0);
    for _ in 0..200 {
        let before = a.clone();
        gibbs_sweep(&mut a, j, 0.0, 2.0, &mut rng);
        changed_gibbs += a.pixels().iter().zip(before.pixels()).filter(|(p, q)| p != q).count();
        changed_metro += metropolis_sweep(&mut b, j, 0.0, 2.0, &mut rng);
    }
    let n = 200.0 * 4096.0;
    println!("\nT = 2, 64 × 64, 200 sweeps: pixels changed per sweep: Gibbs {:.3}, Metropolis {:.3}", changed_gibbs as f32 / n, changed_metro as f32 / n);

    // The phase transition: average |m| over 300 sweeps on 64 × 64, starting from all +1
    // (from random spins a cold lattice can stay stuck in stripes for a long time).
    println!("\nT-scan, 64 × 64 (average |m| over 300 sweeps after 200 sweeps of burn-in, from all +1):");
    for k in 0..=8 {
        let t = 1.6 + 0.2 * k as f32;
        let mut rng = Rng::new(10 + k as u64);
        let mut s = Image::new(64, 64, 1i8);
        let mut sum = 0.0;
        for sweep in 0..500 {
            metropolis_sweep(&mut s, j, 0.0, t, &mut rng);
            if sweep >= 200 {
                sum += energy_and_magnetisation(&s, j, 0.0).1.abs();
            }
        }
        let onsager = if t < 2.269 { (1.0 - (2.0 * j / t).sinh().powi(-4)).powf(0.125) } else { 0.0 };
        println!("  T = {t:.2}: <|m|> = {:.3}   (Onsager, infinite lattice: {onsager:.3})", sum / 300.0);
    }

    // Critical slowing down: from all +1, sweeps until |m| < 0.1 (at most 4000).
    println!("\nfrom all +1 on 64 × 64, sweeps until |m| < 0.1:");
    for t in [4.0, 3.0, 2.5, 2.3] {
        let mut rng = Rng::new(5);
        let mut s = Image::new(64, 64, 1i8);
        let mut sweeps = 0;
        while sweeps < 4000 && energy_and_magnetisation(&s, j, 0.0).1.abs() >= 0.1 {
            metropolis_sweep(&mut s, j, 0.0, t, &mut rng);
            sweeps += 1;
        }
        println!("  T = {t}: {sweeps}{}", if sweeps == 4000 { " (not reached)" } else { "" });
    }

    // Quench against slow cooling, 64 × 64, 600 sweeps each from the same random spins, for
    // 8 different starts: how often does each end in a ground state (all spins equal)?
    println!("\n64 × 64, 600 sweeps: quench at T = 0.5, or cool from 3.5 to 0.5 over 500 sweeps:");
    let (mut ground_quench, mut ground_slow) = (0, 0);
    for seed in 1..=8u64 {
        let start = random_spins(64, 64, &mut Rng::new(seed));
        let mut quench = start.clone();
        let mut rng = Rng::new(100 + seed);
        for _ in 0..600 {
            metropolis_sweep(&mut quench, j, 0.0, 0.5, &mut rng);
        }
        let mut slow = start.clone();
        let mut rng = Rng::new(100 + seed);
        anneal(&mut slow, j, 3.5, 0.5, 500, &mut rng);
        for _ in 500..600 {
            metropolis_sweep(&mut slow, j, 0.0, 0.5, &mut rng);
        }
        let (eq, mq) = energy_and_magnetisation(&quench, j, 0.0);
        let (es, ms) = energy_and_magnetisation(&slow, j, 0.0);
        println!("  start {seed}: quench E/n = {eq:.4}, m = {mq:6.3}   slow cooling E/n = {es:.4}, m = {ms:6.3}");
        ground_quench += (mq.abs() > 0.99) as usize;
        ground_slow += (ms.abs() > 0.99) as usize;
        if seed == 1 {
            save_spins(&quench, 4, format!("{dir}/quench.png"));
            save_spins(&slow, 4, format!("{dir}/slow-cooling.png"));
        }
    }
    println!("  one domain (|m| > 0.99): quench {ground_quench} of 8, slow cooling {ground_slow} of 8");
    println!("\nwrote the lattices to {dir}/");
}
