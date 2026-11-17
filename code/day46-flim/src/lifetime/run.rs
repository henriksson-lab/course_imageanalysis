// Simulates photon counting for single decays and for a synthetic FLIM image of nuclei, and
// estimates lifetimes with your code; pictures in out/lifetime/.
//   cargo run -p day46 -- lifetime                      (your code)
//   cargo run -p day46 --features solution -- lifetime  (the model solution)

use imgcourse::*;
use crate::lifetime::*;

/// Lifetime as colour (blue = 0.5 ns, green ≈ 2 ns, orange = 4 ns and longer),
/// brightness = intensity (0 to 1). Masked pixels (lifetime 0) are black.
fn lifetime_colour(tau: f32, brightness: f32) -> [f32; 3] {
    if tau <= 0.0 {
        return [0.0; 3];
    }
    let stops = [[0.15, 0.35, 1.0], [0.1, 0.85, 0.35], [1.0, 0.55, 0.05]];
    let t = ((tau - 0.5) / 3.5).clamp(0.0, 1.0) * 2.0;
    let (i, f) = ((t as usize).min(1), t - (t as usize).min(1) as f32);
    let b = brightness.clamp(0.0, 1.0);
    [0, 1, 2].map(|k| b * (stops[i][k] * (1.0 - f) + stops[i + 1][k] * f))
}

pub fn main() {
    let dir = "out/lifetime";
    std::fs::create_dir_all(dir).unwrap();

    // 1. One pixel, many photons: the two quick estimates (lecture demo "photons into a histogram").
    let setup = Setup { period: 12.5, bins: 125, irf_delay: 1.0, irf_sigma: 0.15 }; // bins of 0.1 ns
    let mut rng = Rng::new(1);
    let irf_mean = mean_arrival(&simulate(0.0, 100_000, 0.0, &setup, &mut rng), setup.period);
    println!("IRF mean arrival: {irf_mean:.2} ns");
    println!("true τ   background   mean arrival − IRF   two gates (1.5–3.5, 3.5–5.5 ns)");
    for (tau, bg) in [(2.5, 0.0), (8.0, 0.0), (2.5, 0.15)] {
        let counts = simulate(tau, 100_000, bg, &setup, &mut rng);
        let ma = mean_arrival(&counts, setup.period) - irf_mean;
        let tg = two_gates(&counts, setup.period, 1.5, 2.0);
        println!("{tau:5.1} ns   {:5.0} %       {ma:6.2} ns            {tg:6.2} ns", 100.0 * bg);
    }

    // 2. A FLIM image: synthetic nuclei; odd-numbered nuclei hold a 2.5 ns dye, even-numbered
    //    ones the same dye quenched to 1.2 ns. Every pixel also gets 8 photons of flat background.
    let (img, labels) = samples::nuclei(128, 14, 7);
    let mut stack = Volume::new(128, 128, DEMO.bins, 0.0f32);
    let mut true_tau = Image::new(128, 128, 2.5f32);
    for (x, y) in img.coords() {
        let fluorescence = (300.0 * (img[(x, y)] - 0.1)).max(0.0) as usize;
        let tau = if labels[(x, y)] % 2 == 0 && labels[(x, y)] > 0 { 1.2 } else { 2.5 };
        true_tau[(x, y)] = tau;
        let n = fluorescence + 8;
        let counts = simulate(tau, n, 8.0 / n as f32, &DEMO, &mut rng);
        for (j, c) in counts.iter().enumerate() {
            stack[(x, y, j)] = *c;
        }
    }
    let photons = Image::from_fn(128, 128, |x, y| (0..DEMO.bins).map(|j| stack[(x, y, j)]).sum::<f32>());
    save_gray_stretched(&photons, format!("{dir}/intensity.png"));
    let irf_mean = DEMO.irf_delay; // a narrow Gaussian IRF: its mean is its delay
    let bright = photons.map(|p| p / 250.0);

    let maps = [("lifetime-1x1", 0, 0.0), ("lifetime-1x1-masked", 0, 50.0), ("lifetime-3x3-masked", 1, 9.0 * 50.0)];
    println!("\nFLIM image, 128 × 128, 64 bins: inside the nuclei");
    println!("map                     2.5 ns nuclei: mean ± sd      1.2 ns nuclei: mean ± sd");
    for (name, radius, min_photons) in maps {
        let tau = lifetime_image(&stack, DEMO.period, irf_mean, radius, min_photons);
        let colour = Image::from_fn(128, 128, |x, y| lifetime_colour(tau[(x, y)], bright[(x, y)].max(0.25)));
        save_rgb(&colour, format!("{dir}/{name}.png"));
        let stats = |want: f32| {
            let v: Vec<f32> = tau.coords().filter(|&(x, y)| labels[(x, y)] > 0 && true_tau[(x, y)] == want && tau[(x, y)] > 0.0).map(|(x, y)| tau[(x, y)]).collect();
            let m = v.iter().sum::<f32>() / v.len() as f32;
            (m, (v.iter().map(|t| (t - m) * (t - m)).sum::<f32>() / v.len() as f32).sqrt())
        };
        let (a, b) = (stats(2.5), stats(1.2));
        println!("{name:22}  {:5.2} ± {:4.2} ns              {:5.2} ± {:4.2} ns", a.0, a.1, b.0, b.1);
    }
    let raw = lifetime_image(&stack, DEMO.period, irf_mean, 0, 0.0);
    println!("a background pixel (2, 2): {} photons, \"lifetime\" {:.2} ns", photons[(2, 2)], raw[(2, 2)]);
    println!("wrote the pictures to {dir}/");
}
