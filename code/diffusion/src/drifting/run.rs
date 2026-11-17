// Drifting in 1D. Real data: 400 values, 3/4 near −1.5 and 1/4 near +1.5 (sd 0.3).
// Part 1: 100 free points drift from noise (the drifting loop), with other temperatures, attraction
// only, a start on a new draw of the data, and 10 points; mean |V|, coverage and the share in the
// right mode. Part 2: a small generator f(ε) (1 → 16 → 1) trained with the drifting loss, then
// sampled in one forward pass. Pictures in out/drifting/.
//   cargo run -p diffusion -- drifting                      (your code)
//   cargo run -p diffusion --features solution -- drifting  (the model solution)

use crate::drifting::*;
use imgcourse::*;

const MODES: [f32; 2] = [-1.5, 1.5];
const SPREAD: f32 = 0.3;
const ETA: f32 = 0.5;
const STEPS: usize = 300;
const K: usize = 5;
const LR: f32 = 0.1;
const GEN_STEPS: usize = 2000;

fn draw_data(n: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|i| MODES[usize::from(i % 4 == 3)] + SPREAD * rng.normal()).collect()
}

/// The fraction of values in the right mode (> 0).
fn right_share(v: &[f32]) -> f32 {
    v.iter().filter(|&&x| x > 0.0).count() as f32 / v.len() as f32
}

/// Column of x in a picture `w` pixels wide showing −3 … 3.
fn column(x: f32, w: usize) -> Option<usize> {
    let c = ((x + 3.0) / 6.0 * w as f32).floor();
    (0.0..w as f32).contains(&c).then_some(c as usize)
}

/// A histogram (blue bars, as a density) with the density of the data (black), from −3 to 3.
fn histogram(values: &[f32]) -> Image<[f32; 3]> {
    let (w, h, bins) = (300, 150, 60);
    let to_px = |density: f32| ((density * (h - 1) as f32 / 1.5) as usize).min(h - 1);
    let mut img = Image::new(w, h, [1.0f32; 3]);
    let mut count = vec![0usize; bins];
    for &s in values {
        if let Some(b) = column(s, bins) {
            count[b] += 1;
        }
    }
    for (b, &c) in count.iter().enumerate() {
        let top = to_px(c as f32 / (values.len() as f32 * 6.0 / bins as f32));
        for x in b * w / bins..(b + 1) * w / bins - 1 {
            for y in 0..top {
                img[(x, h - 1 - y)] = [0.4, 0.55, 0.9];
            }
        }
    }
    for x in 0..w {
        let v = -3.0 + 6.0 * (x as f32 + 0.5) / w as f32;
        let p: f32 = MODES.iter().zip([0.75, 0.25]).map(|(m, wt)| wt * (-(v - m).powi(2) / (2.0 * SPREAD * SPREAD)).exp() / (SPREAD * (2.0 * std::f32::consts::PI).sqrt())).sum();
        img[(x, h - 1 - to_px(p))] = [0.0, 0.0, 0.0];
    }
    img
}

/// Run the drifting loop on `points` for STEPS steps; print a line at steps 0, 10, 50, 300 and
/// return the paths (one row per step) as a picture: x across, step down.
fn run_points(name: &str, points: &mut [f32], real: &[f32], tau: f32, repel: f32) -> Image<f32> {
    let mut paths = Image::new(300, STEPS + 1, 1.0f32);
    let mut line = format!("{name:34}");
    let start_share = right_share(points);
    for step in 0..=STEPS {
        for &p in points.iter() {
            if let Some(c) = column(p, 300) {
                paths[(c, step)] = 0.0;
            }
        }
        let v = field(points, real, tau, repel);
        if [0, 10, 50, STEPS].contains(&step) {
            let norm = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
            line += &format!("  {step:3}: |V| {norm:5.3} cov {:4.2}", coverage(real, points, K));
        }
        if step < STEPS {
            drift_step(points, real, tau, ETA, repel);
        }
    }
    println!("{line}   right mode {:3.0} % → {:3.0} %", 100.0 * start_share, 100.0 * right_share(points));
    paths
}

pub fn main() {
    let dir = "out/drifting";
    let mut rng = Rng::new(1);
    let real = draw_data(400, &mut rng);
    let fresh = draw_data(400, &mut rng);
    println!("real data: 400 values, {:.0} % in the right mode; a new draw of 400 values covers {:.2}, of 100 covers {:.2}, of 10 covers {:.2}", 100.0 * right_share(&real), coverage(&real, &fresh, K), coverage(&real, &fresh[..100], K), coverage(&real, &fresh[..10], K));

    println!("\nfree points (step η = {ETA}), mean |V| and coverage (k = {K}) after 0, 10, 50, {STEPS} steps:");
    let noise: Vec<f32> = (0..100).map(|_| rng.normal()).collect();
    let cases: [(&str, usize, f32, f32, bool); 6] = [
        ("100 points, τ = 0.2", 100, 0.2, 1.0, false),
        ("100 points, τ = 0.03", 100, 0.03, 1.0, false),
        ("100 points, τ = 3", 100, 3.0, 1.0, false),
        ("100 points, attraction only", 100, 0.2, 0.0, false),
        ("100 points from a new draw of data", 100, 0.2, 1.0, true),
        ("10 points, τ = 0.2", 10, 0.2, 1.0, false),
    ];
    for (i, (name, n, tau, repel, from_data)) in cases.into_iter().enumerate() {
        let mut pts: Vec<f32> = if from_data { fresh[..n].to_vec() } else { noise[..n].to_vec() };
        let paths = run_points(name, &mut pts, &real, tau, repel);
        save_gray(&paths, format!("{dir}/paths{}.png", i + 1));
        save_rgb(&histogram(&pts), format!("{dir}/points{}.png", i + 1));
    }


    println!("\na generator f(ε), 16 hidden units, batches of 128 (noise and real), learning rate {LR}:");
    println!("loss of the batch, and coverage and right-mode share of 400 samples (one forward pass each)");
    let test_eps: Vec<f32> = (0..400).map(|_| rng.normal()).collect();
    for (name, tau, repel) in [("drifting, τ = 1", 1.0, 1.0), ("drifting, τ = 0.2", 0.2, 1.0), ("attraction only, τ = 1", 1.0, 0.0)] {
        let mut net = Net::new(16, &mut Rng::new(7));
        let mut batch_rng = Rng::new(8);
        let mut line = format!("{name:24}");
        for step in 0..=GEN_STEPS {
            let eps: Vec<f32> = (0..128).map(|_| batch_rng.normal()).collect();
            let batch: Vec<f32> = (0..128).map(|_| real[batch_rng.below(real.len())]).collect();
            let loss = train_step(&mut net, &eps, &batch, tau, LR, repel);
            if [0, 100, 500, GEN_STEPS].contains(&step) {
                let s: Vec<f32> = test_eps.iter().map(|&e| net.forward(e)).collect();
                line += &format!("  {step:4}: loss {loss:6.4} cov {:4.2} right {:3.0} %", coverage(&real, &s, K), 100.0 * right_share(&s));
            }
        }
        println!("{line}");
        let tag = format!("{}-tau{tau}", if repel > 0.0 { "drifting" } else { "attraction" });
        let s: Vec<f32> = test_eps.iter().map(|&e| net.forward(e)).collect();
        save_rgb(&histogram(&s), format!("{dir}/generator-{tag}.png"));
        // the map ε → f(ε) for ε from −3 (left) to 3 (right), f from −3 (bottom) to 3 (top)
        let mut map = Image::new(300, 300, 1.0f32);
        for x in 0..300 {
            let e = -3.0 + 6.0 * x as f32 / 299.0;
            if let Some(c) = column(net.forward(e), 300) {
                map[(x, 299 - c)] = 0.0;
            }
        }
        save_gray(&map, format!("{dir}/map-{tag}.png"));
    }
    println!("wrote the results to {dir}/");
}
