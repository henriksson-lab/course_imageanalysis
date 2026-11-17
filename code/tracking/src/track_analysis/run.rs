// Simulated 2D tracks of free, directed and confined particles: the ensemble MSD and D,
// the localisation offset, the spread of per-track D, and the speed trap. Prints the
// numbers and draws the MSD curves (log–log) in out/track-analysis/msd.png.
//   cargo run -p tracking -- track-analysis                      (your code)
//   cargo run -p tracking --features solution -- track-analysis  (the model solution)

use imgcourse::*;
use crate::track_analysis::*;

const D: f32 = 0.1; // µm²/s
const DT: f32 = 0.1; // s
const N: usize = 200; // points per track
const TRACKS: usize = 40;

/// Add a constant velocity `v` (µm/s) in a random direction to a track.
fn add_drift(track: &mut [Point], v: f32, rng: &mut Rng) {
    let angle = 2.0 * std::f32::consts::PI * rng.uniform();
    for (k, p) in track.iter_mut().enumerate() {
        p[0] += v * angle.cos() * k as f32 * DT;
        p[1] += v * angle.sin() * k as f32 * DT;
    }
}

/// A random walk confined to a disc of radius `r`: steps that would leave it are not taken.
fn confined_walk(n: usize, r: f32, rng: &mut Rng) -> Vec<Point> {
    let step = (2.0 * D * DT).sqrt();
    let mut p = [0.0f32; 2];
    let mut track = Vec::new();
    for _ in 0..n {
        track.push(p);
        let q = [p[0] + step * rng.normal(), p[1] + step * rng.normal()];
        if q[0] * q[0] + q[1] * q[1] <= r * r {
            p = q;
        }
    }
    track
}

/// The ensemble MSD: the mean of the single-track MSDs (all tracks equally long).
fn ensemble(tracks: &[Vec<Point>], max_lag: usize) -> Vec<f32> {
    let mut sum = vec![0.0; max_lag + 1];
    for t in tracks {
        for (s, m) in sum.iter_mut().zip(msd(t, max_lag)) {
            *s += m;
        }
    }
    sum.iter().map(|s| s / tracks.len() as f32).collect()
}

/// Fit a line to the MSD over lags 1..=`last` (times in seconds): returns (D, offset).
fn fit_d(m: &[f32], last: usize, dt: f32) -> (f32, f32) {
    let tau: Vec<f32> = (1..=last).map(|l| l as f32 * dt).collect();
    let (slope, intercept) = fit_line(&tau, &m[1..=last]);
    (slope / 4.0, intercept)
}

pub fn main() {
    let dir = "out/track-analysis";
    let mut rng = Rng::new(11);
    let max_lag = N / 4;

    // Free, directed (v = 0.5 µm/s) and confined (R = 0.5 µm): 40 tracks of 200 points each.
    let free: Vec<Vec<Point>> = (0..TRACKS).map(|_| random_walk(N, D, DT, 0.0, &mut rng)).collect();
    let directed: Vec<Vec<Point>> = (0..TRACKS)
        .map(|_| {
            let mut t = random_walk(N, D, DT, 0.0, &mut rng);
            add_drift(&mut t, 0.5, &mut rng);
            t
        })
        .collect();
    let confined: Vec<Vec<Point>> = (0..TRACKS).map(|_| confined_walk(N, 0.5, &mut rng)).collect();
    let curves = [("free", ensemble(&free, max_lag)), ("directed", ensemble(&directed, max_lag)), ("confined", ensemble(&confined, max_lag))];

    println!("{TRACKS} tracks of {N} points, D = {D} µm²/s, Δt = {DT} s; line fitted over lags 1–{} (25 %)", max_lag / 4);
    for (name, m) in &curves {
        let (d, _) = fit_d(m, max_lag / 4, DT);
        let straight: f32 = match *name {
            "free" => free.iter().map(|t| path_stats(t).2).sum::<f32>(),
            "directed" => directed.iter().map(|t| path_stats(t).2).sum::<f32>(),
            _ => confined.iter().map(|t| path_stats(t).2).sum::<f32>(),
        } / TRACKS as f32;
        // log–log slope between lag 1 and the last lag
        let alpha = (m[max_lag] / m[1]).ln() / (max_lag as f32).ln();
        println!(
            "  {name:9} D from the line {d:.3}   MSD(1) {:.4}  MSD({max_lag}) {:.3}   log–log slope {alpha:.2}   mean straightness {straight:.2}",
            m[1], m[max_lag]
        );
    }

    // Localisation error: σ = 30 nm, D = 0.01, Δt = 0.05 s (B4, E4).
    let (d2, dt2, sigma) = (0.01, 0.05, 0.03);
    let noisy: Vec<Vec<Point>> = (0..TRACKS).map(|_| random_walk(N, d2, dt2, sigma, &mut rng)).collect();
    let m = ensemble(&noisy, 4);
    let tau: Vec<f32> = (1..=4).map(|l| l as f32 * dt2).collect();
    let (slope, offset) = fit_line(&tau, &m[1..]);
    let through0 = fit_through_origin(&tau, &m[1..]);
    println!("\nlocalisation error σ = {sigma} µm, D = {d2}, Δt = {dt2} s, lags 1–4:");
    println!("  free intercept: D = {:.4}, offset {offset:.4} (4σ² = {:.4});  through the origin: D = {:.4}", slope / 4.0, 4.0 * sigma * sigma, through0 / 4.0);

    // Per-track D from the first four lags: short against long tracks (E5).
    println!("\nper-track D (line through lags 1–4), 300 free tracks each:");
    for n in [20, 200] {
        let ds: Vec<f32> = (0..300).map(|_| fit_d(&msd(&random_walk(n, D, DT, 0.0, &mut rng), 4), 4, DT).0).collect();
        let mean = ds.iter().sum::<f32>() / ds.len() as f32;
        let sd = (ds.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / ds.len() as f32).sqrt();
        println!("  {n:3} points: mean {mean:.3}, standard deviation {sd:.3} ({:.0} %)", 100.0 * sd / mean);
    }

    // The speed trap: one long walk every 5 ms, then every 2, 4, ... frames (B3, E8).
    let long = random_walk(65536, D, 0.005, 0.0, &mut rng);
    println!("\nthe speed trap (D = {D}):   Δt (s)   apparent speed (µm/s)   √(πD/Δt)");
    for k in [1, 4, 16, 64, 256] {
        let sub: Vec<Point> = long.iter().step_by(k).copied().collect();
        let dt = 0.005 * k as f32;
        println!("                            {dt:6.3}   {:8.2}                {:6.2}", apparent_speed(&sub, dt), (std::f32::consts::PI * D / dt).sqrt());
    }

    // The picture: log10 MSD against log10 lag, 1 decade = 200 px across and 100 px up.
    let mut img = Image::new(400, 420, [1.0f32; 3]);
    let to_px = |lag: f32, m: f32| ((20.0 + 200.0 * lag.log10()) as isize, (400.0 - 100.0 * (m.log10() + 2.5)) as isize);
    let colours = [[0.1, 0.3, 0.8], [0.9, 0.45, 0.0], [0.1, 0.6, 0.2]];
    // guides of slope 1 and 2 (grey) from lag 1
    for k in 0..=400 {
        let lag = 10f32.powf(k as f32 / 200.0);
        for (slope, m1) in [(1.0, 0.04), (2.0, 0.04)] {
            let (x, y) = to_px(lag, m1 * lag.powf(slope));
            if img.contains(x, y) {
                img[(x as usize, y as usize)] = [0.7, 0.7, 0.7];
            }
        }
    }
    for ((_, m), c) in curves.iter().zip(colours) {
        for lag in 1..=max_lag {
            let (x, y) = to_px(lag as f32, m[lag]);
            for dy in -2..=2 {
                for dx in -2..=2 {
                    if img.contains(x + dx, y + dy) {
                        img[((x + dx) as usize, (y + dy) as usize)] = c;
                    }
                }
            }
        }
    }
    save_rgb(&img, format!("{dir}/msd.png"));
    println!("\nwrote {dir}/msd.png: MSD against lag on log–log axes (lags 1–{max_lag}); free blue, directed orange,");
    println!("confined green; grey guides of slope 1 and 2");
}
