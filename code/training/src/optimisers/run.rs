// An optimiser race on four loss surfaces over two weights (a, b), the surfaces of the demo:
// plain gradient descent, momentum and Adam, each with several learning rates. Prints how many
// steps each needs, and draws the paths into out/optimisers/.
//   cargo run -p training -- optimisers                      (your code)
//   cargo run -p training --features solution -- optimisers  (the model solution)

use crate::optimisers::*;
use imgcourse::*;

/// A loss surface: its value and gradient at (a, b), and its lowest value.
struct Surface {
    name: &'static str,
    f: fn(f32, f32) -> f32,
    grad: fn(f32, f32) -> [f32; 2],
    min: f32,
    start: [f32; 2],
}

const SURFACES: [Surface; 4] = [
    // a long valley: curvature 0.2 along a, 10 across (along b)
    Surface { name: "valley", f: |a, b| 0.1 * a * a + 5.0 * b * b, grad: |a, b| [0.2 * a, 10.0 * b], min: 0.0, start: [-1.8, 0.8] },
    // a saddle at (0, 0); the minima are at (0, -1) and (0, 1), loss -0.25
    Surface {
        name: "saddle",
        f: |a, b| 0.5 * a * a - 0.5 * b * b + 0.25 * b.powi(4),
        grad: |a, b| [a, -b + b.powi(3)],
        min: -0.25,
        start: [-1.8, 0.001],
    },
    // a curved valley (a gentle Rosenbrock function), minimum 0 at (1, 1)
    Surface {
        name: "curved",
        f: |a, b| 0.5 * (1.0 - a).powi(2) + 5.0 * (b - a * a).powi(2),
        grad: |a, b| [-(1.0 - a) - 20.0 * a * (b - a * a), 10.0 * (b - a * a)],
        min: 0.0,
        start: [-1.5, 1.5],
    },
    // a round bowl, used with noisy gradients
    Surface { name: "bowl", f: |a, b| 0.5 * (a * a + b * b), grad: |a, b| [a, b], min: 0.0, start: [-1.8, 1.5] },
];

#[derive(Clone, Copy, PartialEq)]
enum Opt {
    Gd,
    Momentum,
    Adam,
}

/// Runs `steps` steps from the surface's start; the gradient gets noise of standard deviation
/// `noise`, and is multiplied by `scale` (a loss `scale` times larger). `schedule` gives the
/// learning rate of each step. Returns the path.
fn race(s: &Surface, opt: Opt, schedule: &dyn Fn(usize) -> f32, steps: usize, noise: f32, scale: f32, seed: u64) -> Vec<[f32; 2]> {
    let mut rng = Rng::new(seed);
    let mut w = s.start.to_vec();
    let mut v = vec![0.0; 2];
    let mut adam = AdamState::new(2);
    let mut path = vec![s.start];
    for t in 0..steps {
        let g0 = (s.grad)(w[0], w[1]);
        let g = [scale * (g0[0] + noise * rng.normal()), scale * (g0[1] + noise * rng.normal())];
        let eta = schedule(t);
        match opt {
            Opt::Gd => gd_step(&mut w, &g, eta),
            Opt::Momentum => momentum_step(&mut w, &mut v, &g, eta, 0.9),
            Opt::Adam => adam_step(&mut w, &mut adam, &g, eta),
        }
        if !w[0].is_finite() || w[0].abs() > 1e6 || w[1].abs() > 1e6 {
            path.push([f32::NAN, f32::NAN]);
            break;
        }
        path.push([w[0], w[1]]);
    }
    path
}

/// The first step at which the loss is within `tol` of the minimum (and stays there for the
/// rest of the path), as text.
fn steps_to(s: &Surface, path: &[[f32; 2]], tol: f32) -> String {
    if path.last().unwrap()[0].is_nan() {
        return "diverges".to_string();
    }
    let ok: Vec<bool> = path.iter().map(|p| (s.f)(p[0], p[1]) - s.min < tol).collect();
    match (0..ok.len()).find(|&i| ok[i..].iter().all(|&o| o)) {
        Some(i) => format!("{i}"),
        None => "-".to_string(),
    }
}

/// Loss bands of the surface over [-2, 2]², with the paths drawn on top (red, orange, purple).
fn draw(s: &Surface, paths: &[Vec<[f32; 2]>], file: &str) {
    let n = 300;
    let to_px = |v: f32| (v + 2.0) / 4.0 * (n - 1) as f32;
    let mut img = Image::from_fn(n, n, |x, y| {
        let a = -2.0 + 4.0 * x as f32 / (n - 1) as f32;
        let b = 2.0 - 4.0 * y as f32 / (n - 1) as f32;
        let t = ((s.f)(a, b) - s.min + 0.01).log10();
        let band = (t * 4.0).floor() / 4.0; // contour bands on a log scale
        let g = (0.55 + 0.12 * band).clamp(0.25, 0.97);
        [g, g, g * 1.05f32.min(1.0 / g)]
    });
    let colours = [[0.84, 0.15, 0.16], [1.0, 0.5, 0.05], [0.58, 0.4, 0.74]];
    for (k, p) in paths.iter().enumerate() {
        for q in p.windows(2) {
            if q[1][0].is_nan() {
                break;
            }
            let (x0, y0, x1, y1) = (to_px(q[0][0]), to_px(-q[0][1]), to_px(q[1][0]), to_px(-q[1][1]));
            let m = ((x1 - x0).abs().max((y1 - y0).abs()).ceil() as usize).clamp(1, 2000);
            for i in 0..=m {
                let t = i as f32 / m as f32;
                let (x, y) = ((x0 + t * (x1 - x0)).round() as isize, (y0 + t * (y1 - y0)).round() as isize);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1)] {
                    if img.contains(x + dx, y + dy) {
                        img[((x + dx) as usize, (y + dy) as usize)] = colours[k];
                    }
                }
            }
        }
    }
    save_rgb(&img, file);
}

pub fn main() {
    let dir = "out/optimisers";
    let names = ["gradient descent", "momentum (beta 0.9)", "Adam"];
    let opts = [Opt::Gd, Opt::Momentum, Opt::Adam];
    let etas = [0.003, 0.01, 0.03, 0.1, 0.19, 0.3];

    // 1. Steps to get within 0.001 of the minimum, for every optimiser and learning rate.
    for s in &SURFACES[..3] {
        println!("{}: start ({}, {}), steps until the loss stays within 0.001 of the minimum (max 2000):", s.name, s.start[0], s.start[1]);
        println!("    {:<22}{}", "learning rate", etas.map(|e| format!("{e:>9}")).join(""));
        for (o, name) in opts.iter().zip(names) {
            let cells: Vec<String> = etas.iter().map(|&e| format!("{:>9}", steps_to(s, &race(s, *o, &|_| e, 2000, 0.0, 1.0, 1), 1e-3))).collect();
            println!("    {:<22}{}", name, cells.join(""));
        }
    }

    // 2. Pictures of the race at one learning rate per optimiser.
    let pick = [("valley", [0.19, 0.03, 0.1]), ("saddle", [0.1, 0.03, 0.1]), ("curved", [0.03, 0.01, 0.3])];
    for (name, e) in pick {
        let s = SURFACES.iter().find(|s| s.name == name).unwrap();
        let paths: Vec<Vec<[f32; 2]>> = (0..3).map(|k| race(s, opts[k], &|_| e[k], 150, 0.0, 1.0, 1)).collect();
        draw(s, &paths, &format!("{dir}/race-{name}.png"));
        let ends: Vec<String> = (0..3)
            .map(|k| {
                let p = paths[k].last().unwrap();
                format!("{} (eta {}): loss {:.4}", names[k], e[k], (s.f)(p[0], p[1]))
            })
            .collect();
        println!("{name}, after 150 steps: {}", ends.join("; "));
    }

    // 3. A loss 100 times larger (the same surface in other units): the same learning rates.
    let s = &SURFACES[0];
    println!("valley with the loss multiplied by 100:");
    for (o, (name, e)) in opts.iter().zip(names.iter().zip([0.19, 0.03, 0.1])) {
        let p1 = race(s, *o, &|_| e, 2000, 0.0, 1.0, 1);
        let p100 = race(s, *o, &|_| e, 2000, 0.0, 100.0, 1);
        println!("    {name} (eta {e}): steps {} -> {}", steps_to(s, &p1, 1e-3), steps_to(s, &p100, 1e-3));
    }

    // 4. Adam's first steps with and without bias correction (valley, eta 0.1).
    for corr in [true, false] {
        let mut w = s.start.to_vec();
        let mut st = AdamState::new(2);
        st.bias_correction = corr;
        let mut sizes = Vec::new();
        for _ in 0..3 {
            let before = w.clone();
            let g = (s.grad)(w[0], w[1]);
            adam_step(&mut w, &mut st, &g, 0.1);
            sizes.push(format!("({:+.3}, {:+.3})", w[0] - before[0], w[1] - before[1]));
        }
        println!("Adam, eta 0.1, bias correction {corr}: first three steps {}", sizes.join(", "));
    }

    // 5. Noisy gradients on the bowl: constant learning rate against warm-up + cosine decay.
    let s = &SURFACES[3];
    let (steps, warm) = (300, 30);
    println!("bowl with gradient noise 1.0, {steps} steps: mean loss over the last 50 steps (mean of 20 runs)");
    for (o, (name, e)) in opts.iter().zip(names.iter().zip([0.1, 0.01, 0.1])) {
        let mut res = Vec::new();
        for sched in ["constant", "warm-up + cosine"] {
            let mut total = 0.0;
            for seed in 1..=20 {
                let p = if sched == "constant" {
                    race(s, *o, &|_| e, steps, 1.0, 1.0, seed)
                } else {
                    race(s, *o, &|t| learning_rate(t, steps, warm, e), steps, 1.0, 1.0, seed)
                };
                total += p[p.len() - 50..].iter().map(|q| (s.f)(q[0], q[1])).sum::<f32>() / 50.0;
            }
            res.push(format!("{sched} {:.4}", total / 20.0));
        }
        println!("    {name} (eta {e}): {}", res.join(", "));
    }
    let lrs: Vec<String> = [0, 9, 29, 30, 165, 299].iter().map(|&t| format!("step {t}: {:.4}", learning_rate(t, steps, warm, 0.1))).collect();
    println!("schedule (eta_max 0.1, warm-up 30 of 300 steps): {}", lrs.join(", "));
    println!("wrote the pictures to {dir}/");
}
