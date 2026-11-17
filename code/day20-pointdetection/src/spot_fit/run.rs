// Localises simulated spots (width s = 1 px, Poisson noise) with the centroid, the
// (log-)parabola and a Gaussian fit: the bias without noise, and the precision against the
// number of photons. Pictures of a patch and its fit residual in out/spot-fit/.
//   cargo run -p day20 -- spot-fit                      (your code)
//   cargo run -p day20 --features solution -- spot-fit  (the model solution)

use imgcourse::*;
use crate::spot_fit::sim::{brightest, enlarge, spot_mean, with_poisson};
use crate::spot_fit::*;

/// The centroid in a (2m+1) × (2m+1) window around the brightest pixel, in patch coordinates.
fn centroid_window(patch: &Image<f32>, m: usize, background: f32) -> [f32; 2] {
    let (bx, by) = brightest(patch);
    let (x0, y0) = (bx.clamp(m, patch.width() - 1 - m) - m, by.clamp(m, patch.height() - 1 - m) - m);
    let c = centroid(&patch.crop(x0, y0, 2 * m + 1, 2 * m + 1), background);
    [x0 as f32 + c[0], y0 as f32 + c[1]]
}

pub fn main() {
    let dir = "out/spot-fit";
    let (size, s) = (13, 1.0);

    // 1. Bias: no noise, 1 000 photons, background 20 per pixel, true centre 6 + δ.
    println!("bias (estimate − truth, px) without noise; N = 1000, b = 20, s = 1 px");
    println!("    δ   centroid 7×7   parabola   log-parabola   fit");
    for i in 0..=4 {
        let delta = 0.1 * i as f32;
        let patch = spot_mean(size, &[(6.0 + delta, 6.0, 1000.0)], s, 20.0);
        let c = centroid_window(&patch, 3, 0.0)[0]; // background not subtracted
        let p = three_point(&patch, 0.0, false)[0];
        let l = three_point(&patch, 20.0, true)[0]; // background subtracted
        let f = fit_spot(&patch, [6.0, 6.0], s, 10).x;
        let t = 6.0 + delta;
        println!("  {delta:.1}   {:>12.3}   {:>8.3}   {:>12.3}   {:>5.3}", c - t, p - t, l - t, f - t);
    }

    // 2. Precision: 500 seeded repeats, true centre (6.3, 5.8). With pixels of 100 nm and
    // s = 100 nm, the bound s/√N in nm is 100/√N.
    println!("RMS error in nm (pixel 100 nm, s = 100 nm), 500 repeats");
    println!("     N    b   centroid 5×5   log-parabola     fit   bound s/√N");
    let mut rng = Rng::new(1);
    for (n, b) in [(100.0, 0.0), (1000.0, 0.0), (10000.0, 0.0), (1000.0, 10.0), (300.0, 20.0)] {
        let mean = spot_mean(size, &[(6.3, 5.8, n)], s, b);
        let mut err = [0.0f32; 3];
        let repeats = 500;
        for _ in 0..repeats {
            let patch = with_poisson(&mean, &mut rng);
            let (bx, by) = brightest(&patch);
            let est = [
                centroid_window(&patch, 2, b),
                three_point(&patch, b, true),
                {
                    let f = fit_spot(&patch, [bx as f32, by as f32], s, 6);
                    [f.x, f.y]
                },
            ];
            for (e, p) in err.iter_mut().zip(est) {
                *e += (p[0] - 6.3).powi(2) + (p[1] - 5.8).powi(2);
            }
        }
        // per axis, in nm
        let rms = err.map(|e| 100.0 * (e / (2.0 * repeats as f32)).sqrt());
        println!("{n:>6} {b:>4}   {:>12.1}   {:>12.1}   {:>5.1}   {:>10.1}", rms[0], rms[1], rms[2], 100.0 / n.sqrt());
    }

    // 3. Pictures: one spot, and two spots 3 px apart, each fitted with one spot; the residual.
    let mut rng = Rng::new(2);
    for (name, spots) in [("one", vec![(6.3, 5.8, 1000.0)]), ("two", vec![(4.8, 6.0, 1000.0), (7.8, 6.0, 1000.0)])] {
        let patch = with_poisson(&spot_mean(15, &spots, 1.5, 5.0), &mut rng);
        let (bx, by) = brightest(&patch);
        let f = fit_spot(&patch, [bx as f32, by as f32], 1.5, 10);
        let model = Image::from_fn(15, 15, |x, y| f.background + f.amplitude * shape(x, y, f.x, f.y, 1.5));
        let residual = patch.zip_map(&model, |a, b| a - b);
        let rss: f32 = residual.pixels().iter().map(|r| r * r).sum();
        println!("{name} spot(s), fitted with one: centre ({:.2}, {:.2}), residual sum of squares {rss:.0}", f.x, f.y);
        save_gray_stretched(&enlarge(&patch, 12), format!("{dir}/{name}-data.png"));
        save_gray_signed(&enlarge(&residual, 12), format!("{dir}/{name}-residual.png"));
    }
    println!("wrote the pictures to {dir}/");
}
