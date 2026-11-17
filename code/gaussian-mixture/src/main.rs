// Draws points from Gaussian mixtures with your sampler, fits Gaussians to them and draws the
// most probable component everywhere; pictures in out/gaussian-mixture/.
//   cargo run -p gaussian-mixture                      (your code)
//   cargo run -p gaussian-mixture --features solution  (the model solution)

use gaussian_mixture::*;
use imgcourse::*;

/// The colour of component `k`.
fn colour(k: usize) -> [f32; 3] {
    io::label_colours(&Image::new(1, 1, k as u32 + 1))[(0, 0)]
}

/// A picture of the plane `lo..hi` (x1 to the right, x2 up): the background from `shade`,
/// the 1σ and 2σ ellipses (d_M = 1 and 2) of every component in black, and the points as dots
/// in the colour of their label (`None`: grey, the labels are hidden).
fn picture(lo: [f32; 2], hi: [f32; 2], size: usize, shade: impl Fn([f32; 2]) -> [f32; 3], ellipses: &[Gaussian], points: &[[f32; 2]], labels: Option<&[usize]>) -> Image<[f32; 3]> {
    let scale = [(hi[0] - lo[0]) / size as f32, (hi[1] - lo[1]) / size as f32];
    let at = |px: f32, py: f32| [lo[0] + px * scale[0], hi[1] - py * scale[1]];
    let mut img = Image::from_fn(size, size, |px, py| {
        let (x, right, below) = (at(px as f32 + 0.5, py as f32 + 0.5), at(px as f32 + 1.5, py as f32 + 0.5), at(px as f32 + 0.5, py as f32 + 1.5));
        for g in ellipses {
            let (d, dr, db) = (mahalanobis(x, g), mahalanobis(right, g), mahalanobis(below, g));
            for level in [1.0, 2.0] {
                if (d - level) * (dr - level) <= 0.0 || (d - level) * (db - level) <= 0.0 {
                    return [0.0; 3];
                }
            }
        }
        shade(x)
    });
    for (i, p) in points.iter().enumerate() {
        let (cx, cy) = (((p[0] - lo[0]) / scale[0]) as isize, ((hi[1] - p[1]) / scale[1]) as isize);
        let c = labels.map_or([0.45; 3], |l| colour(l[i]));
        for dy in -2..=2isize {
            for dx in -2..=2isize {
                if dx * dx + dy * dy <= 4 && img.contains(cx + dx, cy + dy) {
                    img[((cx + dx) as usize, (cy + dy) as usize)] = c;
                }
            }
        }
    }
    img
}

/// The light colour of the most probable component at `x`.
fn most_probable(x: [f32; 2], weights: &[f32], comps: &[Gaussian]) -> [f32; 3] {
    let r = responsibilities(x, weights, comps);
    let k = (0..r.len()).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap();
    colour(k).map(|c| 0.8 + 0.2 * c)
}

fn main() {
    let dir = "out/gaussian-mixture";
    let white = |_: [f32; 2]| [1.0; 3];

    // 1. The lecture's example: same Euclidean distance, different Mahalanobis distance.
    let g = Gaussian { mu: [0.0, 0.0], sigma: [[5.0, 2.0], [2.0, 2.0]] };
    for (name, x) in [("P", [2.0f32, 1.0]), ("Q", [-1.0, 2.0])] {
        println!("{name} = {x:?}: Euclidean {:.2}, Mahalanobis {:.2}, density {:.4}", (x[0] * x[0] + x[1] * x[1]).sqrt(), mahalanobis(x, &g), density(x, &g));
    }
    let mut rng = Rng::new(3);
    let (points, _) = sample_mixture(&[1.0], &[g], 300, &mut rng);
    let inside = points.iter().filter(|&&p| mahalanobis(p, &g) <= 1.0).count();
    let inside2 = points.iter().filter(|&&p| mahalanobis(p, &g) <= 2.0).count();
    println!("300 samples: {inside} inside the 1σ ellipse, {inside2} inside the 2σ ellipse");
    save_rgb(&picture([-7.0, -5.0], [7.0, 5.0], 336, white, &[g], &points, Some(&[0; 300])), format!("{dir}/one-gaussian.png"));

    // 2. A mixture of three components: the recipe, then what an algorithm sees.
    let weights = [0.5, 0.3, 0.2];
    let comps = [
        Gaussian { mu: [2.0, 2.0], sigma: [[1.0, 0.0], [0.0, 1.0]] },
        Gaussian { mu: [6.0, 3.0], sigma: [[2.0, 1.2], [1.2, 1.0]] },
        Gaussian { mu: [4.0, 6.5], sigma: [[0.3, 0.0], [0.0, 0.3]] },
    ];
    let (points, labels) = sample_mixture(&weights, &comps, 600, &mut rng);
    let (lo, hi) = ([-1.0, -1.0], [10.0, 9.0]);
    save_rgb(&picture(lo, hi, 330, white, &[], &points, Some(&labels)), format!("{dir}/mixture-coloured.png"));
    save_rgb(&picture(lo, hi, 330, white, &[], &points, None), format!("{dir}/mixture-hidden.png"));
    save_rgb(&picture(lo, hi, 330, |x| most_probable(x, &weights, &comps), &comps, &points, Some(&labels)), format!("{dir}/mixture-most-probable.png"));
    println!("mixture of 600 points, fitted with the hidden labels known:");
    for k in 0..3 {
        let mine: Vec<[f32; 2]> = (0..points.len()).filter(|&i| labels[i] == k).map(|i| points[i]).collect();
        let fit = fit_gaussian(&mine);
        println!(
            "  component {}: {} points (weight {:.2}), mean ({:.2}, {:.2}), covariance [[{:.2}, {:.2}], [{:.2}, {:.2}]]; true weight {}, mean {:?}, covariance {:?}",
            k + 1,
            mine.len(),
            mine.len() as f32 / 600.0,
            fit.mu[0],
            fit.mu[1],
            fit.sigma[0][0],
            fit.sigma[0][1],
            fit.sigma[1][0],
            fit.sigma[1][1],
            weights[k],
            comps[k].mu,
            comps[k].sigma
        );
    }
    let x = [4.0, 3.0];
    let r = responsibilities(x, &weights, &comps);
    println!("responsibilities at {x:?}: {:.2} {:.2} {:.2}", r[0], r[1], r[2]);

    // 3. Bayes boundaries of two equally weighted components: same covariance, or not.
    let tilted = [[2.0, 1.0], [1.0, 1.0]];
    let tied = [Gaussian { mu: [2.0, 3.0], sigma: tilted }, Gaussian { mu: [6.0, 3.0], sigma: tilted }];
    let different = [Gaussian { mu: [3.0, 3.5], sigma: [[0.4, 0.0], [0.0, 0.4]] }, Gaussian { mu: [5.0, 3.0], sigma: [[4.0, 0.0], [0.0, 4.0]] }];
    for (name, c) in [("tied", tied), ("different", different)] {
        save_rgb(&picture([-1.0, -1.0], [9.0, 7.0], 300, |x| most_probable(x, &[0.5, 0.5], &c), &c, &[], None), format!("{dir}/boundary-{name}.png"));
    }
    println!("wrote the pictures to {dir}/");
}
