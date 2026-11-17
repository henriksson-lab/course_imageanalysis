// The random walker on chains, on the micrograph, on a flat image and on synthetic nuclei.
// Pictures in out/random-walker/.
//   cargo run -p graphseg -- random-walker                      (your code)
//   cargo run -p graphseg --features solution -- random-walker  (the model solution)

use imgcourse::*;
use crate::random_walker::*;

/// The micrograph at half size, 128 × 128 (each pixel the mean of 2 × 2).
fn half_cell() -> Image<f32> {
    let c = samples::cell();
    Image::from_fn(128, 128, |x, y| (c[(2 * x, 2 * y)] + c[(2 * x + 1, 2 * y)] + c[(2 * x, 2 * y + 1)] + c[(2 * x + 1, 2 * y + 1)]) / 4.0)
}

/// The image in grey with the seeds in colour (1 orange, 2 blue, others green) and the
/// pixels where the probability crosses 0.5 in yellow.
fn overlay(img: &Image<f32>, seeds: &Seeds, prob: &Image<f32>) -> Image<[f32; 3]> {
    img.map_xy(|x, y, v| {
        let inside = prob[(x, y)] > 0.5;
        let edge = (x + 1 < img.width() && (prob[(x + 1, y)] > 0.5) != inside) || (y + 1 < img.height() && (prob[(x, y + 1)] > 0.5) != inside);
        match seeds[(x, y)] {
            1 => [1.0, 0.55, 0.0],
            2 => [0.2, 0.45, 1.0],
            0 if edge => [1.0, 0.95, 0.0],
            0 => [v, v, v],
            _ => [0.2, 0.85, 0.3],
        }
    })
}

pub fn main() {
    let dir = "out/random-walker";

    // Chains (B1, B3).
    let v = chain_voltages(&[1.0, 1.0, 0.25, 1.0], &[Some(1.0), None, None, None, Some(0.0)]);
    println!("B1 chain: voltages {v:.3?}");
    let w3 = [0.2, 1.0, 1.0, 1.0, 1.0, 0.25];
    let v = chain_voltages(&w3, &[Some(1.0), None, None, None, None, None, Some(0.0)]);
    println!("B3 chain: voltages {v:.3?}");
    // The same chain by averaging sweeps, from 0.5 at the free nodes (demo question W3).
    let seeds = Image::from_vec(7, 1, vec![1, 0, 0, 0, 0, 0, 2]);
    let weights = Weights { wx: Image::from_vec(7, 1, vec![0.2, 1.0, 1.0, 1.0, 1.0, 0.25, 0.0]), wy: Image::new(7, 1, 0.0) };
    let mut x = Image::from_vec(7, 1, vec![1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]);
    let mut sweeps = 1;
    while averaging_sweep(&mut x, &seeds, &weights) >= 0.001 {
        sweeps += 1;
    }
    println!("B3 chain by averaging: largest change below 0.001 after {sweeps} sweeps: {:.3?}", x.pixels());

    // The micrograph: one object stroke in the nucleus, background strokes on three borders.
    let img = half_cell();
    let mut seeds = Image::new(128, 128, 0u32);
    for y in 50..80 {
        seeds[(70, y)] = 1;
    }
    for k in 0..128 {
        seeds[(0, k)] = 2;
        seeds[(k, 0)] = 2;
        seeds[(127, k)] = 2;
    }
    let sigma = 0.08;
    let weights = edge_weights(&img, sigma);
    let (prob, iterations) = random_walker(&seeds, &weights, 1, 1e-5, 5000);
    let object = prob.pixels().iter().filter(|&&p| p > 0.5).count();
    println!("\nmicrograph 128 × 128, σ = {sigma}: {iterations} CG iterations, {object} object pixels");
    save_gray(&prob, format!("{dir}/cell-probability.png"));
    save_rgb(&overlay(&img, &seeds, &prob), format!("{dir}/cell-seeds.png"));
    // Probe two pixels with 200 walks each.
    let mut rng = Rng::new(1);
    for p in [(75, 40), (40, 100)] {
        let est = walk_estimate(&seeds, &weights, 1, p, 200, &mut rng);
        println!("  pixel {p:?}: probability {:.3}, 200 walks give {est:.3}", prob[p]);
    }

    // No edges at all: one seed in each corner of a flat 64 × 64 image.
    let flat = Image::new(64, 64, 0.5f32);
    let mut corner = Image::new(64, 64, 0u32);
    corner[(1, 1)] = 1;
    corner[(62, 62)] = 2;
    let (p, _) = random_walker(&corner, &edge_weights(&flat, sigma), 1, 1e-6, 5000);
    let diag: Vec<f32> = [1, 2, 4, 8, 16, 32, 48, 56, 60, 61, 62].iter().map(|&k| p[(k, k)]).collect();
    println!("\nflat image, seeds at (1, 1) and (62, 62): along the diagonal at 1, 2, 4, 8, 16, 32, 48, 56, 60, 61, 62:\n  {diag:.2?}");
    save_gray(&p, format!("{dir}/flat-probability.png"));

    // Synthetic nuclei: a 3 × 3 seed at the centre of each nucleus, background seeds on the
    // border; one system per label, and each pixel gets the label of largest probability.
    let (nuc, truth) = samples::nuclei(128, 8, 3);
    let n = truth.pixels().iter().copied().max().unwrap_or(0);
    let mut seeds = Image::new(128, 128, 0u32);
    for k in 0..128 {
        for (x, y) in [(0, k), (127, k), (k, 0), (k, 127)] {
            seeds[(x, y)] = 1; // background = label 1 here
        }
    }
    for l in 1..=n {
        let pix: Vec<(usize, usize)> = truth.coords().filter(|&p| truth[p] == l).collect();
        if pix.is_empty() {
            continue;
        }
        let cx = pix.iter().map(|p| p.0).sum::<usize>() / pix.len();
        let cy = pix.iter().map(|p| p.1).sum::<usize>() / pix.len();
        for dy in 0..3 {
            for dx in 0..3 {
                seeds[(cx + dx - 1, cy + dy - 1)] = l + 1;
            }
        }
    }
    let weights = edge_weights(&nuc, sigma);
    let mut best = Image::new(128, 128, (0.0f32, 0u32));
    for label in 1..=n + 1 {
        let (p, _) = random_walker(&seeds, &weights, label, 1e-5, 5000);
        best = best.zip_map(&p, |b, v| if v > b.0 { (v, label) } else { b });
    }
    let labels = best.map(|(_, l)| l - 1); // background back to 0, nucleus k to k
    let agree = labels.pixels().iter().zip(truth.pixels()).filter(|(a, b)| a == b).count();
    println!("\nsynthetic nuclei ({n}): {agree} of {} pixels get their true label ({:.1} %)", 128 * 128, 100.0 * agree as f32 / 16384.0);
    save_gray(&nuc, format!("{dir}/nuclei.png"));
    save_labels(&labels, format!("{dir}/nuclei-walker.png"));
    save_labels(&truth, format!("{dir}/nuclei-truth.png"));
    println!("\nwrote the pictures to {dir}/");
}
