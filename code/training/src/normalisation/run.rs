// How the size of the signal changes through a deep stack of layers, for several initial
// scales, bends and normalisations; how large the gradients are on the way back; and the
// batch-statistics trap at test time, with patches of the course micrograph; a deep plain network
// against the same network with residual connections.
// Plots in out/normalisation/.
//   cargo run -p training -- normalisation                      (your code)
//   cargo run -p training --features solution -- normalisation  (the model solution)

use crate::common::{plot_curves, spiral, BLUE, GREEN, ORANGE, RED};
use crate::normalisation::resnet::{train_deep, DeepNet};
use crate::normalisation::*;
use imgcourse::*;

const DEPTH: usize = 50;
const WIDTH: usize = 64;
const BATCH: usize = 128;

/// The sizes after layers 1, 2, 5, 10, 20 and 50, as text.
fn row(sizes: &[f32]) -> String {
    [1, 2, 5, 10, 20, 50].iter().map(|&l| format!("{:>10.3e}", sizes[l - 1])).collect::<Vec<_>>().join("")
}

/// Root mean square of the gradient dL/da at the output of every layer, going backwards, for the
/// stack of `signal_sizes` without normalisation and the loss L = sum of r * output with random r
/// (normal). Element l is for the output of layer l + 1.
fn gradient_sizes(gain: f32, act: Act, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut a: Vec<Vec<f32>> = (0..BATCH).map(|_| (0..WIDTH).map(|_| rng.normal()).collect()).collect();
    let mut ws = Vec::new();
    let mut zs = Vec::new();
    for _ in 0..DEPTH {
        let w = init_weights(WIDTH, WIDTH, gain, &mut rng);
        let z: Vec<Vec<f32>> = a.iter().map(|x| (0..WIDTH).map(|j| (0..WIDTH).map(|i| w[j * WIDTH + i] * x[i]).sum()).collect()).collect();
        a = z.iter().map(|r| r.iter().map(|&v| bend(act, v)).collect()).collect();
        ws.push(w);
        zs.push(z);
    }
    // backwards: d = dL/da of the top layer is r; then dL/dz = d * bend'(z); dL/da_below = W^T dL/dz
    let mut d: Vec<Vec<f32>> = (0..BATCH).map(|_| (0..WIDTH).map(|_| rng.normal()).collect()).collect();
    let mut sizes = vec![0.0; DEPTH];
    for l in (0..DEPTH).rev() {
        sizes[l] = (d.iter().flatten().map(|v| v * v).sum::<f32>() / (BATCH * WIDTH) as f32).sqrt();
        let dz: Vec<Vec<f32>> = d.iter().zip(&zs[l]).map(|(dr, zr)| dr.iter().zip(zr).map(|(&g, &z)| g * slope(act, z)).collect()).collect();
        d = dz.iter().map(|g| (0..WIDTH).map(|i| (0..WIDTH).map(|j| ws[l][j * WIDTH + i] * g[j]).sum()).collect()).collect();
    }
    sizes
}

fn bend(act: Act, v: f32) -> f32 {
    match act {
        Act::Linear => v,
        Act::Relu => v.max(0.0),
        Act::Tanh => v.tanh(),
    }
}

fn slope(act: Act, z: f32) -> f32 {
    match act {
        Act::Linear => 1.0,
        Act::Relu => {
            if z > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Act::Tanh => 1.0 - z.tanh().powi(2),
    }
}

/// `n` random 8 × 8 patches of an image, as vectors of 64 grey values.
fn patches(img: &Image<f32>, n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let (x0, y0) = (rng.below(img.width() - 8), rng.below(img.height() - 8));
            img.crop(x0, y0, 8, 8).pixels().to_vec()
        })
        .collect()
}

/// Mean and standard deviation of all values of a batch.
fn mean_sd(z: &[Vec<f32>]) -> (f32, f32) {
    let v: Vec<f32> = z.iter().flatten().cloned().collect();
    let m = v.iter().sum::<f32>() / v.len() as f32;
    (m, (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt())
}

/// Normalises with stored means and variances (batch normalisation at test time).
fn with_stats(z: &[Vec<f32>], means: &[f32], vars: &[f32]) -> Vec<Vec<f32>> {
    z.iter().map(|r| r.iter().enumerate().map(|(j, &v)| (v - means[j]) / (vars[j] + 1e-5).sqrt()).collect()).collect()
}

pub fn main() {
    let dir = "out/normalisation";
    let sq2 = 2f32.sqrt();
    println!("size of the signal (root mean square) after layer   1         2         5        10        20        50");
    println!("({DEPTH} layers of {WIDTH} units, a batch of {BATCH} random inputs of size 1)");
    let cases: [(&str, f32, Act, Norm); 12] = [
        ("linear, gain 0.5", 0.5, Act::Linear, Norm::None),
        ("linear, gain 1", 1.0, Act::Linear, Norm::None),
        ("linear, gain 1.5", 1.5, Act::Linear, Norm::None),
        ("ReLU, gain 1", 1.0, Act::Relu, Norm::None),
        ("ReLU, gain sqrt 2 (He)", sq2, Act::Relu, Norm::None),
        ("ReLU, gain 2", 2.0, Act::Relu, Norm::None),
        ("tanh, gain 1", 1.0, Act::Tanh, Norm::None),
        ("tanh, gain 3", 3.0, Act::Tanh, Norm::None),
        ("ReLU, gain 1, batch norm", 1.0, Act::Relu, Norm::Batch),
        ("ReLU, gain 2, batch norm", 2.0, Act::Relu, Norm::Batch),
        ("ReLU, gain 1, layer norm", 1.0, Act::Relu, Norm::Layer),
        ("ReLU, gain 2, layer norm", 2.0, Act::Relu, Norm::Layer),
    ];
    let mut curves = Vec::new();
    for (name, gain, act, norm) in cases {
        let s = signal_sizes(DEPTH, WIDTH, gain, act, norm, BATCH, 1);
        println!("    {name:<28}{}", row(&s));
        curves.push(s);
    }
    // log10 of the sizes for the ReLU cases: gain 1, sqrt 2, 2, and gain 1 with batch norm
    let logs: Vec<Vec<f32>> = [3, 4, 5, 8].iter().map(|&k| curves[k].iter().map(|v| v.max(1e-12).log10()).collect()).collect();
    plot_curves(&[(&logs[0], BLUE), (&logs[1], GREEN), (&logs[2], RED), (&logs[3], ORANGE)], (-8.0, 8.0), 2.0, &format!("{dir}/signal-relu.png"));

    println!("size of the gradient at the output of layer        1         2         5        10        20        50");
    println!("(the same stacks without normalisation, loss = random weights times the last layer)");
    for (name, gain, act) in [("ReLU, gain 1", 1.0, Act::Relu), ("ReLU, gain sqrt 2 (He)", sq2, Act::Relu), ("tanh, gain 1", 1.0, Act::Tanh), ("tanh, gain 3", 3.0, Act::Tanh)] {
        println!("    {name:<28}{}", row(&gradient_sizes(gain, act, 1)));
    }

    // The batch-statistics trap: one batch-norm layer on 64 grey values of a patch.
    let cell = samples::cell();
    let mut train = patches(&cell, 500, 2);
    let (means, vars) = batch_norm(&mut train, 1e-5);
    let (m, s) = mean_sd(&train);
    println!("batch norm on 500 training patches (8 x 8) of the micrograph: outputs mean {m:.3}, sd {s:.3}");
    let test = patches(&cell, 200, 3);
    let (m, s) = mean_sd(&with_stats(&test, &means, &vars));
    println!("    new patches, stored statistics: mean {m:.3}, sd {s:.3}");
    let mut one = vec![test[0].clone()];
    batch_norm(&mut one, 1e-5);
    let largest = one[0].iter().fold(0.0f32, |a, v| a.max(v.abs()));
    println!("    one patch normalised with its own batch statistics (a batch of 1): largest |output| {largest:.3}");
    let other = cell.map(|v| (1.6 * v + 0.15).min(1.0));
    let shifted = patches(&other, 200, 3);
    let (m, s) = mean_sd(&with_stats(&shifted, &means, &vars));
    println!("    'other microscope' (grey value x 1.6 + 0.15), stored statistics: mean {m:.3}, sd {s:.3}");
    let mut target = patches(&other, 50, 4);
    let (m2, v2) = batch_norm(&mut target, 1e-5);
    let (m, s) = mean_sd(&with_stats(&shifted, &m2, &v2));
    println!("    the same, statistics re-estimated on 50 patches of the new microscope: mean {m:.3}, sd {s:.3}");
    save_gray(&other, format!("{dir}/other-microscope.png"));

    // Residual connections: plain and residual networks of 16 units on the three-armed spiral.
    let (xs, ys) = spiral([100, 100, 100], 0.1, 1);
    println!("plain and residual networks, 16 units per layer, three-armed spiral (300 points), Adam 0.003, 60 epochs");
    println!("    training loss at epoch      0      5     20     60   (gradient size at the first and the last block, first batch)");
    let mut plots = Vec::new();
    for (blocks, residual) in [(2, false), (20, false), (20, true), (50, false), (50, true)] {
        let mut net = DeepNet::new(16, blocks, residual, 4);
        let (curve, sizes) = train_deep(&mut net, &xs, &ys, 60, 0.003);
        let acc = net.loss_and_accuracy(&xs, &ys).1;
        println!(
            "    {:<3} blocks, {:<8}  {}   {:.1e} / {:.1e}   accuracy {:.0} %",
            blocks,
            if residual { "residual" } else { "plain" },
            [0, 5, 20, 60].map(|e| format!("{:>7.3}", curve[e])).join(""),
            sizes[1],
            sizes[blocks],
            100.0 * acc
        );
        plots.push(curve);
    }
    plot_curves(&[(&plots[0], GREEN), (&plots[1], RED), (&plots[2], BLUE)], (0.0, 1.2), 0.2, &format!("{dir}/plain-vs-residual.png"));
    // the curves as a table, for figures: one line per epoch
    let mut csv = String::from("epoch,plain2,plain20,residual20,plain50,residual50\n");
    for e in 0..plots[0].len() {
        csv += &format!("{e},{}\n", plots.iter().map(|c| format!("{:.4}", c[e])).collect::<Vec<_>>().join(","));
    }
    std::fs::write(format!("{dir}/plain-vs-residual.csv"), csv).expect("write the table");
    println!("wrote the plots to {dir}/ (plain-vs-residual.png: training loss, green 2 plain blocks, red 20 plain, blue 20 residual; signal-relu.png: log10 of the size, -8 to 8; blue gain 1, green sqrt 2, red 2, orange gain 1 + batch norm)");
}
