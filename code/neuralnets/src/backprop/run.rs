// Fits the line-scan network (1 input, H hidden ReLU units, 1 output) by backpropagation and
// gradient descent to a noisy line scan across two nuclei. Plots in out/backprop/.
//   cargo run -p neuralnets -- backprop                      (your code)
//   cargo run -p neuralnets --features solution -- backprop  (the model solution)

use crate::backprop::*;
use imgcourse::*;

/// A line scan across two nuclei of different brightness: 100 pixels, positions 0 to 1,
/// with noise of standard deviation 0.03.
fn line_scan() -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(7);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..100 {
        let x = i as f32 / 99.0;
        // smooth steps up and down (a blurred edge)
        let step = |a: f32| 1.0 / (1.0 + (-(x - a) / 0.015).exp());
        let bright = 0.75 * (step(0.08) - step(0.42));
        let dim = 0.4 * (step(0.6) - step(0.85));
        xs.push(x);
        ys.push(0.15 + bright + dim + 0.03 * rng.normal());
    }
    (xs, ys)
}

/// Draws the data (grey dots) and the network's output (red) into a 400 × 200 picture.
fn plot(net: &Net, xs: &[f32], ys: &[f32]) -> Image<[f32; 3]> {
    let (w, h) = (400, 200);
    let mut img = Image::new(w, h, [1.0f32; 3]);
    let row = |v: f32| (h as f32 - 1.0 - v * (h as f32 - 1.0)).round() as isize;
    for (&x, &y) in xs.iter().zip(ys) {
        let (cx, cy) = ((x * (w as f32 - 1.0)) as isize, row(y));
        for dy in -1..=1 {
            for dx in -1..=1 {
                if img.contains(cx + dx, cy + dy) {
                    img[((cx + dx) as usize, (cy + dy) as usize)] = [0.55; 3];
                }
            }
        }
    }
    let mut last = row(forward(net, 0.0).y_hat);
    for px in 0..w {
        let r = row(forward(net, px as f32 / (w as f32 - 1.0)).y_hat);
        // a vertical run from the last row to this one, so the curve has no gaps
        for y in r.min(last)..=r.max(last) {
            if img.contains(px as isize, y) {
                img[(px, y as usize)] = [0.85, 0.1, 0.1];
            }
        }
        last = r;
    }
    img
}

pub fn main() {
    let dir = "out/backprop";
    let (xs, ys) = line_scan();
    let mean = ys.iter().sum::<f32>() / ys.len() as f32;
    let flat: f32 = ys.iter().map(|y| (y - mean) * (y - mean)).sum::<f32>() / ys.len() as f32;
    println!("loss of the flat line at the mean: {flat:.4}; noise variance 0.03^2 = 0.0009");

    for (hidden, eta) in [(1, 0.05), (3, 0.05), (12, 0.05), (12, 0.5)] {
        let mut net = Net::new(hidden);
        net.b2 = mean;
        print!("{hidden:2} hidden units, learning rate {eta}: loss");
        for step in 0..=20000 {
            let l = train_step(&mut net, &xs, &ys, eta);
            if [0, 100, 1000, 5000, 20000].contains(&step) {
                print!("  {l:.4} (step {step})");
            }
        }
        // where every hinge bends: w1 x + b1 = 0
        let mut bends: Vec<String> = Vec::new();
        for k in 0..hidden {
            let at = -net.b1[k] / net.w1[k];
            let on_somewhere = forward(&net, 0.0).h[k] > 0.0 || forward(&net, 1.0).h[k] > 0.0;
            bends.push(if on_somewhere { format!("{at:.2}") } else { "dead".to_string() });
        }
        println!("\n    hinges at {}", bends.join(", "));
        save_rgb(&plot(&net, &xs, &ys), format!("{dir}/fit-{hidden}-units-eta{eta}.png"));
    }
    println!("wrote the plots to {dir}/");
}
