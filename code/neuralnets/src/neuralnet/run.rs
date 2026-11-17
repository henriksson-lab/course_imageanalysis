// Trains one neuron to tell nucleus pixels from background in a synthetic fluorescence image,
// and runs the XOR network of the lecture. Results in out/neuralnet/.
//   cargo run -p neuralnets -- neuralnet                      (your code)
//   cargo run -p neuralnets --features solution -- neuralnet  (the model solution)

use imgcourse::*;
use crate::neuralnet::*;

/// The mean of the (2r+1) × (2r+1) window around every pixel (the edge copied outside).
fn local_mean(img: &Image<f32>, r: isize) -> Image<f32> {
    let n = ((2 * r + 1) * (2 * r + 1)) as f32;
    img.map_xy(|x, y, _| {
        let mut s = 0.0;
        for dy in -r..=r {
            for dx in -r..=r {
                s += img.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
            }
        }
        s / n
    })
}

pub fn main() {
    let dir = "out/neuralnet";

    // XOR with two hidden units (slide "Two hidden units solve XOR").
    let a = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let (b, c) = (vec![0.0, -1.0], vec![1.0, -2.0]);
    for x in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] {
        println!("XOR network at {x:?}: {}", network(&a, &b, &c, 0.0, &x, true));
    }
    let corners = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let (w, bias) = train_neuron(&corners, &[0.0, 1.0, 1.0, 0.0], 1.0, 2000);
    println!("one neuron trained on XOR: loss {:.3} (ln 2 = 0.693)\n", neuron_loss(&w, bias, &corners, &[0.0, 1.0, 1.0, 0.0]));

    // A pixel classifier: two measurements per pixel, the intensity and the mean of its
    // 15 × 15 neighbourhood. The truth comes with the synthetic image.
    let (img, truth) = samples::nuclei(256, 25, 3);
    let around = local_mean(&img, 7);
    save_gray(&img, format!("{dir}/input.png"));
    save_gray(&truth.map(|l| if l > 0 { 1.0 } else { 0.0 }), format!("{dir}/truth.png"));

    // Train on every 5th pixel of every 5th row.
    let mut points = Vec::new();
    let mut labels = Vec::new();
    for (x, y) in img.coords() {
        if x % 5 == 0 && y % 5 == 0 {
            points.push(vec![img[(x, y)], around[(x, y)]]);
            labels.push(if truth[(x, y)] > 0 { 1.0 } else { 0.0 });
        }
    }
    println!("{} training pixels, {:.0} % of them nucleus", points.len(), 100.0 * labels.iter().sum::<f32>() / labels.len() as f32);
    for steps in [0, 10, 100, 1000, 5000] {
        let (w, b) = train_neuron(&points, &labels, 5.0, steps);
        println!("after {steps:5} steps: loss {:.3}, w = ({:.2}, {:.2}), b = {:.2}", neuron_loss(&w, b, &points, &labels), w[0], w[1], b);
    }
    let (w, b) = train_neuron(&points, &labels, 5.0, 5000);

    // Apply the neuron to every pixel.
    let prob = img.map_xy(|x, y, v| sigmoid(weighted_sum(&w, b, &[v, around[(x, y)]])));
    save_gray(&prob, format!("{dir}/probability.png"));
    let mut correct = 0;
    for (x, y) in img.coords() {
        if (prob[(x, y)] > 0.5) == (truth[(x, y)] > 0) {
            correct += 1;
        }
    }
    println!("pixel accuracy on the whole image: {:.1} %", 100.0 * correct as f32 / (256.0 * 256.0));
    let all_background = truth.pixels().iter().filter(|&&l| l == 0).count();
    println!("(always answering background: {:.1} %)", 100.0 * all_background as f32 / (256.0 * 256.0));

    // The feature space: intensity across, neighbourhood mean up; background pixels blue,
    // nucleus pixels orange, the background tinted by the neuron's decision.
    let n = 256;
    let to_value = |i: usize| -0.1 + 1.2 * i as f32 / n as f32;
    let mut plot = Image::from_fn(n, n, |i, j| {
        let p = sigmoid(weighted_sum(&w, b, &[to_value(i), to_value(n - 1 - j)]));
        if p > 0.5 { [0.98, 0.9, 0.82] } else { [0.85, 0.9, 0.98] }
    });
    for (x, &y) in points.iter().zip(&labels) {
        let i = ((x[0] + 0.1) / 1.2 * n as f32) as isize;
        let j = n as isize - 1 - ((x[1] + 0.1) / 1.2 * n as f32) as isize;
        if plot.contains(i, j) {
            plot[(i as usize, j as usize)] = if y > 0.5 { [0.9, 0.45, 0.0] } else { [0.1, 0.3, 0.8] };
        }
    }
    save_rgb(&plot, format!("{dir}/feature-space.png"));
    println!("wrote the pictures to {dir}/");
}
