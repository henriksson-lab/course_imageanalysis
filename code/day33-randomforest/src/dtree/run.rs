// Grows your decision trees on pixels of a synthetic fluorescence image and tests them on
// another image; results in out/dtree/.
//   cargo run -p day33 -- dtree                      (your code)
//   cargo run -p day33 --features solution -- dtree  (the model solution)

use crate::dtree::*;
use imgcourse::*;

/// Two features per pixel, as in the exercise: the 3 × 3 mean (intensity) and the 3 × 3
/// local variance (texture). Not part of this unit's exercises.
fn features(img: &Image<f32>) -> Image<[f32; 2]> {
    img.map_xy(|x, y, _| {
        let (mut sum, mut sum2) = (0.0, 0.0);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let v = img.get_border(x as isize + dx, y as isize + dy, Border::Mirror);
                sum += v;
                sum2 += v * v;
            }
        }
        let mean = sum / 9.0;
        [mean, sum2 / 9.0 - mean * mean]
    })
}

/// A noisy synthetic image, its pixel features and classes (1 = nucleus, 0 = background).
fn image(seed: u64) -> (Image<f32>, Image<[f32; 2]>, Image<usize>) {
    let (img, labels) = samples::nuclei(128, 8, seed);
    let img = samples::add_noise(&img, 0.6, seed);
    let f = features(&img);
    (img, f, labels.map(|l| (l > 0) as usize))
}

pub fn main() {
    let dir = "out/dtree";
    let (img_a, feat_a, class_a) = image(1);
    let (img_b, feat_b, class_b) = image(2);
    save_gray(&img_a, format!("{dir}/train-image.png"));
    save_gray(&img_b, format!("{dir}/test-image.png"));

    // Training: 300 random pixels of image A. Test: every pixel of image B.
    let mut rng = Rng::new(5);
    let (mut points, mut labels) = (Vec::new(), Vec::new());
    for _ in 0..300 {
        let (x, y) = (rng.below(128), rng.below(128));
        points.push(feat_a[(x, y)].to_vec());
        labels.push(class_a[(x, y)]);
    }
    let test_points: Vec<Vec<f32>> = feat_b.pixels().iter().map(|f| f.to_vec()).collect();
    let test_labels = class_b.pixels().to_vec();

    println!("depth  leaves  training error  test error (other image)");
    for depth in [1, 2, 3, 4, 5, 6, 8, 10, 15] {
        let tree = grow(&points, &labels, 2, depth, 1);
        println!(
            "{depth:5}  {:6}  {:13.1} %  {:9.1} %",
            leaves(&tree),
            100.0 * error_rate(&tree, &points, &labels),
            100.0 * error_rate(&tree, &test_points, &test_labels)
        );
        if depth == 3 || depth == 15 {
            let seg = feat_b.map(|f| predict(&tree, &f) as f32);
            save_gray(&seg, format!("{dir}/test-depth{depth}.png"));
        }
    }
    let stump = grow(&points, &labels, 2, 1, 1);
    if let Tree::Split { feature, threshold, .. } = stump {
        println!("the root asks: {} ≤ {threshold:.3}?", ["mean", "variance"][feature]);
    }
    save_gray(&class_b.map(|c| c as f32), format!("{dir}/test-truth.png"));

    // A regression stump across a blurred, noisy edge from 40 to 160 (lecture demo).
    let xs: Vec<f32> = (0..60).map(|i| i as f32 / 59.0).collect();
    let mut rng = Rng::new(3);
    let ys: Vec<f32> = xs.iter().map(|&x| 40.0 + 120.0 / (1.0 + (-(x - 0.45) / 0.06).exp()) + 10.0 * rng.normal()).collect();
    let (threshold, sse) = regression_stump(&xs, &ys);
    println!("regression stump on a noisy edge: split at x = {threshold:.3}, mean squared error {:.1}", sse / 60.0);
    println!("wrote the results to {dir}/");
}
