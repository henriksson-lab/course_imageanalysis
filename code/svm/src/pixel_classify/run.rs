// A pixel classifier with linear SVMs on a synthetic fluorescence image: painted strokes, a
// small feature stack, standardisation, class weights and cross-validation by stroke.
// Pictures in out/pixel-classify/.
//   cargo run -p svm -- pixel-classify                      (your code)
//   cargo run -p svm --features solution -- pixel-classify  (the model solution)

use imgcourse::*;
use crate::pixel_classify::*;

const NAMES: [&str; 3] = ["background", "nucleus", "edge"];

/// The mean and standard deviation of the 5 × 5 neighbourhood of every pixel.
fn local_mean_sd(img: &Image<f32>) -> (Image<f32>, Image<f32>) {
    let stats = img.map_xy(|x, y, _| {
        let (mut s, mut s2) = (0.0, 0.0);
        for dy in -2..=2isize {
            for dx in -2..=2isize {
                let v = img.get_border(x as isize + dx, y as isize + dy, Border::Mirror);
                s += v / 25.0;
                s2 += v * v / 25.0;
            }
        }
        [s, (s2 - s * s).max(0.0).sqrt()]
    });
    (stats.map(|v| v[0]), stats.map(|v| v[1]))
}

/// The length of the Sobel gradient.
fn gradient(img: &Image<f32>) -> Image<f32> {
    img.map_xy(|x, y, _| {
        let p = |dx: isize, dy: isize| img.get_border(x as isize + dx, y as isize + dy, Border::Mirror);
        let gx = p(1, -1) + 2.0 * p(1, 0) + p(1, 1) - p(-1, -1) - 2.0 * p(-1, 0) - p(-1, 1);
        let gy = p(-1, 1) + 2.0 * p(0, 1) + p(1, 1) - p(-1, -1) - 2.0 * p(0, -1) - p(1, -1);
        (gx * gx + gy * gy).sqrt()
    })
}

/// The true class of every pixel: 2 (edge) within 2 pixels of a change of label, otherwise
/// 1 (nucleus) or 0 (background).
fn true_classes(labels: &Image<u32>) -> Image<usize> {
    labels.map_xy(|x, y, l| {
        for dy in -2..=2isize {
            for dx in -2..=2isize {
                if labels.get_border(x as isize + dx, y as isize + dy, Border::Clamp) != l {
                    return 2;
                }
            }
        }
        if l > 0 { 1 } else { 0 }
    })
}

/// Painted strokes: `count[c]` short horizontal strokes (up to 10 pixels) inside class `c`.
/// Returns the pixel positions and, for each pixel, its class and its stroke number.
fn paint(truth: &Image<usize>, count: [usize; 3], rng: &mut Rng) -> Vec<((usize, usize), usize, usize)> {
    let mut out = Vec::new();
    let mut stroke = 0;
    for (c, &n) in count.iter().enumerate() {
        let mut made = 0;
        while made < n {
            let (x0, y) = (rng.below(truth.width() - 10), rng.below(truth.height()));
            if truth[(x0, y)] != c {
                continue;
            }
            for x in x0..x0 + 10 {
                if truth[(x, y)] != c {
                    break;
                }
                out.push(((x, y), c, stroke));
            }
            stroke += 1;
            made += 1;
        }
    }
    out
}

/// Classify every pixel with one-vs-rest linear SVMs; print the share of correct pixels and
/// the recall of each class; save the result.
fn classify_image(name: &str, models: &[Linear], features: &Image<[f32; 4]>, truth: &Image<usize>) {
    let result = features.map(|f| predict_one_vs_rest(models, &f));
    let correct = truth.coords().filter(|&p| result[p] == truth[p]).count();
    let mut line = format!("  {name:<32} {:.1} % correct; found", 100.0 * correct as f32 / truth.pixels().len() as f32);
    for c in 0..3 {
        let total = truth.pixels().iter().filter(|&&t| t == c).count();
        let found = truth.coords().filter(|&p| truth[p] == c && result[p] == c).count();
        line += &format!(" {:.0} % of {}", 100.0 * found as f32 / total as f32, NAMES[c]);
        line += if c < 2 { "," } else { "" };
    }
    println!("{line}");
    save_labels(&result.map(|c| c as u32 + 1), format!("out/pixel-classify/{name}.png"));
}

pub fn main() {
    let dir = "out/pixel-classify";
    let (img, labels) = samples::nuclei(128, 14, 21);
    let truth = true_classes(&labels);
    save_gray(&img, format!("{dir}/image.png"));
    save_labels(&truth.map(|c| c as u32 + 1), format!("{dir}/truth.png"));

    // The feature stack, in "raw" units: grey value and local mean as 12-bit camera counts
    // (0 to 4095), local standard deviation and gradient in units of 0 to 1.
    let (mean, sd) = local_mean_sd(&img);
    let grad = gradient(&img);
    let features = Image::from_fn(128, 128, |x, y| [4095.0 * img[(x, y)], 4095.0 * mean[(x, y)], sd[(x, y)], grad[(x, y)]]);

    // Paint: many background strokes, fewer nucleus and edge strokes.
    let mut rng = Rng::new(5);
    let painted = paint(&truth, [16, 4, 2], &mut rng);
    let mut overlay = gray_to_rgb(&img);
    for &((x, y), c, _) in &painted {
        overlay[(x, y)] = [[0.3, 0.5, 1.0], [1.0, 0.6, 0.1], [0.2, 0.9, 0.3]][c];
    }
    save_rgb(&overlay, format!("{dir}/strokes.png"));
    let raw: Vec<Vec<f32>> = painted.iter().map(|&(p, _, _)| features[p].to_vec()).collect();
    let classes: Vec<usize> = painted.iter().map(|&(_, c, _)| c).collect();
    let strokes: Vec<usize> = painted.iter().map(|&(_, _, s)| s).collect();
    let counts: Vec<usize> = (0..3).map(|c| classes.iter().filter(|&&x| x == c).count()).collect();
    println!("painted {} pixels in 22 strokes: {} background, {} nucleus, {} edge", raw.len(), counts[0], counts[1], counts[2]);

    // Standardise with the painted pixels' statistics.
    let (m, s) = fit_standardise(&raw);
    println!("feature means {:.3?}\n  standard deviations {:.3?}", m, s);
    let z: Vec<Vec<f32>> = raw.iter().map(|x| standardise(x, &m, &s)).collect();
    let z_features = features.map(|f| {
        let z = standardise(&f, &m, &s);
        [z[0], z[1], z[2], z[3]]
    });

    println!("one-vs-rest linear SVMs, C = 1, on all 16384 pixels:");
    let steps = 3000;
    classify_image("raw-features", &train_one_vs_rest(&raw, &classes, 3, 1.0, steps, false), &features, &truth);
    classify_image("standardised", &train_one_vs_rest(&z, &classes, 3, 1.0, steps, false), &z_features, &truth);
    classify_image("standardised-balanced", &train_one_vs_rest(&z, &classes, 3, 1.0, steps, true), &z_features, &truth);

    // Cross-validation with 5 folds, by pixel and by stroke. Standardisation is part of the
    // model, so it is fitted inside every fold, on the training folds only.
    let by_pixel: Vec<usize> = (0..raw.len()).map(|i| i % 5).collect();
    let by_stroke: Vec<usize> = strokes.iter().map(|s| s % 5).collect();
    let svm = |c: f32| {
        move |train: &[Vec<f32>], cl: &[usize], test: &[Vec<f32>]| -> Vec<usize> {
            let (m, s) = fit_standardise(train);
            let zt: Vec<Vec<f32>> = train.iter().map(|x| standardise(x, &m, &s)).collect();
            let models = train_one_vs_rest(&zt, cl, 3, c, 1000, true);
            test.iter().map(|x| predict_one_vs_rest(&models, &standardise(x, &m, &s))).collect()
        }
    };
    let nearest_neighbour = |train: &[Vec<f32>], cl: &[usize], test: &[Vec<f32>]| -> Vec<usize> {
        let (m, s) = fit_standardise(train);
        let zt: Vec<Vec<f32>> = train.iter().map(|x| standardise(x, &m, &s)).collect();
        test.iter()
            .map(|x| {
                let zx = standardise(x, &m, &s);
                let d = |p: &Vec<f32>| p.iter().zip(&zx).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
                cl[(0..zt.len()).min_by(|&a, &b| d(&zt[a]).total_cmp(&d(&zt[b]))).unwrap()]
            })
            .collect()
    };
    println!("5-fold cross-validation error (by pixel / by stroke):");
    for c in [0.01, 0.1, 1.0, 10.0] {
        let (p, s) = (cross_validate(&raw, &classes, &by_pixel, 5, svm(c)), cross_validate(&raw, &classes, &by_stroke, 5, svm(c)));
        println!("  linear SVM, C = {c:<5} {:>5.1} % / {:>5.1} %", 100.0 * p, 100.0 * s);
    }
    let (p, s) = (cross_validate(&raw, &classes, &by_pixel, 5, nearest_neighbour), cross_validate(&raw, &classes, &by_stroke, 5, nearest_neighbour));
    println!("  nearest neighbour       {:>5.1} % / {:>5.1} %", 100.0 * p, 100.0 * s);
    println!("wrote the pictures to {dir}/");
}
