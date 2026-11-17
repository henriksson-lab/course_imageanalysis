// Evaluates thresholds of synthetic nuclei against their ground truth: many nuclei, and few
// (rare objects); results in out/seg-eval/.
//   cargo run -p seg-eval                      (your code)
//   cargo run -p seg-eval --features solution  (the model solution)

use imgcourse::*;
use seg_eval::*;

/// TP green, FP red, FN blue, TN grey (as on the slide "The confusion matrix").
fn overlay(pred: &Image<bool>, truth: &Image<bool>) -> Image<[f32; 3]> {
    pred.zip_map(truth, |p, t| match (p, t) {
        (true, true) => [0.2, 0.8, 0.2],
        (true, false) => [0.9, 0.15, 0.15],
        (false, true) => [0.2, 0.4, 1.0],
        (false, false) => [0.3, 0.3, 0.3],
    })
}

/// A 201 × 201 plot of a curve through the points (x, y) in [0, 1]², y up, with the diagonal.
fn plot(points: &[(f32, f32)]) -> Image<f32> {
    let size = 201;
    let mut img = Image::from_fn(size, size, |x, y| if x == size - 1 - y { 0.8 } else { 1.0 });
    for w in points.windows(2) {
        for k in 0..=50 {
            let s = k as f32 / 50.0;
            let (x, y) = (w[0].0 + s * (w[1].0 - w[0].0), w[0].1 + s * (w[1].1 - w[0].1));
            if x.is_finite() && y.is_finite() {
                img[((x * 200.0).round() as usize, 200 - (y * 200.0).round() as usize)] = 0.0;
            }
        }
    }
    img
}

fn report(name: &str, img: &Image<f32>, truth: &Image<bool>) {
    let dir = "out/seg-eval";
    let n_obj = truth.pixels().iter().filter(|&&t| t).count();
    println!("{name}: {:.1} % of the pixels are object", 100.0 * n_obj as f32 / truth.pixels().len() as f32);
    let nothing = confusion(&img.map(|_| false), truth);
    println!("  all background:  accuracy {:.3}, Dice {:.3}", nothing.accuracy(), nothing.dice());
    let (t, _) = best_dice(img, truth, 256);
    let t_mean = bin_of(img.mean(), 256);
    for (label, t) in [("t = mean", t_mean), ("best Dice", t)] {
        let pred = img.map(|v| bin_of(v, 256) > t);
        let c = confusion(&pred, truth);
        println!(
            "  {label:<10} (t = {t:3}): TP {:5} FP {:5} FN {:5} TN {:5}; precision {:.3}, recall {:.3}, FPR {:.3}, accuracy {:.3}, Dice {:.3}, Jaccard {:.3}",
            c.tp, c.fp, c.fn_, c.tn, c.precision(), c.recall(), c.fpr(), c.accuracy(), c.dice(), c.jaccard()
        );
        save_rgb(&overlay(&pred, truth), format!("{dir}/{name}-t{t}.png"));
    }
    let points = roc(img, truth, 256);
    println!("  AUC {:.3}", auc(&points));
    save_gray(&plot(&points), format!("{dir}/{name}-roc.png"));
    // precision–recall curve: precision = TP / (TP + FP) from the same two rates
    let n_bg = (truth.pixels().len() - n_obj) as f32;
    let pr: Vec<(f32, f32)> = points
        .iter()
        .filter(|&&(f, r)| f > 0.0 || r > 0.0)
        .map(|&(f, r)| (r, r * n_obj as f32 / (r * n_obj as f32 + f * n_bg)))
        .collect();
    save_gray(&plot(&pr), format!("{dir}/{name}-pr.png"));
}

fn main() {
    let dir = "out/seg-eval";
    for (name, n) in [("many", 25), ("few", 3)] {
        let (img, labels) = samples::nuclei(256, n, 4);
        // less contrast and more noise, so that the classes overlap
        let img = samples::add_noise(&img.map(|v| 0.3 + 0.4 * v), 0.08, 5);
        let truth = labels.map(|l| l > 0);
        save_gray(&img, format!("{dir}/{name}.png"));
        report(name, &img, &truth);
    }
    println!("wrote the results to {dir}/");
}
