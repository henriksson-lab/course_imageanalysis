// Trains a boosted detector for nuclei in synthetic fluorescence images and runs it as a
// sliding window over a new image. Pictures in out/viola-jones/.
//   cargo run -p integral -- viola-jones                      (your code)
//   cargo run -p integral --features solution -- viola-jones  (the model solution)

use imgcourse::*;
use crate::viola_jones::data::*;
use crate::viola_jones::*;

/// The values of every feature on every window: `values[f][i]`.
fn feature_values(pool: &[Feature], images: &[Image<f32>], windows: &[(usize, usize, usize)]) -> Vec<Vec<f32>> {
    let tables: Vec<_> = images.iter().map(|im| (integral(im, |v| v), integral(im, |v| v * v))).collect();
    pool.iter().map(|f| windows.iter().map(|&(k, x, y)| f.value(&tables[k].0, &tables[k].1, x, y)).collect()).collect()
}

fn error_rate(stumps: &[Stump], values: &[Vec<f32>], labels: &[f64]) -> f64 {
    let wrong = (0..labels.len()).filter(|&i| score(stumps, |f| values[f][i]) * labels[i] <= 0.0).count();
    wrong as f64 / labels.len() as f64
}

pub fn main() {
    let dir = "out/viola-jones";
    let pool = feature_pool();
    let tr = examples(&(1..=8).collect::<Vec<u64>>());
    let te = examples(&(101..=104).collect::<Vec<u64>>());
    let (train_img, train_win, train_lab) = (&tr.images, &tr.windows, &tr.labels);
    let (test_lab, test_win, test_img) = (&te.labels, &te.windows, &te.images);
    let positives = |l: &[f64]| l.iter().filter(|&&v| v > 0.0).count();
    println!("{} features; training set {} windows ({} nuclei), test set {} windows ({} nuclei)", pool.len(), train_lab.len(), positives(train_lab), test_lab.len(), positives(test_lab));
    let train = feature_values(&pool, train_img, train_win);
    let test = feature_values(&pool, test_img, test_win);

    // A sheet of training windows: the first 8 nuclei (top row) and 8 background windows.
    let mut sheet = Image::new(8 * 26, 2 * 26, 1.0f32);
    let (mut np, mut nn) = (0, 0);
    for (i, &(k, x, y)) in train_win.iter().enumerate() {
        let (row, col) = if train_lab[i] > 0.0 { (0, &mut np) } else { (1, &mut nn) };
        if *col < 8 {
            for (dx, dy) in Image::new(WINDOW, WINDOW, 0u8).coords() {
                sheet[(*col * 26 + dx, row * 26 + dy)] = train_img[k][(x + dx, y + dy)];
            }
            *col += 1;
        }
    }
    save_gray(&sheet, format!("{dir}/examples.png"));

    let rounds = 20;
    let stumps = adaboost(&train, train_lab, rounds);
    println!("\nround  error  alpha   feature");
    for (t, s) in stumps.iter().enumerate().take(5) {
        println!("{:5} {:6.3} {:6.2}   {} ({})", t + 1, s.error, s.alpha, pool[s.feature].name(), if s.polarity > 0.0 { "face if above" } else { "face if below" });
    }
    println!("\nstumps  training error  test error");
    for t in [1, 2, 5, 10, 20] {
        println!("{t:6} {:14.1}% {:10.1}%", 100.0 * error_rate(&stumps[..t], &train, train_lab), 100.0 * error_rate(&stumps[..t], &test, test_lab));
    }

    // Detection: every window position (step 2) of a new image, score > 0, and of nearby
    // detections only the best one is kept.
    let (img, labels) = samples::nuclei(SIZE, 12, 7777);
    let truth = centres(&labels);
    let (s1, s2) = (integral(&img, |v| v), integral(&img, |v| v * v));
    let mut hits = Vec::new();
    for y in (0..=SIZE - WINDOW).step_by(2) {
        for x in (0..=SIZE - WINDOW).step_by(2) {
            let sc = score(&stumps, |f| pool[f].value(&s1, &s2, x, y));
            if sc > 0.0 {
                hits.push((sc, x + WINDOW / 2, y + WINDOW / 2));
            }
        }
    }
    hits.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut kept: Vec<(usize, usize)> = Vec::new();
    for &(_, x, y) in &hits {
        if kept.iter().all(|&(kx, ky)| (kx as f32 - x as f32).hypot(ky as f32 - y as f32) > 10.0) {
            kept.push((x, y));
        }
    }
    let near = |(x, y): (usize, usize), (cx, cy): (f32, f32)| (x as f32 - cx).hypot(y as f32 - cy) <= 6.0;
    let found = truth.iter().filter(|&&c| kept.iter().any(|&k| near(k, c))).count();
    let wrong = kept.iter().filter(|&&k| truth.iter().all(|&c| !near(k, c))).count();
    println!("\ndetection on a new image: {} windows scored, {} with score > 0, {} detections after merging", 69 * 69, hits.len(), kept.len());
    println!("{found} of {} nuclei found, {wrong} false detections", truth.len());

    let mut out = gray_to_rgb(&img.map(|v| v.min(1.0)));
    for &(x, y) in &kept {
        let colour = if truth.iter().any(|&c| near((x, y), c)) { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        for d in 0..WINDOW {
            let (x0, y0) = (x - WINDOW / 2, y - WINDOW / 2);
            for (px, py) in [(x0 + d, y0), (x0 + d, y0 + WINDOW - 1), (x0, y0 + d), (x0 + WINDOW - 1, y0 + d)] {
                out[(px, py)] = colour;
            }
        }
    }
    save_rgb(&out, format!("{dir}/detections.png"));

    // The cascade of the lecture: ten stages, each keeping 99 % of the faces and 30 % of the
    // background, with 2, 5, 10, 20, ... features.
    let stages: Vec<Stage> = [2, 5, 10, 20, 20, 30, 30, 40, 50, 50].iter().map(|&n| Stage { d: 0.99, f: 0.3, n }).collect();
    let (d, f, n) = cascade_rates(&stages);
    let total: usize = stages.iter().map(|s| s.n).sum();
    println!("\ncascade of 10 stages: D = {d:.3}, F = {f:.1e}, {n:.2} features per background window ({total} in the cascade)");
    println!("wrote the results to {dir}/");
}
