// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day34 pixel_classify::

use imgcourse::Rng;
use day34::pixel_classify::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_standardise_b1() {
    // B1: class A (1000, 0.2), (3000, 0.3); class B (1300, 0.8), (2900, 0.9).
    let train = vec![vec![1000.0, 0.2], vec![3000.0, 0.3], vec![1300.0, 0.8], vec![2900.0, 0.9]];
    let (mean, sd) = fit_standardise(&train);
    assert!(close(mean[0], 2050.0, 0.01) && close(mean[1], 0.55, 1e-5), "B1(b) means: got {mean:?}, expected [2050, 0.55]");
    assert!(close(sd[0], 906.9, 0.1) && close(sd[1], 0.304, 1e-3), "B1(b) standard deviations: got {sd:?}, expected [906.9, 0.304]");
    let z = standardise(&[2080.0, 0.3], &mean, &sd);
    assert!(close(z[0], 0.033, 1e-3) && close(z[1], -0.822, 1e-3), "B1(b) test pixel: got {z:?}, expected [0.033, −0.822]");
    let za = standardise(&[2000.0, 0.25], &mean, &sd);
    assert!(close(za[0], -0.055, 1e-3) && close(za[1], -0.986, 1e-3), "B1(b) mean of class A: got {za:?}, expected [−0.055, −0.986]");
}

#[test]
fn c2_class_weights_b4() {
    let mut classes = vec![0; 2000];
    classes.extend(vec![1; 100]);
    let w = class_weights(&classes, 2);
    assert!(close(w[0], 0.525, 1e-5) && close(w[1], 10.5, 1e-4), "B4: got {w:?}, expected [0.525, 10.5]");
}

#[test]
fn c3_linear_svm_lecture_example() {
    // The maxmargin unit's example, all weights 1: w = (1, 0), b = −1.
    let points = vec![vec![0.0, 0.0], vec![0.0, 2.0], vec![2.0, 1.0], vec![3.0, 0.0]];
    let m = train_linear_svm(&points, &[-1.0, -1.0, 1.0, 1.0], &[1.0; 4], 10.0, 20000);
    assert!(close(m.w[0], 1.0, 0.03) && close(m.w[1], 0.0, 0.03) && close(m.b, -1.0, 0.03), "got w = {:?}, b = {}, expected (1, 0), −1", m.w, m.b);
}

#[test]
fn c3_weights_move_the_boundary() {
    // 40 background points at 0, 0.1, ..., 3.9 and 4 rare points at 3, 3.5, 4, 4.5 (1D).
    let mut points: Vec<Vec<f32>> = (0..40).map(|i| vec![i as f32 * 0.1]).collect();
    let mut labels = vec![-1.0; 40];
    for x in [3.0, 3.5, 4.0, 4.5] {
        points.push(vec![x]);
        labels.push(1.0);
    }
    let classes: Vec<usize> = labels.iter().map(|&y| if y > 0.0 { 1 } else { 0 }).collect();
    let cw = class_weights(&classes, 2);
    let balanced: Vec<f32> = classes.iter().map(|&c| cw[c]).collect();
    let plain = train_linear_svm(&points, &labels, &[1.0; 44], 1.0, 20000);
    let weighted = train_linear_svm(&points, &labels, &balanced, 1.0, 20000);
    let (t_plain, t_weighted) = (-plain.b / plain.w[0], -weighted.b / weighted.w[0]);
    let found = |m: &Linear| [3.0, 3.5, 4.0, 4.5].iter().filter(|&&x| m.f(&[x]) > 0.0).count();
    assert!(t_weighted < t_plain - 0.1, "balanced weights should move the threshold into the large class: plain {t_plain}, balanced {t_weighted}");
    assert!(found(&weighted) >= found(&plain), "balanced weights should find at least as many rare points: plain {}, balanced {}", found(&plain), found(&weighted));
}

#[test]
fn c4_predict_one_vs_rest_b2() {
    // B2: models that ignore x and return the f values of the table (classes 1, 2, 3 are 0, 1, 2 here).
    let table = [[1.2, -0.8, -1.5], [0.4, 0.6, -1.0], [-0.3, -0.7, -0.2], [-1.1, 0.9, -0.4]];
    let expected = [0, 1, 2, 1];
    for (name, (row, e)) in ["p", "q", "r", "s"].iter().zip(table.iter().zip(expected)) {
        let models: Vec<Linear> = row.iter().map(|&b| Linear { w: vec![], b }).collect();
        let got = predict_one_vs_rest(&models, &[]);
        assert_eq!(got, e, "B2, pixel {name}: got class {}, expected class {}", got + 1, e + 1);
    }
}

#[test]
fn c4_train_one_vs_rest_three_clusters() {
    let mut rng = Rng::new(3);
    let (mut points, mut classes) = (Vec::new(), Vec::new());
    for (c, (cx, cy)) in [(0.0, 0.0), (4.0, 0.0), (2.0, 4.0)].iter().enumerate() {
        for _ in 0..15 {
            points.push(vec![cx + 0.4 * rng.normal(), cy + 0.4 * rng.normal()]);
            classes.push(c);
        }
    }
    let models = train_one_vs_rest(&points, &classes, 3, 1.0, 5000, false);
    let wrong = (0..points.len()).filter(|&i| predict_one_vs_rest(&models, &points[i]) != classes[i]).count();
    assert_eq!(wrong, 0, "three separate clusters: expected 0 errors, got {wrong}");
}

#[test]
fn c5_cross_validate_counts() {
    // Predict the most common class of the training folds. Holding out the single class-1
    // point gives the only error: 1 of 4.
    let points = vec![vec![0.0]; 4];
    let classes = [0, 0, 0, 1];
    let majority = |_: &[Vec<f32>], c: &[usize], test: &[Vec<f32>]| {
        let ones = c.iter().filter(|&&x| x == 1).count();
        vec![if 2 * ones > c.len() { 1 } else { 0 }; test.len()]
    };
    let e = cross_validate(&points, &classes, &[0, 1, 2, 3], 4, majority);
    assert!(close(e, 0.25, 1e-6), "got {e}, expected 0.25");
}

#[test]
fn c5_folds_by_pixel_are_optimistic_b3() {
    // 20 strokes of 6 nearly identical pixels each, with a random class per stroke: nothing
    // can be learnt. 1-nearest-neighbour looks perfect with folds by pixel, not by stroke.
    let mut rng = Rng::new(9);
    let (mut points, mut classes, mut strokes) = (Vec::new(), Vec::new(), Vec::new());
    for s in 0..20 {
        let (cx, cy, c) = (rng.uniform(), rng.uniform(), rng.below(2));
        for _ in 0..6 {
            points.push(vec![cx + 0.001 * rng.normal(), cy + 0.001 * rng.normal()]);
            classes.push(c);
            strokes.push(s);
        }
    }
    let nearest = |train: &[Vec<f32>], c: &[usize], test: &[Vec<f32>]| -> Vec<usize> {
        test.iter()
            .map(|x| {
                let d = |p: &Vec<f32>| (p[0] - x[0]).powi(2) + (p[1] - x[1]).powi(2);
                let best = (0..train.len()).min_by(|&a, &b| d(&train[a]).total_cmp(&d(&train[b]))).unwrap();
                c[best]
            })
            .collect()
    };
    let by_pixel: Vec<usize> = (0..points.len()).map(|i| i % 5).collect();
    let by_stroke: Vec<usize> = strokes.iter().map(|s| s % 5).collect();
    let e_pixel = cross_validate(&points, &classes, &by_pixel, 5, nearest);
    let e_stroke = cross_validate(&points, &classes, &by_stroke, 5, nearest);
    assert!(e_pixel < 0.05, "folds by pixel: expected an error near 0, got {e_pixel}");
    assert!(e_stroke > 0.25, "folds by stroke: expected an error near 0.5 (random classes), got {e_stroke}");
}
