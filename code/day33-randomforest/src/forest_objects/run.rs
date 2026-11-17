// Feature importance of a forest on a table of 400 synthetic cells; prints the results.
//   cargo run -p day33 -- forest-objects                      (your code)
//   cargo run -p day33 --features solution -- forest-objects  (the model solution)

use crate::forest_objects::*;
use imgcourse::Rng;

const NAMES: [&str; 8] = ["area", "perimeter", "diameter", "intensity", "texture", "solidity", "noise", "coin"];

/// 400 segmented cells, 200 interphase (class 0) and 200 mitotic (class 1), with eight
/// measurements each. Mitotic cells are a little smaller and brighter, and much
/// more textured; area, perimeter and diameter all measure size; noise and coin are random.
fn cell_table(rng: &mut Rng) -> (Vec<Vec<f32>>, Vec<usize>) {
    let (mut rows, mut labels) = (Vec::new(), Vec::new());
    for i in 0..400 {
        let c = (i % 2) as f32;
        let area = 300.0 - 80.0 * c + 50.0 * rng.normal();
        let perimeter = 2.0 * (std::f32::consts::PI * area).sqrt() * (1.08 + 0.02 * rng.normal());
        let diameter = 2.0 * (area / std::f32::consts::PI).sqrt();
        let intensity = 0.5 + 0.05 * c + 0.1 * rng.normal();
        let texture = 8.2 + 5.9 * c + 3.0 * rng.normal();
        let solidity = 0.93 + 0.01 * c + 0.03 * rng.normal();
        let noise = rng.uniform();
        let coin = (rng.uniform() < 0.5) as u8 as f32;
        rows.push(vec![area, perimeter, diameter, intensity, texture, solidity, noise, coin]);
        labels.push(i % 2);
    }
    (rows, labels)
}

/// Keep only the columns `keep` of every row.
fn select(rows: &[Vec<f32>], keep: &[usize]) -> Vec<Vec<f32>> {
    rows.iter().map(|r| keep.iter().map(|&j| r[j]).collect()).collect()
}

/// Train a forest on the columns `keep` and print both importances of every feature.
fn report(title: &str, rows: &[Vec<f32>], labels: &[usize], keep: &[usize], names: &[&str], groups: &[(&str, Vec<usize>)]) {
    let points = select(rows, keep);
    let settings = Settings { n_trees: 100, m: 3, class_weights: vec![1.0, 1.0], seed: 1 };
    let forest = Forest::train(&points, labels, 2, &settings);
    let mdi = impurity_importance(&forest.trees, keep.len());
    let total: f32 = mdi.iter().sum();
    let mut rng = Rng::new(2);
    println!("\n{title}: OOB error {:.1} %", 100.0 * forest.oob_error(&points, labels));
    println!("  feature      impurity  permutation");
    for j in 0..keep.len() {
        let pi = permutation_importance(&forest, &points, labels, &[j], 5, &mut rng);
        println!("  {:<11} {:7.0} %  {:+8.1} points", names[j], 100.0 * mdi[j] / total, 100.0 * pi);
    }
    for (name, columns) in groups {
        let pi = permutation_importance(&forest, &points, labels, columns, 5, &mut rng);
        println!("  {name:<11}           {:+8.1} points (shuffled together)", 100.0 * pi);
    }
}

pub fn main() {
    let start = std::time::Instant::now();
    let mut rng = Rng::new(7);
    let (rows, labels) = cell_table(&mut rng);
    let column = |j: usize| rows.iter().map(|r| r[j]).collect::<Vec<f32>>();
    println!("correlations: area–perimeter {:.2}, area–diameter {:.2}, area–texture {:.2}",
        correlation(&column(0), &column(1)), correlation(&column(0), &column(2)), correlation(&column(0), &column(4)));

    // 1. All eight features: texture decides.
    report("all eight features", &rows, &labels, &[0, 1, 2, 3, 4, 5, 6, 7], &NAMES, &[]);

    // 2. Texture left out: the size block shares (or hides) its importance.
    let keep = [0, 1, 2, 3, 5, 6, 7];
    let names: Vec<&str> = keep.iter().map(|&j| NAMES[j]).collect();
    report("without texture", &rows, &labels, &keep, &names, &[("size block", vec![0, 1, 2])]);

    // 3. Area with an exact copy of itself (and no other size feature, no texture).
    let with_copy: Vec<Vec<f32>> = rows.iter().map(|r| vec![r[0], r[3], r[5], r[6], r[7], r[0]]).collect();
    report("area and its copy", &with_copy, &labels, &[0, 1, 2, 3, 4, 5], &["area", "intensity", "solidity", "noise", "coin", "area copy"], &[]);
    report("area without the copy", &with_copy, &labels, &[0, 1, 2, 3, 4], &["area", "intensity", "solidity", "noise", "coin"], &[]);
    println!("\n(took {:.1} s)", start.elapsed().as_secs_f64());
}
