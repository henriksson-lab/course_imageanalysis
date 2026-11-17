// Grows one deep tree, bagged trees and random forests on two overlapping clouds of points;
// results in out/forest/.
//   cargo run -p day33 -- forest                      (your code)
//   cargo run -p day33 --features solution -- forest  (the model solution)

use crate::forest::*;
use imgcourse::*;

/// `n` points of two overlapping round clouds (class 0 around (0, 0), class 1 around
/// (1.5, 1.5), standard deviation 1), with the derived features x + y and x − y.
fn clouds(n: usize, rng: &mut Rng) -> (Vec<Vec<f32>>, Vec<usize>) {
    let (mut points, mut labels) = (Vec::new(), Vec::new());
    for i in 0..n {
        let class = i % 2;
        let c = 1.5 * class as f32;
        let (x, y) = (c + rng.normal(), c + rng.normal());
        points.push(vec![x, y, x + y, x - y]);
        labels.push(class);
    }
    (points, labels)
}

/// The first `k` trees of a forest, as a forest of their own.
fn first(forest: &Forest, k: usize) -> Forest {
    Forest { trees: forest.trees[..k].to_vec(), in_bag: forest.in_bag[..k].to_vec() }
}

pub fn main() {
    let dir = "out/forest";
    let mut rng = Rng::new(1);
    let (train4, train_labels) = clouds(200, &mut rng);
    let (test4, test_labels) = clouds(1000, &mut rng);
    // the first two features, x and y
    let train: Vec<Vec<f32>> = train4.iter().map(|p| p[..2].to_vec()).collect();
    let test: Vec<Vec<f32>> = test4.iter().map(|p| p[..2].to_vec()).collect();
    let pct = |e: f32| format!("{:.1} %", 100.0 * e);

    // One deep tree on all training points (no bootstrap, both features at every split).
    let all: Vec<usize> = (0..train.len()).collect();
    let tree = grow_tree(&train, &train_labels, &all, 2, &mut || vec![0, 1]);
    let one = [tree];
    println!(
        "one deep tree: training error {}, test error {}",
        pct(error_rate(&one, &train, &train_labels, 2)),
        pct(error_rate(&one, &test, &test_labels, 2))
    );

    // Bagging: 100 trees on bootstrap samples (m = d = 2).
    let start = std::time::Instant::now();
    let bagged = train_forest(&train, &train_labels, 2, 100, 2, &mut rng);
    let ms = start.elapsed().as_secs_f64() * 1e3;
    println!("bagging, 100 trees ({ms:.0} ms):");
    for k in [1, 10, 50, 100] {
        let f = first(&bagged, k);
        println!(
            "  {k:3} trees: training {}, OOB {}, test {}",
            pct(error_rate(&f.trees, &train, &train_labels, 2)),
            pct(oob_error(&f, &train, &train_labels, 2)),
            pct(error_rate(&f.trees, &test, &test_labels, 2))
        );
    }

    // The same points with the derived features x + y and x − y (d = 4): m = 4, 2, 1.
    for m in [4, 2, 1] {
        let f = train_forest(&train4, &train_labels, 2, 100, m, &mut rng);
        println!(
            "d = 4 (x, y, x + y, x − y), m = {m}: OOB {}, test {}, one tree alone {}",
            pct(oob_error(&f, &train4, &train_labels, 2)),
            pct(error_rate(&f.trees, &test4, &test_labels, 2)),
            pct(error_rate(&f.trees[..1], &test4, &test_labels, 2))
        );
    }

    // Probability maps over the square −3 .. 4.5 in both features, with the training points:
    // blue = class 0, orange = class 1, pale = uncertain.
    let size = 300;
    let to_feature = |p: usize| -3.0 + 7.5 * p as f32 / size as f32;
    let colour = |p1: f32| {
        let (blue, orange) = ([0.2, 0.45, 0.9], [0.95, 0.55, 0.1]);
        let conf = (2.0 * (p1 - 0.5)).abs();
        let c = if p1 > 0.5 { orange } else { blue };
        [1.0 - conf * (1.0 - c[0]), 1.0 - conf * (1.0 - c[1]), 1.0 - conf * (1.0 - c[2])]
    };
    for (name, trees) in [("one-tree", &one[..]), ("bagged-100", &bagged.trees[..])] {
        let mut map = Image::from_fn(size, size, |x, y| colour(vote(trees, &[to_feature(x), to_feature(size - 1 - y)], 2)[1]));
        for (p, &l) in train.iter().zip(&train_labels) {
            let px = ((p[0] + 3.0) / 7.5 * size as f32) as isize;
            let py = size as isize - 1 - ((p[1] + 3.0) / 7.5 * size as f32) as isize;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                if map.contains(px + dx, py + dy) {
                    map[((px + dx) as usize, (py + dy) as usize)] = if l == 0 { [0.0, 0.1, 0.5] } else { [0.6, 0.2, 0.0] };
                }
            }
        }
        save_rgb(&map, format!("{dir}/probability-{name}.png"));
    }
    for corner in [[4.5, 4.5], [-3.0, -3.0], [-3.0, 4.5]] {
        let p = vote(&bagged.trees, &corner, 2);
        println!("far from the data, at ({}, {}): P(class 1) = {:.2}", corner[0], corner[1], p[1]);
    }
    println!("wrote the results to {dir}/");
}
