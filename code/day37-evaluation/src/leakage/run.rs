// A k-nearest-neighbour pixel classifier on 8 small synthetic images from 4 days, scored
// with five ways to split, once with real labels and once with random labels ("nothing to
// learn"); then the optimism of reporting the best of k test scores. Pictures in out/leakage/.
//   cargo run -p day37 -- leakage                      (your code)
//   cargo run -p day37 --features solution -- leakage  (the model solution)

use imgcourse::*;
use crate::leakage::*;

const SIZE: usize = 48;
const K: usize = 3;

fn box_mean(img: &Image<f32>, r: isize) -> Image<f32> {
    let n = ((2 * r + 1) * (2 * r + 1)) as f32;
    img.map_xy(|x, y, _| {
        let mut s = 0.0;
        for dy in -r..=r {
            for dx in -r..=r {
                s += img.get_border(x as isize + dx, y as isize + dy, Border::Mirror);
            }
        }
        s / n
    })
}

/// The feature vector of every pixel: the means of the 3 × 3, 5 × 5, 9 × 9 and 17 × 17
/// windows around it, and the slopes across and down (differences of 5 × 5 means 3 pixels
/// to either side).
fn features(img: &Image<f32>) -> Vec<Vec<f32>> {
    let m: Vec<Image<f32>> = [1, 2, 4, 8].iter().map(|&r| box_mean(img, r)).collect();
    let at = |x: usize, y: usize, dx: isize, dy: isize| m[1].get_border(x as isize + dx, y as isize + dy, Border::Clamp);
    img.coords()
        .map(|(x, y)| vec![m[0][(x, y)], m[1][(x, y)], m[2][(x, y)], m[3][(x, y)], at(x, y, 3, 0) - at(x, y, -3, 0), at(x, y, 0, 3) - at(x, y, 0, -3)])
        .collect()
}

/// Mean accuracy over the folds; `fold_maps[i]` gives the fold of every pixel of image `i`.
/// Every 6th test pixel is classified (to save time).
fn cross_validate(feats: &[Vec<Vec<f32>>], labels: &[Vec<bool>], fold_maps: &[Image<usize>], folds: usize, buffer: usize) -> f32 {
    let mut accuracies = Vec::new();
    for test in 0..folds {
        let (mut train_x, mut train_y) = (Vec::new(), Vec::new());
        let mut queries = Vec::new();
        for i in 0..feats.len() {
            let r = roles(&fold_maps[i], test, buffer);
            for (p, (x, y)) in r.coords().enumerate() {
                match r[(x, y)] {
                    Role::Train => {
                        train_x.push(feats[i][p].clone());
                        train_y.push(labels[i][p]);
                    }
                    Role::Test if p % 6 == 0 => queries.push((i, p)),
                    _ => {}
                }
            }
        }
        if queries.is_empty() || train_x.is_empty() {
            continue;
        }
        let right = queries.iter().filter(|&&(i, p)| knn_predict(&train_x, &train_y, &feats[i][p], K) == labels[i][p]).count();
        accuracies.push(right as f32 / queries.len() as f32);
    }
    accuracies.iter().sum::<f32>() / accuracies.len() as f32
}

pub fn main() {
    let dir = "out/leakage";
    // 8 images, 2 per day; each day has its own brightness and contrast.
    let day_look = [(1.0, 0.0), (0.8, 0.1), (1.2, -0.05), (0.6, 0.45)];
    let mut images = Vec::new();
    let mut truth = Vec::new();
    let mut random_labels = Vec::new();
    for i in 0..8 {
        let day = i / 2;
        // a 48 × 48 piece of a larger image, so that the nuclei are 10 to 18 px across
        let (big, big_labels) = samples::nuclei(128, 25, 100 + i as u64);
        let (img, lab) = (big.crop(40, 40, SIZE, SIZE), big_labels.crop(40, 40, SIZE, SIZE));
        let (gain, shift) = day_look[day];
        images.push(img.map(|v| gain * v + shift));
        truth.push(lab.pixels().iter().map(|&l| l > 0).collect::<Vec<bool>>());
        // random blobs, unrelated to the image: smoothed noise above its median
        let noise = box_mean(&box_mean(&samples::add_noise(&Image::new(SIZE, SIZE, 0.0), 1.0, 500 + i as u64), 3), 3);
        let mut sorted = noise.pixels().to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = sorted[sorted.len() / 2];
        random_labels.push(noise.pixels().iter().map(|&v| v > median).collect::<Vec<bool>>());
    }
    let feats: Vec<Vec<Vec<f32>>> = images.iter().map(features).collect();
    // "Nothing to learn": 8 pieces of the textured micrograph with the random labels.
    let cell = samples::cell();
    let pieces: Vec<Image<f32>> = (0..8).map(|i| cell.crop((i % 4) * 60 + 8, (i / 4) * 120 + 30, SIZE, SIZE)).collect();
    let piece_feats: Vec<Vec<Vec<f32>>> = pieces.iter().map(features).collect();

    // A sheet of the 8 images (rows: days) and their labels.
    let sheet = Image::from_fn(4 * (SIZE + 2), 4 * (SIZE + 2), |x, y| {
        let (col, row) = (x / (SIZE + 2), y / (SIZE + 2));
        let (u, v) = (x % (SIZE + 2), y % (SIZE + 2));
        if u >= SIZE || v >= SIZE {
            return 1.0;
        }
        let i = row * 2 + col % 2;
        match col / 2 {
            0 => images[i][(u, v)].clamp(0.0, 1.0),
            _ => if truth[i][v * SIZE + u] { 1.0 } else { 0.0 },
        }
    });
    save_gray(&sheet, format!("{dir}/images-and-truth.png"));

    let folds = 4;
    let splits: Vec<(&str, Vec<Image<usize>>, usize)> = vec![
        ("random pixels", (0..8).map(|i| block_folds(SIZE, SIZE, 1, folds, i)).collect(), 0),
        ("random 8 × 8 blocks", (0..8).map(|i| block_folds(SIZE, SIZE, 8, folds, i)).collect(), 0),
        ("blocks + 4 px buffer", (0..8).map(|i| block_folds(SIZE, SIZE, 8, folds, i)).collect(), 4),
        ("by image", (0..8).map(|i| Image::new(SIZE, SIZE, i % folds)).collect(), 0),
        ("by day", (0..8).map(|i| Image::new(SIZE, SIZE, i / 2)).collect(), 0),
    ];
    let background: usize = truth.iter().map(|t| t.iter().filter(|&&v| !v).count()).sum();
    println!("always answering background: {:.1} %", 100.0 * background as f32 / (8 * SIZE * SIZE) as f32);
    println!("split                   nuclei, real labels   micrograph, random labels (chance 50 %)");
    for (name, maps, buffer) in &splits {
        let real = cross_validate(&feats, &truth, maps, folds, *buffer);
        let nothing = cross_validate(&piece_feats, &random_labels, maps, folds, *buffer);
        println!("{name:22}  {:5.1} %               {:5.1} %", 100.0 * real, 100.0 * nothing);
    }

    // The roles in one fold of the block split with buffer, for the first image.
    let r = roles(&splits[2].1[0], 0, 4);
    let pic = r.map(|role| match role {
        Role::Test => [0.9, 0.2, 0.2],
        Role::Train => [0.2, 0.4, 0.9],
        Role::Buffer => [0.75, 0.75, 0.75],
    });
    save_rgb(&Image::from_fn(4 * SIZE, 4 * SIZE, |x, y| pic[(x / 4, y / 4)]), format!("{dir}/roles-blocks-buffer.png"));

    println!("\nbest of k test scores, true accuracy 0.80 for every candidate (1000 studies):");
    for (k, n) in [(1, 100), (10, 100), (50, 100), (50, 1000)] {
        let (reported, fresh) = best_of_k(k, n, 0.8, 1000, 9);
        println!("  k = {k:2}, n = {n:4}: reported {:.1} %, fresh {:.1} %, optimism {:.1} points", 100.0 * reported, 100.0 * fresh, 100.0 * (reported - fresh));
    }
    println!("wrote the pictures to {dir}/");
}
