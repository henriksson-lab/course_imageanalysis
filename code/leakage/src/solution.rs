// Model solution for the coding exercises of "Honest validation: leakage, splits and domain shift".

use imgcourse::{Image, Rng};

/// C1. k nearest neighbours: the `k` training samples closest to `query` (Euclidean
/// distance of the feature vectors) vote; the majority wins. Use an odd `k`.
pub fn knn_predict(train_x: &[Vec<f32>], train_y: &[bool], query: &[f32], k: usize) -> bool {
    // the k best so far, as (squared distance, label), sorted from near to far
    let mut best: Vec<(f32, bool)> = Vec::with_capacity(k + 1);
    for (x, &y) in train_x.iter().zip(train_y) {
        let mut d = 0.0;
        for i in 0..query.len() {
            d += (x[i] - query[i]) * (x[i] - query[i]);
        }
        if best.len() < k || d < best[best.len() - 1].0 {
            // insert in order, drop the farthest if there are now k + 1
            let pos = best.iter().position(|b| b.0 > d).unwrap_or(best.len());
            best.insert(pos, (d, y));
            best.truncate(k);
        }
    }
    let yes = best.iter().filter(|b| b.1).count();
    2 * yes > best.len()
}

/// C2. Random blocks: cut a `w × h` image into `block × block` squares and give every
/// square a random fold `0..folds`. With `block = 1` every pixel gets its own random fold.
pub fn block_folds(w: usize, h: usize, block: usize, folds: usize, seed: u64) -> Image<usize> {
    let mut rng = Rng::new(seed);
    let (bw, bh) = (w.div_ceil(block), h.div_ceil(block));
    let mut fold_of_block = Vec::with_capacity(bw * bh);
    for _ in 0..bw * bh {
        fold_of_block.push(rng.below(folds));
    }
    Image::from_fn(w, h, |x, y| fold_of_block[(y / block) * bw + x / block])
}

/// The role of a pixel in one fold of a cross-validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Train,
    Test,
    Buffer,
}

/// C3. The roles of the pixels when `test_fold` is tested: its pixels are Test; any other
/// pixel within `buffer` pixels of a Test pixel (in x and in y) is Buffer, used for
/// nothing; the rest is Train.
pub fn roles(folds: &Image<usize>, test_fold: usize, buffer: usize) -> Image<Role> {
    let b = buffer as isize;
    folds.map_xy(|x, y, f| {
        if f == test_fold {
            return Role::Test;
        }
        for dy in -b..=b {
            for dx in -b..=b {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if folds.contains(nx, ny) && folds[(nx as usize, ny as usize)] == test_fold {
                    return Role::Buffer;
                }
            }
        }
        Role::Train
    })
}

/// C4. "Peeking at the test set", simulated: `k` candidate methods all have the true
/// accuracy `accuracy`. In each of `runs` studies, test every candidate on `n` objects
/// (each object is right with probability `accuracy`), report the best score, and re-test
/// that candidate on `n` fresh objects. Returns (mean reported, mean fresh).
pub fn best_of_k(k: usize, n: usize, accuracy: f32, runs: usize, seed: u64) -> (f32, f32) {
    let mut rng = Rng::new(seed);
    let test = |rng: &mut Rng| {
        let mut right = 0;
        for _ in 0..n {
            if rng.uniform() < accuracy {
                right += 1;
            }
        }
        right as f32 / n as f32
    };
    let (mut reported, mut fresh) = (0.0, 0.0);
    for _ in 0..runs {
        let mut best = 0.0f32;
        for _ in 0..k {
            best = best.max(test(&mut rng));
        }
        reported += best;
        fresh += test(&mut rng); // all candidates are equally good: any fresh test will do
    }
    (reported / runs as f32, fresh / runs as f32)
}
