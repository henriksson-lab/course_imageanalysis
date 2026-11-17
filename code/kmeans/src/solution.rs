// Model solution for the coding exercises of "Feature spaces and k-means".

use imgcourse::Rng;

/// The squared distance `‖a − b‖²` of two points with any number of features. (Given.)
pub fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// C1. The assign step (slide "Lloyd's algorithm"): every point goes to its nearest centre
/// (the first one, if two are equally near). Returns the group of every point and
/// `J = Σ_i ‖x_i − m_c(i)‖²`.
pub fn assign(points: &[Vec<f32>], centres: &[Vec<f32>]) -> (Vec<usize>, f32) {
    let mut groups = Vec::new();
    let mut j = 0.0;
    for x in points {
        let mut best = 0;
        for k in 1..centres.len() {
            if dist2(x, &centres[k]) < dist2(x, &centres[best]) {
                best = k;
            }
        }
        j += dist2(x, &centres[best]);
        groups.push(best);
    }
    (groups, j)
}

/// C2. The update step: every centre moves to the mean of the points assigned to it. A centre
/// without points stays where it is.
pub fn update(points: &[Vec<f32>], groups: &[usize], centres: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let d = points[0].len();
    let mut sums = vec![vec![0.0; d]; centres.len()];
    let mut counts = vec![0.0; centres.len()];
    for (x, &g) in points.iter().zip(groups) {
        for j in 0..d {
            sums[g][j] += x[j];
        }
        counts[g] += 1.0;
    }
    (0..centres.len())
        .map(|k| if counts[k] > 0.0 { sums[k].iter().map(|s| s / counts[k]).collect() } else { centres[k].clone() })
        .collect()
}

/// C3. k-means (Lloyd's algorithm) from the given starting centres: assign, update, repeat,
/// until no point changes its group (or after `max_iterations`). Returns the final centres,
/// the groups and the value of `J` after every assign step (slide "What k-means minimises").
pub fn kmeans(points: &[Vec<f32>], start: &[Vec<f32>], max_iterations: usize) -> (Vec<Vec<f32>>, Vec<usize>, Vec<f32>) {
    let mut centres = start.to_vec();
    let mut groups: Vec<usize> = Vec::new();
    let mut trace = Vec::new();
    for _ in 0..max_iterations {
        let (new_groups, j) = assign(points, &centres);
        trace.push(j);
        if new_groups == groups {
            break;
        }
        groups = new_groups;
        centres = update(points, &groups, &centres);
    }
    (centres, groups, trace)
}

/// C4a. The k-means++ probabilities (slide "Better starts: k-means++"): every point's squared
/// distance `D²` to its nearest chosen centre, divided by the sum of all of them.
pub fn kmeanspp_probabilities(points: &[Vec<f32>], chosen: &[Vec<f32>]) -> Vec<f32> {
    let d2: Vec<f32> = points.iter().map(|x| chosen.iter().map(|c| dist2(x, c)).fold(f32::INFINITY, f32::min)).collect();
    let total: f32 = d2.iter().sum();
    d2.iter().map(|d| d / total).collect()
}

/// C4b. k-means++ seeding: the first centre is a point chosen at random; every next centre is
/// a point chosen at random with the probabilities of C4a.
pub fn kmeanspp(points: &[Vec<f32>], k: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    let mut centres = vec![points[rng.below(points.len())].clone()];
    while centres.len() < k {
        let p = kmeanspp_probabilities(points, &centres);
        // walk along the cumulative probabilities until they pass a uniform random number
        let u = rng.uniform();
        let mut sum = 0.0;
        let mut pick = points.len() - 1;
        for (i, pi) in p.iter().enumerate() {
            sum += pi;
            if u < sum {
                pick = i;
                break;
            }
        }
        centres.push(points[pick].clone());
    }
    centres
}
