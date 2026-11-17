// Model solution for the coding exercises of "Energy and probability: the Ising model".

use imgcourse::Image;

/// A lattice of spins, `+1` or `-1`, one per pixel.
pub type Spins = Image<i8>;

/// State number `index` of a `width × height` lattice: pixel `i` (row by row) is `+1` if bit
/// `i` of `index` is set, else `-1`. Indices `0 .. 2^(width·height)` list every state once.
pub fn state_from_index(index: usize, width: usize, height: usize) -> Spins {
    Image::from_fn(width, height, |x, y| if (index >> (y * width + x)) & 1 == 1 { 1 } else { -1 })
}

/// C1. The Ising energy `E = −J · Σ s_p s_q − h · Σ s_p`, the first sum over all pairs of
/// 4-neighbours, each pair once. With `periodic`, the right neighbour of the last column is
/// the first column (and likewise for rows), but only along sides of at least 3 pixels: on a
/// side of 2 the wrap-around pair would be the same pair again.
pub fn energy(s: &Spins, j: f32, h: f32, periodic: bool) -> f32 {
    let (w, ht) = (s.width(), s.height());
    let mut pairs = 0.0; // Σ s_p s_q
    let mut field = 0.0; // Σ s_p
    for (x, y) in s.coords() {
        let sp = s[(x, y)] as f32;
        field += sp;
        // right and down neighbour: every pair is counted once
        if x + 1 < w {
            pairs += sp * s[(x + 1, y)] as f32;
        } else if periodic && w >= 3 {
            pairs += sp * s[(0, y)] as f32;
        }
        if y + 1 < ht {
            pairs += sp * s[(x, y + 1)] as f32;
        } else if periodic && ht >= 3 {
            pairs += sp * s[(x, 0)] as f32;
        }
    }
    -j * pairs - h * field
}

/// C2. The energy levels of a `width × height` lattice: every different energy, with the
/// number of states that have it, sorted from the lowest energy up. Lists all `2^n` states.
/// Two energies closer than `1e-4` count as the same level.
pub fn energy_levels(width: usize, height: usize, j: f32, h: f32, periodic: bool) -> Vec<(f32, usize)> {
    let n = width * height;
    let mut levels: Vec<(f32, usize)> = Vec::new();
    for index in 0..(1usize << n) {
        let e = energy(&state_from_index(index, width, height), j, h, periodic);
        match levels.iter_mut().find(|(level, _)| (level - e).abs() < 1e-4) {
            Some(level) => level.1 += 1,
            None => levels.push((e, 1)),
        }
    }
    levels.sort_by(|a, b| a.0.total_cmp(&b.0));
    levels
}

/// C3. The partition function `Z = Σ (number of states) · e^(−E/T)` over the levels, and the
/// probability of each level, `count · e^(−E/T) / Z` (in the order of `levels`).
pub fn level_probabilities(levels: &[(f32, usize)], t: f64) -> (f64, Vec<f64>) {
    let weights: Vec<f64> = levels.iter().map(|&(e, count)| count as f64 * (-e as f64 / t).exp()).collect();
    let z: f64 = weights.iter().sum();
    (z, weights.iter().map(|w| w / z).collect())
}

/// C4. `Z` of a ring of `n` spins (each spin has a left and a right neighbour, the last one
/// next to the first) with the transfer matrix
/// `M = [[e^((J+h)/T), e^(−J/T)], [e^(−J/T), e^((J−h)/T)]]`: `Z = trace(M^n)`.
/// `n` small matrix products instead of `2^n` states.
pub fn ring_partition_function(n: usize, j: f64, h: f64, t: f64) -> f64 {
    // rows and columns: +1, −1 (slide "The math: the transfer matrix")
    let m = [[((j + h) / t).exp(), (-j / t).exp()], [(-j / t).exp(), ((j - h) / t).exp()]];
    let mut p = [[1.0, 0.0], [0.0, 1.0]]; // M^0
    for _ in 0..n {
        let mut next = [[0.0; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                next[r][c] = p[r][0] * m[0][c] + p[r][1] * m[1][c];
            }
        }
        p = next;
    }
    p[0][0] + p[1][1]
}
