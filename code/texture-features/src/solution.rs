// Model solution for the coding exercises of "Texture: co-occurrence matrices".

use imgcourse::Image;

/// The five Haralick features of the lecture (slide "Features: statistics of the matrix").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Haralick {
    pub contrast: f32,
    pub homogeneity: f32,
    pub asm: f32,
    /// In bits (log base 2).
    pub entropy: f32,
    pub correlation: f32,
}

/// C1. Quantise grey values to `levels` levels 0, 1, ..., levels − 1: the range `lo..hi`
/// is cut into `levels` equal parts; values below `lo` go to 0, values from `hi` up to the
/// last level.
pub fn quantise(img: &Image<f32>, levels: u32, lo: f32, hi: f32) -> Image<u32> {
    img.map(|v| {
        let t = (v - lo) / (hi - lo) * levels as f32;
        (t.max(0.0) as u32).min(levels - 1)
    })
}

/// C2. The symmetric co-occurrence counts `P` for the offset `(dx, dy)` (`y` down, so 90° is
/// `(0, −d)`): for every pixel whose partner at the offset is also inside the image and
/// inside `mask` (if given), add 1 to `P[i][j]` and 1 to `P[j][i]`, with `i` the level of the
/// pixel and `j` the level of its partner.
pub fn glcm(q: &Image<u32>, levels: u32, dx: isize, dy: isize, mask: Option<&Image<bool>>) -> Vec<Vec<f32>> {
    let l = levels as usize;
    let mut p = vec![vec![0.0; l]; l];
    let inside = |x: isize, y: isize| q.contains(x, y) && mask.is_none_or(|m| m[(x as usize, y as usize)]);
    for (x, y) in q.coords() {
        let (px, py) = (x as isize + dx, y as isize + dy);
        if inside(x as isize, y as isize) && inside(px, py) {
            let i = q[(x, y)] as usize;
            let j = q[(px as usize, py as usize)] as usize;
            p[i][j] += 1.0;
            p[j][i] += 1.0;
        }
    }
    p
}

/// C3. The Haralick features of a symmetric co-occurrence matrix (counts or probabilities):
/// first divide by the sum to get `p(i, j)`.
pub fn haralick(counts: &[Vec<f32>]) -> Haralick {
    let total: f32 = counts.iter().flatten().sum();
    let l = counts.len();
    let p = |i: usize, j: usize| counts[i][j] / total;
    let (mut contrast, mut homogeneity, mut asm, mut entropy, mut mu) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..l {
        for j in 0..l {
            let (pij, d) = (p(i, j), i as f32 - j as f32);
            contrast += d * d * pij;
            homogeneity += pij / (1.0 + d * d);
            asm += pij * pij;
            if pij > 0.0 {
                entropy -= pij * pij.log2();
            }
            mu += i as f32 * pij;
        }
    }
    let (mut var, mut cov) = (0.0, 0.0);
    for i in 0..l {
        for j in 0..l {
            var += (i as f32 - mu).powi(2) * p(i, j);
            cov += (i as f32 - mu) * (j as f32 - mu) * p(i, j);
        }
    }
    Haralick { contrast, homogeneity, asm, entropy, correlation: cov / var }
}

/// The 8 neighbours of a pixel as offsets `(dx, dy)`, clockwise from the top-left (as the LBP
/// of the lecture reads them).
pub const CLOCKWISE: [(isize, isize); 8] = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)];

/// C4a. The local binary pattern at `(x, y)`: the 8 neighbours clockwise from the top-left,
/// bit 1 if the neighbour is at least the centre, read as a binary number with the first
/// bit most significant. Pixels outside the image count as 0.
pub fn lbp(img: &Image<f32>, x: usize, y: usize) -> u8 {
    let c = img[(x, y)];
    let mut code = 0u8;
    for (dx, dy) in CLOCKWISE {
        let n = img.get_border(x as isize + dx, y as isize + dy, imgcourse::Border::Zero);
        code = (code << 1) | (n >= c) as u8;
    }
    code
}

/// C4b. The rotation-invariant code: the smallest of the 8 circular rotations of the bits.
pub fn rotation_invariant(code: u8) -> u8 {
    (0..8).map(|r| code.rotate_left(r)).min().unwrap()
}
