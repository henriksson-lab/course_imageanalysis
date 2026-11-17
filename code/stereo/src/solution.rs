// Model solution for the coding exercises of "Stereo: from disparity to depth".
// A rectified pair: the left pixel (x, y) matches the right pixel (x − d, y), d ≥ 0 the
// disparity. A cost volume is one image of costs per disparity: `costs[d][(x, y)]`.

use imgcourse::{Border, Image};

/// C1. The SAD cost volume: for every disparity d in 0..=max_d and every left pixel (x, y),
/// the sum over the (2 radius + 1)² window of |L(x + i, y + j) − R(x + i − d, y + j)|. Pixels
/// outside the images are read as 0.
pub fn sad_cost_volume(left: &Image<f32>, right: &Image<f32>, max_d: usize, radius: usize) -> Vec<Image<f32>> {
    let r = radius as isize;
    let mut costs = Vec::new();
    for d in 0..=max_d as isize {
        let mut c = Image::new(left.width(), left.height(), 0.0f32);
        for (x, y) in left.coords() {
            let (x, y) = (x as isize, y as isize);
            let mut sum = 0.0;
            for j in -r..=r {
                for i in -r..=r {
                    let a = left.get_border(x + i, y + j, Border::Zero);
                    let b = right.get_border(x + i - d, y + j, Border::Zero);
                    sum += (a - b).abs();
                }
            }
            c[(x as usize, y as usize)] = sum;
        }
        costs.push(c);
    }
    costs
}

/// C2a. The census transform with a 3 × 3 window: for every pixel an 8-bit code, one bit per
/// neighbour, row by row and skipping the centre, the first neighbour in the highest bit.
/// A bit is 1 if the neighbour is darker than the centre. Outside the image, the edge is copied.
pub fn census(img: &Image<f32>) -> Image<u32> {
    img.map_xy(|x, y, centre| {
        let mut code = 0u32;
        for j in -1..=1isize {
            for i in -1..=1isize {
                if i == 0 && j == 0 {
                    continue;
                }
                let darker = img.get_border(x as isize + i, y as isize + j, Border::Clamp) < centre;
                code = (code << 1) | darker as u32;
            }
        }
        code
    })
}

/// C2b. The Hamming distance of two codes: the number of bits in which they differ.
pub fn hamming(a: u32, b: u32) -> u32 {
    let mut diff = a ^ b;
    let mut count = 0;
    while diff != 0 {
        count += diff & 1;
        diff >>= 1;
    }
    count
}

/// C3. Winner-take-all: at every pixel, the disparity with the lowest cost (the smallest d on
/// a tie).
pub fn winner_take_all(costs: &[Image<f32>]) -> Image<f32> {
    costs[0].map_xy(|x, y, _| {
        let mut best = 0;
        for d in 1..costs.len() {
            if costs[d][(x, y)] < costs[best][(x, y)] {
                best = d;
            }
        }
        best as f32
    })
}

/// C4. Scanline dynamic programming on one row: `costs[x][d]` for the pixels x of the row.
/// Find the disparity profile with the smallest energy: the sum of its costs, plus `p1` for
/// every step of 1 between neighbours and `p2` for every larger jump (slide "Scanline dynamic
/// programming"). Fill a table of cumulative costs column by column with a pointer to the best
/// predecessor, then backtrack from the best end.
pub fn scanline_dp(costs: &[Vec<f32>], p1: f32, p2: f32) -> Vec<usize> {
    let (n, nd) = (costs.len(), costs[0].len());
    let mut total = vec![vec![0.0f32; nd]; n];
    let mut from = vec![vec![0usize; nd]; n];
    total[0] = costs[0].clone();
    for x in 1..n {
        for d in 0..nd {
            let mut best = f32::INFINITY;
            for e in 0..nd {
                let jump = if e == d { 0.0 } else if e.abs_diff(d) == 1 { p1 } else { p2 };
                if total[x - 1][e] + jump < best {
                    best = total[x - 1][e] + jump;
                    from[x][d] = e;
                }
            }
            total[x][d] = costs[x][d] + best;
        }
    }
    // backtrack from the cheapest end
    let mut path = vec![0; n];
    path[n - 1] = (0..nd).min_by(|&a, &b| total[n - 1][a].total_cmp(&total[n - 1][b])).unwrap();
    for x in (1..n).rev() {
        path[x - 1] = from[x][path[x]];
    }
    path
}

/// C5. Sub-pixel disparity: the offset of the minimum of the parabola through the costs at
/// d − 1, d and d + 1 (slide "The math: the sub-pixel parabola"), between −½ and ½.
pub fn subpixel_offset(c_minus: f32, c0: f32, c_plus: f32) -> f32 {
    let curvature = c_minus - 2.0 * c0 + c_plus;
    if curvature <= 0.0 {
        return 0.0;
    }
    (c_minus - c_plus) / (2.0 * curvature)
}
