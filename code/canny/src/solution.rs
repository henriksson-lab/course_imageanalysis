// Model solution for the coding exercises of "Edges: Gaussian derivatives and Canny".

use imgcourse::{Border, Image};

/// C1a. A 1D Gaussian kernel of width `sigma`: the weights e^(−x²/2σ²) for the offsets
/// x = −r … r with r = ⌈4σ⌉ (cut at 4σ, as in the demos), scaled to sum 1.
/// Element `i` is the weight for the offset `i − r`.
pub fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let r = (4.0 * sigma).ceil() as isize;
    let mut w: Vec<f32> = (-r..=r).map(|x| (-((x * x) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f32 = w.iter().sum();
    for v in &mut w {
        *v /= sum;
    }
    w
}

/// C1b. The derivative of a Gaussian as a 1D kernel ("right side minus left side, with
/// Gaussian weights"): weights x·e^(−x²/2σ²) for x = −r … r, scaled so that a ramp of
/// slope 1 gives exactly 1, i.e. so that the sum of x·w(x) is 1 (slide "Derivative of a
/// Gaussian"). Used without flipping, so the weight for x > 0 is positive.
pub fn gaussian_derivative_kernel(sigma: f32) -> Vec<f32> {
    let r = (4.0 * sigma).ceil() as isize;
    let mut w: Vec<f32> = (-r..=r).map(|x| x as f32 * (-((x * x) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let slope: f32 = (-r..=r).zip(&w).map(|(x, v)| x as f32 * v).sum();
    for v in &mut w {
        *v /= slope;
    }
    w
}

/// C2a. Filter every row with the 1D kernel `w` (odd length, centre in the middle, not
/// flipped): out(x, y) = sum over d of w[d + r] · I(x + d, y). Outside the image: copy the edge.
pub fn filter_rows(img: &Image<f32>, w: &[f32]) -> Image<f32> {
    let r = (w.len() / 2) as isize;
    img.map_xy(|x, y, _| {
        let mut sum = 0.0;
        for d in -r..=r {
            sum += w[(d + r) as usize] * img.get_border(x as isize + d, y as isize, Border::Clamp);
        }
        sum
    })
}

/// C2b. The same along every column: out(x, y) = sum over d of w[d + r] · I(x, y + d).
pub fn filter_columns(img: &Image<f32>, w: &[f32]) -> Image<f32> {
    let r = (w.len() / 2) as isize;
    img.map_xy(|x, y, _| {
        let mut sum = 0.0;
        for d in -r..=r {
            sum += w[(d + r) as usize] * img.get_border(x as isize, y as isize + d, Border::Clamp);
        }
        sum
    })
}

/// C2c. The gradient (Lx, Ly) of the image smoothed at scale `sigma`, in two separable
/// passes each: Lx = derivative along the rows, Gaussian along the columns; Ly = Gaussian
/// along the rows, derivative along the columns (y down, so Ly is the change going down).
pub fn gradient(img: &Image<f32>, sigma: f32) -> (Image<f32>, Image<f32>) {
    let g = gaussian_kernel(sigma);
    let dg = gaussian_derivative_kernel(sigma);
    let lx = filter_columns(&filter_rows(img, &dg), &g);
    let ly = filter_columns(&filter_rows(img, &g), &dg);
    (lx, ly)
}

/// C3a. The gradient direction rounded for non-maximum suppression: 0, 1, 2 or 3 for
/// 0°, 45°, 90° or 135°. The angle is counted counter-clockwise on the screen,
/// θ = atan2(−Ly, Lx), and θ and θ + 180° give the same axis.
pub fn direction_sector(lx: f32, ly: f32) -> usize {
    let theta = (-ly).atan2(lx).to_degrees(); // −180 … 180
    let axis = theta.rem_euclid(180.0); // 0 … 180
    ((axis / 45.0).round() as usize) % 4
}

/// C3b. Non-maximum suppression: keep the magnitude of a pixel if it is at least as large
/// as both neighbours along the rounded gradient direction, else set it to 0.
/// Neighbours outside the image count as 0.
pub fn non_max_suppression(mag: &Image<f32>, lx: &Image<f32>, ly: &Image<f32>) -> Image<f32> {
    // One step "forwards" along each rounded direction, in image coordinates (y down):
    // 0°: right, 45°: up-right, 90°: up, 135°: up-left. The other neighbour is the opposite step.
    let steps = [(1, 0), (1, -1), (0, -1), (-1, -1)];
    mag.map_xy(|x, y, m| {
        let (dx, dy) = steps[direction_sector(lx[(x, y)], ly[(x, y)])];
        let (x, y) = (x as isize, y as isize);
        let ahead = mag.get_border(x + dx, y + dy, Border::Zero);
        let behind = mag.get_border(x - dx, y - dy, Border::Zero);
        if m >= ahead && m >= behind {
            m
        } else {
            0.0
        }
    })
}

/// C4. Hysteresis: pixels ≥ `high` are seeds; a pixel ≥ `low` is kept if it is connected to
/// a seed through pixels ≥ `low` (8 neighbours). Returns true for the kept (edge) pixels.
pub fn hysteresis(strength: &Image<f32>, low: f32, high: f32) -> Image<bool> {
    let mut edge = Image::new(strength.width(), strength.height(), false);
    // The flood: start with all seeds, and keep a list of pixels whose neighbours are still to visit.
    let mut todo = Vec::new();
    for (x, y) in strength.coords() {
        if strength[(x, y)] >= high {
            edge[(x, y)] = true;
            todo.push((x, y));
        }
    }
    while let Some((x, y)) = todo.pop() {
        for dy in -1..=1isize {
            for dx in -1..=1isize {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if !strength.contains(nx, ny) {
                    continue;
                }
                let n = (nx as usize, ny as usize);
                if !edge[n] && strength[n] >= low {
                    edge[n] = true;
                    todo.push(n);
                }
            }
        }
    }
    edge
}
