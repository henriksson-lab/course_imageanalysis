// Model solution for the coding exercises of "Compression and measurements, JPEG 2000, formats".

use imgcourse::Image;

/// Floor of a / b for whole numbers (b > 0), also for negative a: `floor_div(-3, 2) = -2`.
pub fn floor_div(a: i32, b: i32) -> i32 {
    a.div_euclid(b)
}

/// C1. One level of the reversible 5/3 wavelet by lifting (slide "The math: an exactly
/// reversible wavelet"), for a signal of even length. Returns (s, d): the smooth (low-pass)
/// and detail (high-pass) halves.
///
///   d[n] = x[2n+1] − floor((x[2n] + x[2n+2]) / 2)
///   s[n] = x[2n]   + floor((d[n−1] + d[n] + 2) / 4)
///
/// At the ends the signal is mirrored: x[len] means x[len − 2], and d[−1] means d[0].
pub fn lift53(x: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let half = x.len() / 2;
    let mut d = vec![0; half];
    for n in 0..half {
        let right = if 2 * n + 2 < x.len() { x[2 * n + 2] } else { x[2 * n] }; // mirror at the end
        d[n] = x[2 * n + 1] - floor_div(x[2 * n] + right, 2); // predict: odd minus the mean of its even neighbours
    }
    let mut s = vec![0; half];
    for n in 0..half {
        let left = if n > 0 { d[n - 1] } else { d[0] }; // mirror at the start
        s[n] = x[2 * n] + floor_div(left + d[n] + 2, 4); // update
    }
    (s, d)
}

/// C2. The inverse of `lift53`: undo the update, then undo the prediction, with exactly the
/// same rounded terms. Integers in, the same integers out.
pub fn unlift53(s: &[i32], d: &[i32]) -> Vec<i32> {
    let half = s.len();
    let mut x = vec![0; 2 * half];
    for n in 0..half {
        let left = if n > 0 { d[n - 1] } else { d[0] };
        x[2 * n] = s[n] - floor_div(left + d[n] + 2, 4);
    }
    for n in 0..half {
        let right = if n + 1 < half { x[2 * n + 2] } else { x[2 * n] };
        x[2 * n + 1] = d[n] + floor_div(x[2 * n] + right, 2);
    }
    x
}

/// C3. One level of the 2D wavelet: `lift53` on every row, then on every column. The result has
/// the same size, with the four subbands in the four quarters: LL (smooth in both directions)
/// top left, then the detail bands. Width and height must be even.
pub fn wavelet2d(img: &Image<i32>) -> Image<i32> {
    let (w, h) = (img.width(), img.height());
    let mut rows = Image::new(w, h, 0);
    for y in 0..h {
        let (s, d) = lift53(img.row(y));
        for x in 0..w / 2 {
            rows[(x, y)] = s[x];
            rows[(w / 2 + x, y)] = d[x];
        }
    }
    let mut out = Image::new(w, h, 0);
    for x in 0..w {
        let column: Vec<i32> = (0..h).map(|y| rows[(x, y)]).collect();
        let (s, d) = lift53(&column);
        for y in 0..h / 2 {
            out[(x, y)] = s[y];
            out[(x, h / 2 + y)] = d[y];
        }
    }
    out
}

/// C4. Bit planes: a decoder has received the top `planes` bits of a `bits`-bit value. Which
/// value does it guess? The middle of all values that start with those bits.
pub fn bitplane_guess(value: u32, bits: u32, planes: u32) -> f32 {
    let dropped = bits - planes; // the bits not received yet
    let low = (value >> dropped) << dropped; // the received bits, the rest 0
    let high = low + (1 << dropped) - 1; // the received bits, the rest 1
    (low + high) as f32 / 2.0
}

/// C5. How many chunks a region read touches (slide "The math: how many chunks?"): the region
/// is `[a_d, b_d)` along each dimension d, the chunks have size `c_d` and start at 0.
pub fn chunks_read(region: &[(usize, usize)], chunk: &[usize]) -> usize {
    let mut count = 1;
    for (&(a, b), &c) in region.iter().zip(chunk) {
        count *= (b - 1) / c - a / c + 1;
    }
    count
}
