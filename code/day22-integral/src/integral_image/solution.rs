// Model solution for the coding exercises of "Integral images".

use imgcourse::Image;

/// C1. The running total of a row: `s[0] = 0` and `s[i] = f[0] + ... + f[i-1]`, so `s` is one
/// longer than `f` (slide "The running total").
pub fn running_total(f: &[f64]) -> Vec<f64> {
    let mut s = vec![0.0; f.len() + 1];
    for i in 0..f.len() {
        s[i + 1] = s[i] + f[i];
    }
    s
}

/// C1. The sum of the pixels `a..=b` of the row (counted from 0), from its running total `s`:
/// one subtraction, whatever the length of the range.
pub fn range_sum(s: &[f64], a: usize, b: usize) -> f64 {
    s[b + 1] - s[a]
}

/// C2. The integral image of `img`, one larger than the image in each direction:
/// `table[(x, y)]` is the sum of the pixels in columns `0..x` and rows `0..y` (not including
/// `x` and `y`), so row 0 and column 0 are zero. Built in one pass with the recursion of the
/// slide "Building the table in one pass".
pub fn integral_image(img: &Image<f32>) -> Image<f64> {
    let mut s = Image::new(img.width() + 1, img.height() + 1, 0.0);
    for (x, y) in img.coords() {
        // pixel (x, y) is entry (x + 1, y + 1) of the table
        s[(x + 1, y + 1)] = img[(x, y)] as f64 + s[(x, y + 1)] + s[(x + 1, y)] - s[(x, y)];
    }
    s
}

/// C3. The sum of the pixels in columns `x0..=x1` and rows `y0..=y1` (counted from 0), from the
/// table: four lookups (slide "Any rectangle from four corners").
pub fn rect_sum(s: &Image<f64>, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    s[(x1 + 1, y1 + 1)] - s[(x0, y1 + 1)] - s[(x1 + 1, y0)] + s[(x0, y0)]
}

/// C4. The same table in two passes: running totals along every row, then running totals of
/// the result down every column (exercise B8).
pub fn integral_two_pass(img: &Image<f32>) -> Image<f64> {
    let (w, h) = (img.width(), img.height());
    let mut s = Image::new(w + 1, h + 1, 0.0);
    for y in 0..h {
        for x in 0..w {
            s[(x + 1, y + 1)] = s[(x, y + 1)] + img[(x, y)] as f64;
        }
    }
    for x in 1..=w {
        for y in 2..=h {
            let above = s[(x, y - 1)];
            s[(x, y)] += above;
        }
    }
    s
}

/// C5. The sum of every `k × k` window that fits inside the image, added up directly
/// (`k * k` pixels per window). The window with top-left corner `(x, y)` goes to `out[(x, y)]`.
pub fn window_sums_direct(img: &Image<f32>, k: usize) -> Image<f64> {
    let (w, h) = (img.width() + 1 - k, img.height() + 1 - k);
    let mut out = Image::new(w, h, 0.0);
    for (x, y) in out.coords() {
        let mut sum = 0.0;
        for dy in 0..k {
            for dx in 0..k {
                sum += img[(x + dx, y + dy)] as f64;
            }
        }
        out[(x, y)] = sum;
    }
    out
}

/// C5. The same window sums from the table: four lookups per window.
pub fn window_sums_table(img: &Image<f32>, k: usize) -> Image<f64> {
    let s = integral_image(img);
    let (w, h) = (img.width() + 1 - k, img.height() + 1 - k);
    let mut out = Image::new(w, h, 0.0);
    for (x, y) in out.coords() {
        out[(x, y)] = rect_sum(&s, x, y, x + k - 1, y + k - 1);
    }
    out
}
