// Coding exercises for "Working with label images". Replace each `todo!()` with your code,
// then check it with `cargo test -p labelling label_measure::` and look at `cargo run -p labelling -- label-measure`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;
use std::collections::VecDeque;

/// Given (from the labelling unit, not part of this exercise): label the 8-connected
/// components of a mask 1, 2, 3, ... in raster order. Returns the labels and their number.
pub fn label_components(mask: &Image<bool>) -> (Image<u32>, u32) {
    let mut labels = Image::new(mask.width(), mask.height(), 0u32);
    let mut n = 0;
    for start in mask.coords() {
        if !mask[start] || labels[start] != 0 {
            continue;
        }
        n += 1;
        labels[start] = n;
        let mut queue = VecDeque::from([start]);
        while let Some((x, y)) = queue.pop_front() {
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if mask.contains(nx, ny) && mask[(nx as usize, ny as usize)] && labels[(nx as usize, ny as usize)] == 0 {
                        labels[(nx as usize, ny as usize)] = n;
                        queue.push_back((nx as usize, ny as usize));
                    }
                }
            }
        }
    }
    (labels, n)
}

/// One row of the measurement table.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Row {
    pub area: usize,
    /// bounding box, inclusive
    pub x_min: usize,
    pub x_max: usize,
    pub y_min: usize,
    pub y_max: usize,
}

impl Row {
    /// Does the object touch the edge of a `width × height` image?
    pub fn touches_border(&self, width: usize, height: usize) -> bool {
        self.x_min == 0 || self.y_min == 0 || self.x_max == width - 1 || self.y_max == height - 1
    }
}

/// C1. The measurement table: one row per label `1..=max` (element `k - 1` is label `k`) with
/// the area and the bounding box. Labels that do not occur get area 0.
///
/// Hint: start every row with `x_min: usize::MAX, y_min: usize::MAX` and the rest 0
/// (`..Default::default()`), then widen it at every pixel of its label.
pub fn table(labels: &Image<u32>) -> Vec<Row> {
    todo!("C1: table")
}

/// C2. Keep the labels `k` with `keep[k] == true` (`keep[0]` is ignored), set all others to 0,
/// and renumber the kept ones 1, 2, 3, ... in raster order of their first pixel. Returns the
/// new label image and the number of kept objects.
///
/// Hint: a vector `new_label` (one entry per old label, 0 = not seen yet) filled during one
/// raster scan gives the order of the first pixels.
pub fn keep_and_relabel(labels: &Image<u32>, keep: &[bool]) -> (Image<u32>, u32) {
    todo!("C2: keep_and_relabel")
}

/// C3. The area opening: keep the objects with at least `min_area` pixels, whole, and relabel.
///
/// Hint: `table`, then a `keep` vector, then `keep_and_relabel`.
pub fn area_opening(labels: &Image<u32>, min_area: usize) -> (Image<u32>, u32) {
    todo!("C3: area_opening")
}

/// C4. The unbiased counting frame with margin `m` on a `W × H` label image. The frame is
/// [m, W − m) × [m, H − m); the exclusion line is column W − m (rows 0 to H − m) and row H − m
/// (columns 0 to W − m). An object is counted if it has a pixel in the frame and none on the
/// exclusion line. Returns the counted labels, in increasing order.
///
/// Hint: two vectors of `bool` per label: "has a pixel in the frame" and "has a pixel on the
/// exclusion line". `(a..b).contains(&x)` tests a ≤ x < b.
pub fn counting_frame(labels: &Image<u32>, m: usize) -> Vec<u32> {
    todo!("C4: counting_frame")
}

/// C5. Hysteresis: label the mask `img > low`, keep the objects that contain at least one pixel
/// with `img > high`, and relabel. Returns the label image and the number of objects.
///
/// Hint: `label_components(&img.map(|v| v > low))` labels the low mask.
pub fn hysteresis(img: &Image<f32>, low: f32, high: f32) -> (Image<u32>, u32) {
    todo!("C5: hysteresis")
}
