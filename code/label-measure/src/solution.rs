// Model solution for the coding exercises of "Working with label images".

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
pub fn table(labels: &Image<u32>) -> Vec<Row> {
    let n = labels.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut rows = vec![Row { x_min: usize::MAX, y_min: usize::MAX, ..Default::default() }; n];
    for (x, y) in labels.coords() {
        let l = labels[(x, y)] as usize;
        if l > 0 {
            let r = &mut rows[l - 1];
            r.area += 1;
            r.x_min = r.x_min.min(x);
            r.x_max = r.x_max.max(x);
            r.y_min = r.y_min.min(y);
            r.y_max = r.y_max.max(y);
        }
    }
    rows
}

/// C2. Keep the labels `k` with `keep[k] == true` (`keep[0]` is ignored), set all others to 0,
/// and renumber the kept ones 1, 2, 3, ... in raster order of their first pixel. Returns the
/// new label image and the number of kept objects.
pub fn keep_and_relabel(labels: &Image<u32>, keep: &[bool]) -> (Image<u32>, u32) {
    let mut new_label = vec![0u32; keep.len()];
    let mut n = 0;
    let mut out = Image::new(labels.width(), labels.height(), 0u32);
    for (x, y) in labels.coords() {
        let l = labels[(x, y)] as usize;
        if l == 0 || !keep[l] {
            continue;
        }
        if new_label[l] == 0 {
            // the first pixel of this label in raster order
            n += 1;
            new_label[l] = n;
        }
        out[(x, y)] = new_label[l];
    }
    (out, n)
}

/// C3. The area opening: keep the objects with at least `min_area` pixels, whole, and relabel.
pub fn area_opening(labels: &Image<u32>, min_area: usize) -> (Image<u32>, u32) {
    let rows = table(labels);
    let mut keep = vec![false; rows.len() + 1];
    for (k, r) in rows.iter().enumerate() {
        keep[k + 1] = r.area >= min_area;
    }
    keep_and_relabel(labels, &keep)
}

/// C4. The unbiased counting frame with margin `m` on a `W × H` label image. The frame is
/// [m, W − m) × [m, H − m); the exclusion line is column W − m (rows 0 to H − m) and row H − m
/// (columns 0 to W − m). An object is counted if it has a pixel in the frame and none on the
/// exclusion line. Returns the counted labels, in increasing order.
pub fn counting_frame(labels: &Image<u32>, m: usize) -> Vec<u32> {
    let (w, h) = (labels.width(), labels.height());
    let n = labels.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut hits = vec![false; n + 1];
    let mut excluded = vec![false; n + 1];
    for (x, y) in labels.coords() {
        let l = labels[(x, y)] as usize;
        if l == 0 {
            continue;
        }
        if (m..w - m).contains(&x) && (m..h - m).contains(&y) {
            hits[l] = true;
        }
        if (x == w - m && y <= h - m) || (y == h - m && x <= w - m) {
            excluded[l] = true;
        }
    }
    (1..=n).filter(|&l| hits[l] && !excluded[l]).map(|l| l as u32).collect()
}

/// C5. Hysteresis: label the mask `img > low`, keep the objects that contain at least one pixel
/// with `img > high`, and relabel. Returns the label image and the number of objects.
pub fn hysteresis(img: &Image<f32>, low: f32, high: f32) -> (Image<u32>, u32) {
    let (labels, n) = label_components(&img.map(|v| v > low));
    let mut keep = vec![false; n as usize + 1];
    for (x, y) in img.coords() {
        if img[(x, y)] > high {
            keep[labels[(x, y)] as usize] = true;
        }
    }
    keep[0] = false;
    keep_and_relabel(&labels, &keep)
}
