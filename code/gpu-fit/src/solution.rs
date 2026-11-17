// Model solution for the coding exercises of "What GPUs are good and bad at, and precision".
// The parallel algorithms are written as passes: every pass reads only the result of the
// previous pass, so all pixels of a pass could be done at the same time.

use imgcourse::Image;

/// C1a. One pass of label propagation, all at once: each foreground pixel takes the smallest
/// label among itself and its foreground 4-neighbours in `labels`. Returns the new labels and
/// whether anything changed.
pub fn propagate_once(mask: &Image<bool>, labels: &Image<u32>) -> (Image<u32>, bool) {
    let mut next = labels.clone();
    let mut changed = false;
    for (x, y) in mask.coords() {
        if !mask[(x, y)] {
            continue;
        }
        let mut m = labels[(x, y)];
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if mask.contains(nx, ny) && mask[(nx as usize, ny as usize)] {
                m = m.min(labels[(nx as usize, ny as usize)]);
            }
        }
        if m != labels[(x, y)] {
            next[(x, y)] = m;
            changed = true;
        }
    }
    (next, changed)
}

/// C1b. Label propagation: every foreground pixel starts with its raster number (row by row
/// from the top left, counted from 1) as its label. In every pass, all at once, each
/// foreground pixel takes the smallest label among itself and its foreground 4-neighbours
/// (from the previous pass; C1a). Stop when a pass changes nothing. Returns the labels (0 for
/// the background) and the number of passes that changed something.
pub fn label_propagation(mask: &Image<bool>) -> (Image<u32>, usize) {
    let (w, h) = (mask.width(), mask.height());
    let mut labels = Image::from_fn(w, h, |x, y| if mask[(x, y)] { (y * w + x + 1) as u32 } else { 0 });
    let mut passes = 0;
    loop {
        let (next, changed) = propagate_once(mask, &labels);
        if !changed {
            return (labels, passes);
        }
        labels = next;
        passes += 1;
    }
}

/// C2. Label propagation with pointer jumping: after each propagation pass, every
/// foreground pixel replaces its label `L` by the label that pixel number `L` has now (its
/// label's label). Returns the labels and the number of passes that changed something.
pub fn label_pointer_jumping(mask: &Image<bool>) -> (Image<u32>, usize) {
    let (w, h) = (mask.width(), mask.height());
    let mut labels = Image::from_fn(w, h, |x, y| if mask[(x, y)] { (y * w + x + 1) as u32 } else { 0 });
    let mut passes = 0;
    loop {
        let (mut next, mut changed) = propagate_once(mask, &labels);
        // The jump, all at once: read from the labels after propagation.
        let after = next.clone();
        for (x, y) in mask.coords() {
            let l = after[(x, y)];
            if l > 0 {
                let i = (l - 1) as usize; // pixel number l is at (i % w, i / w)
                let jumped = after[(i % w, i / w)];
                if jumped < l {
                    next[(x, y)] = jumped;
                    changed = true;
                }
            }
        }
        if !changed {
            return (labels, passes);
        }
        labels = next;
        passes += 1;
    }
}

/// C3. Dynamic programming by wavefronts. The table `D` has the size of `cost`;
/// `D(x, y) = cost(x, y) + min(D(x − 1, y), D(x, y − 1), D(x − 1, y − 1))` (neighbours outside
/// the table are left out; `D(0, 0) = cost(0, 0)`). The cells with the same `x + y` (an
/// anti-diagonal) depend only on earlier anti-diagonals: fill one anti-diagonal per parallel
/// step. Returns the table and the number of cells in each step.
pub fn dp_wavefront(cost: &Image<f32>) -> (Image<f32>, Vec<usize>) {
    let (w, h) = (cost.width(), cost.height());
    let mut d = Image::new(w, h, 0.0f32);
    let mut cells = Vec::new();
    for s in 0..w + h - 1 {
        // All cells with x + y = s; they could all be computed at the same time.
        let mut count = 0;
        for y in s.saturating_sub(w - 1)..=s.min(h - 1) {
            let x = s - y;
            let mut best = f32::INFINITY;
            if x > 0 {
                best = best.min(d[(x - 1, y)]);
            }
            if y > 0 {
                best = best.min(d[(x, y - 1)]);
            }
            if x > 0 && y > 0 {
                best = best.min(d[(x - 1, y - 1)]);
            }
            d[(x, y)] = cost[(x, y)] + if best.is_finite() { best } else { 0.0 };
            count += 1;
        }
        cells.push(count);
    }
    (d, cells)
}

/// C4. Amdahl's law: the speed-up of the whole when a fraction `serial` of the time stays as
/// it is and the rest becomes `k` times faster.
pub fn amdahl(serial: f64, k: f64) -> f64 {
    1.0 / (serial + (1.0 - serial) / k)
}

/// The nearest float16 value of `v` (round to nearest, ties to even): 11 significant bits,
/// largest finite value 65 504, larger magnitudes become infinite. The result is stored in an
/// `f32`, which holds every float16 value exactly.
pub fn to_f16(v: f32) -> f32 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    // The power of two of v (from its float32 bits), at least −14 (float16 subnormals).
    let exponent = (((v.to_bits() >> 23) & 0xff) as i32 - 127).max(-14);
    let spacing = 2f32.powi(exponent - 10);
    let r = (v / spacing).round_ties_even() * spacing;
    if r.abs() > 65504.0 {
        f32::INFINITY.copysign(v)
    } else {
        r
    }
}

/// C5a. Sum in float16: every value is stored in float16 (`to_f16`), and every partial sum
/// is rounded to float16 too.
pub fn sum_f16(values: &[f32]) -> f32 {
    let mut s = 0.0;
    for &v in values {
        s = to_f16(s + to_f16(v));
    }
    s
}

/// C5b. Mixed precision: the values stored in float16, the sum kept in float32.
pub fn sum_mixed(values: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &v in values {
        s += to_f16(v);
    }
    s
}
