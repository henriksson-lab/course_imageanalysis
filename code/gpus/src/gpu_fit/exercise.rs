// Coding exercises for "What GPUs are good and bad at, and precision".
// Replace each `todo!()` with your code, then check it with `cargo test -p gpus gpu_fit::`
// and look at `cargo run -p gpus -- gpu-fit`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// C1a. One pass of label propagation, all at once: each foreground pixel takes the smallest
/// label among itself and its foreground 4-neighbours in `labels`. Returns the new labels and
/// whether anything changed.
///
/// Hint: start from a copy of `labels`; read neighbours only from `labels` (the previous pass),
/// write only to the copy. `mask.contains(x, y)` takes `isize` and tells whether a position is
/// inside the image.
pub fn propagate_once(mask: &Image<bool>, labels: &Image<u32>) -> (Image<u32>, bool) {
    todo!("C1a: propagate_once")
}

/// C1b. Label propagation: every foreground pixel starts with its raster number (row by row
/// from the top left, counted from 1) as its label. In every pass, all at once, each
/// foreground pixel takes the smallest label among itself and its foreground 4-neighbours
/// (from the previous pass; C1a). Stop when a pass changes nothing. Returns the labels (0 for
/// the background) and the number of passes that changed something.
///
/// Hint: pixel `(x, y)` has the raster number `y · width + x + 1`. Call C1a until it reports no
/// change.
pub fn label_propagation(mask: &Image<bool>) -> (Image<u32>, usize) {
    todo!("C1b: label_propagation")
}

/// C2. Label propagation with pointer jumping: after each propagation pass, every
/// foreground pixel replaces its label `L` by the label that pixel number `L` has now (its
/// label's label). Returns the labels and the number of passes that changed something.
///
/// Hint: after C1a, pixel number `L` sits at `((L − 1) % width, (L − 1) / width)`. Read the
/// jumped labels from the labels after propagation, not from the ones you are writing.
pub fn label_pointer_jumping(mask: &Image<bool>) -> (Image<u32>, usize) {
    todo!("C2: label_pointer_jumping")
}

/// C3. Dynamic programming by wavefronts. The table `D` has the size of `cost`;
/// `D(x, y) = cost(x, y) + min(D(x − 1, y), D(x, y − 1), D(x − 1, y − 1))` (neighbours outside
/// the table are left out; `D(0, 0) = cost(0, 0)`). The cells with the same `x + y` (an
/// anti-diagonal) depend only on earlier anti-diagonals: fill one anti-diagonal per parallel
/// step. Returns the table and the number of cells in each step.
///
/// Hint: for the anti-diagonal `s = x + y`, `y` runs from `max(0, s − (width − 1))` to `min(s, height − 1)`
/// (`s.saturating_sub(w - 1)` does the first), and `x = s − y`.
pub fn dp_wavefront(cost: &Image<f32>) -> (Image<f32>, Vec<usize>) {
    todo!("C3: dp_wavefront")
}

/// C4. Amdahl's law: the speed-up of the whole when a fraction `serial` of the time stays as
/// it is and the rest becomes `k` times faster.
///
/// Hint: one line.
pub fn amdahl(serial: f64, k: f64) -> f64 {
    todo!("C4: amdahl")
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
///
/// Hint: round with `to_f16` after every addition.
pub fn sum_f16(values: &[f32]) -> f32 {
    todo!("C5a: sum_f16")
}

/// C5b. Mixed precision: the values stored in float16, the sum kept in float32.
///
/// Hint: round only the values.
pub fn sum_mixed(values: &[f32]) -> f32 {
    todo!("C5b: sum_mixed")
}
