// Coding exercises for "Data-parallel patterns: map, stencil, reduction, scan and histogram".
// Replace each `todo!()` with your code, then check it with `cargo test -p gpus gpu_patterns::`
// and look at `cargo run -p gpus -- gpu-patterns`.
#![allow(unused_variables, unused_imports)]

/// C1. Tree reduction (sum): in every step, add neighbouring pairs of the previous row (an
/// odd one out at the end is carried over). Returns the rows after each step; the last row
/// holds the sum alone.
///
/// Hint: `row.chunks(2)` gives the pairs (the last chunk may have one element). Repeat while
/// the row is longer than 1.
pub fn reduce_tree(values: &[f32]) -> Vec<Vec<f32>> {
    todo!("C1: reduce_tree")
}

/// C2. The Hillis–Steele inclusive scan: in step `d` (offset `2^d` = 1, 2, 4, ...), every
/// element adds the element `2^d` places to its left, all at once from the previous row.
/// Returns the running totals and the number of additions.
///
/// Hint: copy the row before each step (`row.clone()`) and read only from the copy: that is
/// what "all at once" means.
pub fn scan_hillis_steele(x: &[f32]) -> (Vec<f32>, usize) {
    todo!("C2: scan_hillis_steele")
}

/// C3. The Blelloch exclusive scan (`x.len()` a power of two): an up-sweep that builds a
/// reduction tree in place, the last element set to 0, and a down-sweep in which every node
/// passes its value to its left child and gives its right child the sum of the two.
/// Returns the exclusive scan and the number of additions.
///
/// Hint: up-sweep: for `step = 1, 2, 4, ...` and `i = 2·step − 1, 4·step − 1, ...` (`(2 * step - 1..n).step_by(2 * step)`),
/// `a[i] += a[i − step]`. Then `a[n − 1] = 0`. Down-sweep: `step = n/2, ..., 1` over the same
/// `i`: remember `a[i − step]`, set it to `a[i]`, and add the remembered value to `a[i]`.
pub fn scan_blelloch(x: &[f32]) -> (Vec<f32>, usize) {
    todo!("C3: scan_blelloch")
}

/// C4. Stream compaction: keep the values whose `keep` flag is set, densely packed, in their
/// order. Every kept value's output position is the exclusive scan of the flags (a 0/1
/// list) at its index.
///
/// Hint: compute the exclusive scan of the flags (a running count) first; then write each kept
/// value to its position. A `Vec<Option<T>>` of the right length works:
/// `out.into_iter().flatten().collect()` drops the `Option`s at the end.
pub fn compact<T: Copy>(values: &[T], keep: &[bool]) -> Vec<T> {
    todo!("C4: compact")
}

/// The result of an atomic histogram.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomicHistogram {
    pub counts: Vec<usize>,
    /// Cycles with one global histogram: one atomic update per cycle per bin.
    pub global_cycles: usize,
    /// Cycles with a private histogram per block of `block` threads: the busiest private
    /// bin over all blocks (blocks run in parallel), plus one cycle per block to add the
    /// block histograms into the global one.
    pub private_cycles: usize,
}

/// C5. A histogram of `values` (each in `0..bins`, one thread per value) with atomic
/// additions, and its cost in cycles with and without private block histograms.
///
/// Hint: `values.chunks(block)` gives the blocks; count each block's own histogram and keep the
/// largest count seen.
pub fn atomic_histogram(values: &[usize], bins: usize, block: usize) -> AtomicHistogram {
    todo!("C5: atomic_histogram")
}
