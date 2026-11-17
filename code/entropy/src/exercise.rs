// Coding exercises for "Information, entropy and entropy codes". Replace each `todo!()` with
// your code, then check it with `cargo test -p entropy` and look at `cargo run -p entropy`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// C1. The histogram of an image with grey values in [0, 1], as 256 counts: the value `v`
/// counts as grey level `round(255 v)` (clipped to 0..=255).
///
/// Hint: `let mut counts = vec![0; 256];`, then loop over `img.pixels()`.
pub fn histogram(img: &Image<f32>) -> Vec<usize> {
    todo!("C1: histogram")
}

/// C2. The entropy of a histogram in bits per symbol: H = −Σ p log2 p, with p = count / total.
/// Symbols with count 0 add nothing (0 log 0 = 0).
///
/// Hint: `p.log2()` is the logarithm to base 2.
pub fn entropy(counts: &[usize]) -> f32 {
    todo!("C2: entropy")
}

/// C3. The code word lengths of a Huffman code for the given counts (slide "Huffman coding"):
/// repeatedly merge the two trees with the smallest counts; every merge makes the code words
/// of all symbols in both trees one bit longer. Among equal counts, take the tree that was
/// created first (leaves, in their order, before merged trees). Symbols with count 0 get
/// length 0; a single symbol gets length 1.
///
/// Hint: keep a list of trees `(count, creation number, symbols in the tree)`. You never need
/// the tree's shape, only which symbols are in it: each merge adds 1 to their lengths.
pub fn huffman_lengths(counts: &[usize]) -> Vec<usize> {
    todo!("C3: huffman_lengths")
}

/// C4. Run-length coding of a binary row (values 0 and 1): the lengths of the runs, starting
/// with a run of zeros (of length 0 if the row starts with a 1).
pub fn run_lengths(row: &[u8]) -> Vec<usize> {
    todo!("C4: run_lengths")
}

/// C5. Arithmetic coding: the interval [low, high) that a message narrows [0, 1) to. Symbol `s`
/// has probability `p[s]`; the symbols are laid out on [0, 1) in their order.
///
/// Hint: keep `low` and `width`. Symbol `s` starts at `p[0] + ... + p[s-1]` within the interval.
pub fn arithmetic_interval(p: &[f64], message: &[usize]) -> (f64, f64) {
    todo!("C5: arithmetic_interval")
}
