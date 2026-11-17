// Model solution for the coding exercises of "Information, entropy and entropy codes".

use imgcourse::Image;

/// C1. The histogram of an image with grey values in [0, 1], as 256 counts: the value `v`
/// counts as grey level `round(255 v)` (clipped to 0..=255).
pub fn histogram(img: &Image<f32>) -> Vec<usize> {
    let mut counts = vec![0; 256];
    for &v in img.pixels() {
        let level = (v * 255.0).round().clamp(0.0, 255.0) as usize;
        counts[level] += 1;
    }
    counts
}

/// C2. The entropy of a histogram in bits per symbol: H = −Σ p log2 p, with p = count / total.
/// Symbols with count 0 add nothing (0 log 0 = 0).
pub fn entropy(counts: &[usize]) -> f32 {
    let total: usize = counts.iter().sum();
    let mut h = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f32 / total as f32;
            h -= p * p.log2(); // the information −log2 p, weighted by p (slide "Entropy")
        }
    }
    h
}

/// C3. The code word lengths of a Huffman code for the given counts (slide "Huffman coding"):
/// repeatedly merge the two trees with the smallest counts; every merge makes the code words
/// of all symbols in both trees one bit longer. Among equal counts, take the tree that was
/// created first (leaves, in their order, before merged trees). Symbols with count 0 get
/// length 0; a single symbol gets length 1.
pub fn huffman_lengths(counts: &[usize]) -> Vec<usize> {
    let mut lengths = vec![0; counts.len()];
    // A tree: (count, creation number, the symbols in it).
    let mut trees: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    for (s, &c) in counts.iter().enumerate() {
        if c > 0 {
            trees.push((c, trees.len(), vec![s]));
        }
    }
    if trees.len() == 1 {
        lengths[trees[0].2[0]] = 1;
        return lengths;
    }
    let mut created = trees.len();
    while trees.len() > 1 {
        // the smallest tree (ties: created first), taken out; then the next smallest
        let take_smallest = |trees: &mut Vec<(usize, usize, Vec<usize>)>| {
            let mut best = 0;
            for i in 1..trees.len() {
                if (trees[i].0, trees[i].1) < (trees[best].0, trees[best].1) {
                    best = i;
                }
            }
            trees.remove(best)
        };
        let first = take_smallest(&mut trees);
        let second = take_smallest(&mut trees);
        let mut symbols = first.2;
        symbols.extend(second.2);
        for &s in &symbols {
            lengths[s] += 1;
        }
        trees.push((first.0 + second.0, created, symbols));
        created += 1;
    }
    lengths
}

/// C4. Run-length coding of a binary row (values 0 and 1): the lengths of the runs, starting
/// with a run of zeros (of length 0 if the row starts with a 1).
pub fn run_lengths(row: &[u8]) -> Vec<usize> {
    let mut runs = Vec::new();
    let mut current = 0u8; // the first run counts zeros
    let mut length = 0;
    for &v in row {
        if v == current {
            length += 1;
        } else {
            runs.push(length);
            current = v;
            length = 1;
        }
    }
    runs.push(length);
    runs
}

/// C5. Arithmetic coding: the interval [low, high) that a message narrows [0, 1) to. Symbol `s`
/// has probability `p[s]`; the symbols are laid out on [0, 1) in their order.
pub fn arithmetic_interval(p: &[f64], message: &[usize]) -> (f64, f64) {
    let (mut low, mut width) = (0.0, 1.0);
    for &s in message {
        let below: f64 = p[..s].iter().sum(); // the cumulative probability below symbol s
        low += width * below;
        width *= p[s];
    }
    (low, low + width)
}
