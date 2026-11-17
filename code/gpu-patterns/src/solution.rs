// Model solution for the coding exercises of "Data-parallel patterns: map, stencil,
// reduction, scan and histogram". Every parallel step is written as a loop that reads only
// the previous row of values, so all its iterations could run at the same time.

/// C1. Tree reduction (sum): in every step, add neighbouring pairs of the previous row (an
/// odd one out at the end is carried over). Returns the rows after each step; the last row
/// holds the sum alone.
pub fn reduce_tree(values: &[f32]) -> Vec<Vec<f32>> {
    let mut rows = Vec::new();
    let mut row = values.to_vec();
    while row.len() > 1 {
        let mut next = Vec::with_capacity(row.len().div_ceil(2));
        for pair in row.chunks(2) {
            next.push(if pair.len() == 2 { pair[0] + pair[1] } else { pair[0] });
        }
        rows.push(next.clone());
        row = next;
    }
    rows
}

/// C2. The Hillis–Steele inclusive scan: in step `d` (offset `2^d` = 1, 2, 4, ...), every
/// element adds the element `2^d` places to its left, all at once from the previous row.
/// Returns the running totals and the number of additions.
pub fn scan_hillis_steele(x: &[f32]) -> (Vec<f32>, usize) {
    let mut row = x.to_vec();
    let mut additions = 0;
    let mut offset = 1;
    while offset < row.len() {
        let previous = row.clone(); // all lanes read the previous row
        for i in offset..row.len() {
            row[i] = previous[i] + previous[i - offset];
            additions += 1;
        }
        offset *= 2;
    }
    (row, additions)
}

/// C3. The Blelloch exclusive scan (`x.len()` a power of two): an up-sweep that builds a
/// reduction tree in place, the last element set to 0, and a down-sweep in which every node
/// passes its value to its left child and gives its right child the sum of the two.
/// Returns the exclusive scan and the number of additions.
pub fn scan_blelloch(x: &[f32]) -> (Vec<f32>, usize) {
    let n = x.len();
    assert!(n.is_power_of_two(), "the Blelloch scan here needs a power of two");
    let mut a = x.to_vec();
    let mut additions = 0;
    // Up-sweep: block sums of 2, 4, 8, ... stored at the right end of each block.
    let mut step = 1;
    while step < n {
        for i in (2 * step - 1..n).step_by(2 * step) {
            a[i] += a[i - step];
            additions += 1;
        }
        step *= 2;
    }
    a[n - 1] = 0.0;
    // Down-sweep: walk back down the tree.
    step = n / 2;
    while step >= 1 {
        for i in (2 * step - 1..n).step_by(2 * step) {
            let left = a[i - step];
            a[i - step] = a[i]; // the left child gets the node's value
            a[i] += left; // the right child gets the node's value + the left sum
            additions += 1;
        }
        step /= 2;
    }
    (a, additions)
}

/// C4. Stream compaction: keep the values whose `keep` flag is set, densely packed, in their
/// order. Every kept value's output position is the exclusive scan of the flags (a 0/1
/// list) at its index.
pub fn compact<T: Copy>(values: &[T], keep: &[bool]) -> Vec<T> {
    // The exclusive scan of the flags (here as a plain loop; on a GPU, C3).
    let mut position = vec![0; keep.len()];
    let mut count = 0;
    for i in 0..keep.len() {
        position[i] = count;
        count += keep[i] as usize;
    }
    // Scatter: every kept value writes itself to its position (no two collide).
    let mut out: Vec<Option<T>> = vec![None; count];
    for i in 0..values.len() {
        if keep[i] {
            out[position[i]] = Some(values[i]);
        }
    }
    out.into_iter().flatten().collect()
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
pub fn atomic_histogram(values: &[usize], bins: usize, block: usize) -> AtomicHistogram {
    let mut counts = vec![0; bins];
    for &v in values {
        counts[v] += 1;
    }
    let global_cycles = counts.iter().copied().max().unwrap_or(0);
    let mut busiest_private = 0;
    let blocks = values.chunks(block);
    let n_blocks = blocks.len();
    for chunk in blocks {
        let mut private = vec![0; bins];
        for &v in chunk {
            private[v] += 1;
        }
        busiest_private = busiest_private.max(private.into_iter().max().unwrap_or(0));
    }
    AtomicHistogram { counts, global_cycles, private_cycles: busiest_private + n_blocks }
}
