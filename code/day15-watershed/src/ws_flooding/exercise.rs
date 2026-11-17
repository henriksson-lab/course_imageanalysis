// Coding exercises for "The flooding algorithm on a grid". Replace each `todo!()` with your code,
// then check it with `cargo test -p day15 ws_flooding::` and look at `cargo run -p day15 -- ws-flooding`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;
use crate::common::{Entry, N8};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

/// The 4 neighbours `(dx, dy)` in reading order: up, left, right, down.
pub const N4: [(isize, isize); 4] = [(0, -1), (-1, 0), (1, 0), (0, 1)];

/// The neighbour offsets for 8 neighbours (`eight = true`) or 4.
pub fn neighbours(eight: bool) -> &'static [(isize, isize)] {
    if eight {
        &N8
    } else {
        &N4
    }
}

/// A queue for flooding: `pop` returns the queued pixel of smallest height; among equal
/// heights, the one that arrived first.
pub trait FloodQueue {
    fn push(&mut self, height: f32, pixel: (usize, usize));
    fn pop(&mut self) -> Option<(usize, usize)>;
}

/// Given: a priority queue on a binary heap, with an arrival number to break ties
/// (first in, first out; or last in, first out with `HeapQueue::lifo()`).
pub struct HeapQueue {
    heap: BinaryHeap<Reverse<Entry>>,
    arrivals: i64,
    lifo: bool,
}

impl HeapQueue {
    /// Ties first in, first out.
    pub fn new() -> Self {
        HeapQueue { heap: BinaryHeap::new(), arrivals: 0, lifo: false }
    }
    /// Ties last in, first out (exercise B3 (b)).
    pub fn lifo() -> Self {
        HeapQueue { heap: BinaryHeap::new(), arrivals: 0, lifo: true }
    }
}

impl Default for HeapQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl FloodQueue for HeapQueue {
    fn push(&mut self, height: f32, pixel: (usize, usize)) {
        let k = if self.lifo { -self.arrivals } else { self.arrivals };
        self.arrivals += 1;
        self.heap.push(Reverse(Entry(height, k, pixel)));
    }
    fn pop(&mut self) -> Option<(usize, usize)> {
        self.heap.pop().map(|Reverse(e)| e.2)
    }
}

/// The label of a watershed line pixel while flooding (it is 0 in the result).
pub const LINE: u32 = u32::MAX;

/// C1. The regional minima: plateaus (connected sets of equal height, with 4 or 8 neighbours)
/// that have no lower neighbour. Returns the minima labelled 1, 2, 3, ... in raster order of
/// their first pixel, and their number.
///
/// Hint: go through the pixels; at every pixel not yet seen, collect its plateau with a flood
/// fill over neighbours of the same height (a `VecDeque` as the queue), and note on the way
/// whether any neighbour is lower.
pub fn regional_minima(f: &Image<f32>, eight: bool) -> (Image<u32>, u32) {
    todo!("C1: regional_minima")
}

/// C2. Meyer's flooding from the `markers` (0 = no marker). Every unlabelled neighbour of a
/// marker enters the queue (markers in raster order, neighbours in reading order). Then pop
/// the lowest pixel: if `line` is true and its labelled neighbours carry different labels, it
/// becomes a line pixel (and pushes nothing); otherwise it takes the label of the pixel that
/// queued it. Then it pushes its unlabelled neighbours that are not yet queued.
/// Returns the labels (0 = watershed line) and the pixels in the order they were popped.
///
/// Hint: keep `labels` (start: a copy of the markers; use `LINE` for line pixels while
/// flooding), `queued` (an `Image<bool>`) and `by` (the label of the pixel that queued each
/// pixel). The queue is used as `queue.push(f[q], q)` and `while let Some(p) = queue.pop() { ... }`.
/// At the end, `labels.map(|l| if l == LINE { 0 } else { l })`.
pub fn watershed(f: &Image<f32>, markers: &Image<u32>, eight: bool, line: bool, queue: &mut impl FloodQueue) -> (Image<u32>, Vec<(usize, usize)>) {
    todo!("C2: watershed")
}

/// C3. A bucket queue for heights that are whole numbers 0, 1, ..., 255: one first-in-first-out
/// list per height and a pointer `current` to the lowest list that may be non-empty.
pub struct BucketQueue {
    pub buckets: Vec<VecDeque<(usize, usize)>>,
    pub current: usize,
}

impl BucketQueue {
    pub fn new() -> Self {
        BucketQueue { buckets: vec![VecDeque::new(); 256], current: 0 }
    }
}

impl Default for BucketQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl FloodQueue for BucketQueue {
    /// C3a. Append to the list of this height; move `current` down if the height is below it.
    fn push(&mut self, height: f32, pixel: (usize, usize)) {
        todo!("C3a: push")
    }

    /// C3b. Move `current` up past the empty lists; take the front of the first non-empty one
    /// (`None` if all lists are empty).
    fn pop(&mut self) -> Option<(usize, usize)> {
        todo!("C3b: pop")
    }
}
