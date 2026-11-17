//! A priority queue of pixels: it hands out the pixel with the smallest value first. This is
//! plumbing for Dijkstra's algorithm, not part of the exercises.
//!
//! ```
//! # use livewire::MinQueue;
//! let mut queue = MinQueue::new();
//! queue.push(3.0, (0, 0));
//! queue.push(1.0, (5, 2));
//! assert_eq!(queue.pop(), Some((1.0, (5, 2))));
//! ```

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Pixels `(x, y)` with a value; `pop` returns the smallest value first, and of equal values
/// the one pushed first. A pixel may be pushed again with a new value: then it is in the queue
/// twice, and the caller skips the old entry when it comes out (it is already final).
#[derive(Default)]
pub struct MinQueue {
    heap: BinaryHeap<Entry>,
    pushed: u64,
}

impl MinQueue {
    pub fn new() -> Self {
        MinQueue::default()
    }

    pub fn push(&mut self, value: f32, pixel: (usize, usize)) {
        self.heap.push(Entry { value, order: self.pushed, pixel });
        self.pushed += 1;
    }

    /// The entry with the smallest value, or `None` when the queue is empty.
    pub fn pop(&mut self) -> Option<(f32, (usize, usize))> {
        self.heap.pop().map(|e| (e.value, e.pixel))
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }
}

struct Entry {
    value: f32,
    order: u64,
    pixel: (usize, usize),
}

// The standard heap gives the largest first, so compare the other way round.
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.value.total_cmp(&self.value).then(other.order.cmp(&self.order))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Entry {}
