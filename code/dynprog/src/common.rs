//! Helpers shared by the units of day 32 (plumbing, not part of the exercises): a priority
//! queue of pixels for Dijkstra's algorithm and fast marching (it hands out the pixel with the
//! smallest value first), and Gaussian smoothing for the programs.
//!
//! ```
//! # use dynprog::common::MinQueue;
//! let mut queue = MinQueue::new();
//! queue.push(3.0, (0, 0));
//! queue.push(1.0, (5, 2));
//! assert_eq!(queue.pop(), Some((1.0, (5, 2))));
//! ```

use imgcourse::{Border, Image};
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

/// Gaussian smoothing in two passes (from the filtering unit; not part of these exercises),
/// cut at 3σ, copying the edge pixel outside the image.
pub fn smooth(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for d in -r..=r {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += w[(d + r) as usize] * src.get_border(xx, yy, Border::Clamp);
            }
            sum / total
        })
    };
    pass(&pass(img, true), false)
}
