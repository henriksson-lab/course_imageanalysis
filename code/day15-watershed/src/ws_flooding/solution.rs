// Model solution for the coding exercises of "The flooding algorithm on a grid".

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
pub fn regional_minima(f: &Image<f32>, eight: bool) -> (Image<u32>, u32) {
    let (w, h) = (f.width(), f.height());
    let mut seen = Image::new(w, h, false);
    let mut labels = Image::new(w, h, 0u32);
    let mut count = 0;
    for start in f.coords() {
        if seen[start] {
            continue;
        }
        // collect the plateau of `start` by a flood fill over equal heights
        let v = f[start];
        let mut plateau = vec![start];
        let mut queue = VecDeque::from([start]);
        seen[start] = true;
        let mut is_minimum = true;
        while let Some((x, y)) = queue.pop_front() {
            for &(dx, dy) in neighbours(eight) {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                if !f.contains(nx, ny) {
                    continue;
                }
                let q = (nx as usize, ny as usize);
                if f[q] < v {
                    is_minimum = false;
                } else if f[q] == v && !seen[q] {
                    seen[q] = true;
                    plateau.push(q);
                    queue.push_back(q);
                }
            }
        }
        if is_minimum {
            count += 1;
            for p in plateau {
                labels[p] = count;
            }
        }
    }
    (labels, count)
}

/// Queue the neighbours of `p` that have no label and are not yet queued, remembering that
/// they were queued by `p`'s label.
fn push_neighbours(f: &Image<f32>, p: (usize, usize), eight: bool, labels: &Image<u32>, queued: &mut Image<bool>, by: &mut Image<u32>, queue: &mut impl FloodQueue) {
    for &(dx, dy) in neighbours(eight) {
        let (nx, ny) = (p.0 as isize + dx, p.1 as isize + dy);
        if f.contains(nx, ny) {
            let q = (nx as usize, ny as usize);
            if labels[q] == 0 && !queued[q] {
                queued[q] = true;
                by[q] = labels[p];
                queue.push(f[q], q);
            }
        }
    }
}

/// C2. Meyer's flooding from the `markers` (0 = no marker). Every unlabelled neighbour of a
/// marker enters the queue (markers in raster order, neighbours in reading order). Then pop
/// the lowest pixel: if `line` is true and its labelled neighbours carry different labels, it
/// becomes a line pixel (and pushes nothing); otherwise it takes the label of the pixel that
/// queued it. Then it pushes its unlabelled neighbours that are not yet queued.
/// Returns the labels (0 = watershed line) and the pixels in the order they were popped.
pub fn watershed(f: &Image<f32>, markers: &Image<u32>, eight: bool, line: bool, queue: &mut impl FloodQueue) -> (Image<u32>, Vec<(usize, usize)>) {
    let (w, h) = (f.width(), f.height());
    let mut labels = markers.clone();
    let mut queued = markers.map(|m| m > 0);
    let mut by = Image::new(w, h, 0u32); // the label of the pixel that queued it
    let mut order = Vec::new();

    // start: the neighbours of all markers
    for p in f.coords() {
        if markers[p] > 0 {
            push_neighbours(f, p, eight, &labels, &mut queued, &mut by, queue);
        }
    }
    while let Some(p) = queue.pop() {
        order.push(p);
        // the labels of the labelled neighbours
        let mut first = 0;
        let mut different = false;
        for &(dx, dy) in neighbours(eight) {
            let (nx, ny) = (p.0 as isize + dx, p.1 as isize + dy);
            if f.contains(nx, ny) {
                let l = labels[(nx as usize, ny as usize)];
                if l != 0 && l != LINE {
                    if first == 0 {
                        first = l;
                    } else if l != first {
                        different = true;
                    }
                }
            }
        }
        if line && different {
            labels[p] = LINE; // a dam: it pushes nothing
            continue;
        }
        labels[p] = by[p];
        push_neighbours(f, p, eight, &labels, &mut queued, &mut by, queue);
    }
    (labels.map(|l| if l == LINE { 0 } else { l }), order)
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
        let b = height as usize;
        self.buckets[b].push_back(pixel);
        self.current = self.current.min(b);
    }

    /// C3b. Move `current` up past the empty lists; take the front of the first non-empty one
    /// (`None` if all lists are empty).
    fn pop(&mut self) -> Option<(usize, usize)> {
        while self.current < self.buckets.len() && self.buckets[self.current].is_empty() {
            self.current += 1;
        }
        if self.current == self.buckets.len() {
            return None;
        }
        self.buckets[self.current].pop_front()
    }
}
