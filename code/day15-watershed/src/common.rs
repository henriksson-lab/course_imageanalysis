//! Helpers shared by the units of day 15 (given code, not part of the exercises).

use std::cmp::Ordering;

/// The 8 neighbours `(dx, dy)` in reading order.
pub const N8: [(isize, isize); 8] = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

/// An entry of a flooding priority queue (a `BinaryHeap` of `Reverse(Entry)`): the height, an
/// arrival number that breaks ties between equal heights, and the pixel.
#[derive(PartialEq)]
pub struct Entry(pub f32, pub i64, pub (usize, usize));
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}
