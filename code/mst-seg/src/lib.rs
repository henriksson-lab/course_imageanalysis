//! Coding exercises for the unit "Spanning trees: single linkage and Felzenszwalb–Huttenlocher".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p mst-seg`, and run it with
//! `cargo run -p mst-seg` (results in `out/mst-seg/`). `src/solution.rs` is a model solution:
//! `cargo test -p mst-seg --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
