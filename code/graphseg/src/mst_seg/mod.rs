//! Coding exercises for the unit "Spanning trees: single linkage and Felzenszwalb–Huttenlocher".
//!
//! Write your code in `src/mst_seg/exercise.rs`. Check it with `cargo test -p graphseg mst_seg::`, and run it with
//! `cargo run -p graphseg -- mst-seg` (results in `out/mst-seg/`). `src/mst_seg/solution.rs` is a model solution:
//! `cargo test -p graphseg --features solution mst_seg::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p graphseg -- mst-seg`.
pub mod run;
