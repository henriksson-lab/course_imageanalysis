//! Coding exercises for the unit "Superpixels: SLIC and smaller graphs".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p superpixels`, and run it with
//! `cargo run -p superpixels` (results in `out/superpixels/`). `src/solution.rs` is a model solution:
//! `cargo test -p superpixels --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
