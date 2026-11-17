//! Coding exercises for the unit "Edges: Gaussian derivatives and Canny".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p canny`, and run it on
//! the micrograph with `cargo run -p canny` (pictures in `out/canny/`).
//! `src/solution.rs` is a model solution: `cargo test -p canny --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
