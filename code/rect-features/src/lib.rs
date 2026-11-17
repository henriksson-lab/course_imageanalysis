//! Coding exercises for the unit "Rectangle features".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p rect-features`, and run it on
//! the micrograph with `cargo run -p rect-features` (pictures in `out/rect-features/`).
//! `src/solution.rs` is a model solution: `cargo test -p rect-features --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
