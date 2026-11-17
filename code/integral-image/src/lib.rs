//! Coding exercises for the unit "Integral images".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p integral-image`, and run it on
//! the micrograph with `cargo run -p integral-image` (pictures in `out/integral-image/`).
//! `src/solution.rs` is a model solution: `cargo test -p integral-image --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
