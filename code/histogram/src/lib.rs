//! Coding exercises for the unit "Histograms and a global threshold".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p histogram`, and run it on
//! synthetic blobs and the micrograph with `cargo run -p histogram` (pictures in `out/histogram/`).
//! `src/solution.rs` is a model solution: `cargo test -p histogram --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
