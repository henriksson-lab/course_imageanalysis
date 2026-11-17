//! Coding exercises for the unit "Compression and measurements, JPEG 2000, formats".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p compress-science`, and run it on
//! the micrograph with `cargo run -p compress-science` (pictures in `out/compress-science/`).
//! `src/solution.rs` is a model solution: `cargo test -p compress-science --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
