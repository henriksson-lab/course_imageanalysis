//! Coding exercises for the unit "Compression and measurements, JPEG 2000, formats".
//!
//! Write your code in `src/compress_science/exercise.rs`. Check it with `cargo test -p compression compress_science::`, and run it on
//! the micrograph with `cargo run -p compression -- compress-science` (pictures in `out/compress-science/`).
//! `src/compress_science/solution.rs` is a model solution: `cargo test -p compression --features solution compress_science::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p compression -- compress-science`.
pub mod run;
