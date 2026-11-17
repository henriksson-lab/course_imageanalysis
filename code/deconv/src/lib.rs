//! Coding exercises for the unit "Linear deconvolution: inverse, Wiener, Tikhonov".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p deconv`, and run it on
//! the micrograph with `cargo run -p deconv` (pictures in `out/deconv/`).
//! `src/solution.rs` is a model solution: `cargo test -p deconv --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
