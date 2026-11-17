//! Coding exercises for the unit "Linear deconvolution: inverse, Wiener, Tikhonov".
//!
//! Write your code in `src/deconv/exercise.rs`. Check it with `cargo test -p deconvolution deconv::`, and run it on
//! the micrograph with `cargo run -p deconvolution -- deconv` (pictures in `out/deconv/`).
//! `src/deconv/solution.rs` is a model solution: `cargo test -p deconvolution --features solution deconv::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p deconvolution -- deconv`.
pub mod run;
