//! Coding exercises for the unit "Linear deconvolution: inverse, Wiener, Tikhonov".
//!
//! Write your code in `src/deconv/exercise.rs`. Check it with `cargo test -p day05 deconv::`, and run it on
//! the micrograph with `cargo run -p day05 -- deconv` (pictures in `out/deconv/`).
//! `src/deconv/solution.rs` is a model solution: `cargo test -p day05 --features solution deconv::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day05 -- deconv`.
pub mod run;
