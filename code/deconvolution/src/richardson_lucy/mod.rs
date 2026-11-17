//! Coding exercises for the unit "Richardson–Lucy, 3D, and the PSF you use".
//!
//! Write your code in `src/richardson_lucy/exercise.rs`. Check it with `cargo test -p deconvolution richardson_lucy::`, and run it on
//! the inverted micrograph with `cargo run -p deconvolution -- richardson-lucy` (pictures in `out/richardson-lucy/`).
//! `src/richardson_lucy/solution.rs` is a model solution: `cargo test -p deconvolution --features solution richardson_lucy::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p deconvolution -- richardson-lucy`.
pub mod run;
