//! Coding exercises for the unit "Cross-correlation with the FFT".
//!
//! Write your code in `src/xcorr_fft/exercise.rs`. Check it with `cargo test -p fftreg xcorr_fft::`, and run it on
//! the micrograph with `cargo run -p fftreg -- xcorr-fft` (pictures in `out/xcorr-fft/`).
//! `src/xcorr_fft/solution.rs` is a model solution: `cargo test -p fftreg --features solution xcorr_fft::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p fftreg -- xcorr-fft`.
pub mod run;
