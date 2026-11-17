//! Coding exercises for the unit "Cross-correlation with the FFT".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p xcorr-fft`, and run it on
//! the micrograph with `cargo run -p xcorr-fft` (pictures in `out/xcorr-fft/`).
//! `src/solution.rs` is a model solution: `cargo test -p xcorr-fft --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
