//! Coding exercises for the unit "Rotation and scale become shifts: Fourier–Mellin".
//!
//! Write your code in `src/fourier_mellin/exercise.rs`. Check it with `cargo test -p shapematch fourier_mellin::`,
//! and run it on the micrograph with `cargo run -p shapematch -- fourier-mellin` (pictures in `out/fourier-mellin/`).
//! `src/fourier_mellin/solution.rs` is a model solution: `cargo test -p shapematch --features solution fourier_mellin::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p shapematch -- fourier-mellin`.
pub mod run;
