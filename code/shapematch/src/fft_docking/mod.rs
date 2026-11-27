//! Coding exercises for the unit "Docking by correlation".
//!
//! Write your code in `src/fft_docking/exercise.rs`. Check it with `cargo test -p shapematch fft_docking::`,
//! and run it on a synthetic receptor and ligand with `cargo run -p shapematch -- fft-docking`
//! (pictures in `out/fft-docking/`). `src/fft_docking/solution.rs` is a model solution:
//! `cargo test -p shapematch --features solution fft_docking::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p shapematch -- fft-docking`.
pub mod run;
