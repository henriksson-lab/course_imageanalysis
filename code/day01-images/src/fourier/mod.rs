//! Coding exercises for the unit "Images in frequency space".
//!
//! Write your code in `src/fourier/exercise.rs`. Check it with `cargo test -p day01 fourier::`, and run it on
//! the micrograph with `cargo run -p day01 -- fourier` (pictures in `out/fourier/`).
//! `src/fourier/solution.rs` is a model solution: `cargo test -p day01 --features solution fourier::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day01 -- fourier`.
pub mod run;
