//! Coding exercises for the unit "Camera noise, SNR and PSNR".
//!
//! Write your code in `src/cameranoise/exercise.rs`. Check it with `cargo test -p day03 cameranoise::`, and run it
//! with `cargo run -p day03 -- cameranoise` (pictures in `out/cameranoise/`).
//! `src/cameranoise/solution.rs` is a model solution: `cargo test -p day03 --features solution cameranoise::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day03 -- cameranoise`.
pub mod run;
