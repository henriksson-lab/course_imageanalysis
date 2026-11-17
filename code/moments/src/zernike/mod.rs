//! Coding exercises for the unit "Zernike moments on the unit disc".
//!
//! Write your code in `src/zernike/exercise.rs`. Check it with `cargo test -p moments zernike::`, and run it
//! with `cargo run -p moments -- zernike` (results in `out/zernike/`).
//! `src/zernike/solution.rs` is a model solution: `cargo test -p moments --features solution zernike::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p moments -- zernike`.
pub mod run;
