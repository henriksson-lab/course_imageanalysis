//! Coding exercises for the unit "Localised basis functions and the Haar transform".
//!
//! Write your code in `src/haar/exercise.rs`. Check it with `cargo test -p day07 haar::`, and run it
//! with `cargo run -p day07 -- haar` (pictures in `out/haar/`).
//! `src/haar/solution.rs` is a model solution: `cargo test -p day07 --features solution haar::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day07 -- haar`.
pub mod run;
