//! Coding exercises for the unit "Lifetimes, decays and photon counting".
//!
//! Write your code in `src/lifetime/exercise.rs`. Check it with `cargo test -p day46 lifetime::`, and run it
//! with `cargo run -p day46 -- lifetime` (pictures in `out/lifetime/`).
//! `src/lifetime/solution.rs` is a model solution: `cargo test -p day46 --features solution lifetime::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day46 -- lifetime`.
pub mod run;
