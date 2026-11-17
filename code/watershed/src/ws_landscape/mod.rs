//! Coding exercises for the unit "Images as landscapes: watershed by flooding".
//!
//! Write your code in `src/ws_landscape/exercise.rs`. Check it with `cargo test -p watershed ws_landscape::`, and run it on
//! 1D profiles with `cargo run -p watershed -- ws-landscape` (pictures in `out/ws-landscape/`).
//! `src/ws_landscape/solution.rs` is a model solution: `cargo test -p watershed --features solution ws_landscape::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p watershed -- ws-landscape`.
pub mod run;
