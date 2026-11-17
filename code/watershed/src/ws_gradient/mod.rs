//! Coding exercises for the unit "Watershed segmentation of a real image".
//!
//! Write your code in `src/ws_gradient/exercise.rs`. Check it with `cargo test -p watershed ws_gradient::`, and run it on
//! the micrograph with `cargo run -p watershed -- ws-gradient` (pictures in `out/ws-gradient/`).
//! `src/ws_gradient/solution.rs` is a model solution: `cargo test -p watershed --features solution ws_gradient::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p watershed -- ws-gradient`.
pub mod run;
