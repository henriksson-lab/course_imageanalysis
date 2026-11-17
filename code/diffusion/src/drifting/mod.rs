//! Coding exercises for the unit "Drifting: generating in one step".
//!
//! Write your code in `src/drifting/exercise.rs`. Check it with `cargo test -p diffusion drifting::`, and run it
//! with `cargo run -p diffusion -- drifting` (pictures in `out/drifting/`).
//! `src/drifting/solution.rs` is a model solution: `cargo test -p diffusion --features solution drifting::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p diffusion -- drifting`.
pub mod run;
