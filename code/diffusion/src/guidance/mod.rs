//! Coding exercises for the unit "Conditioning, guidance and latent diffusion".
//!
//! Write your code in `src/guidance/exercise.rs`. Check it with `cargo test -p diffusion guidance::`, and run it
//! with `cargo run -p diffusion -- guidance` (pictures in `out/guidance/`).
//! `src/guidance/solution.rs` is a model solution: `cargo test -p diffusion --features solution guidance::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p diffusion -- guidance`.
pub mod run;
