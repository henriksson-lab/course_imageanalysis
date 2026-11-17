//! Coding exercises for the unit "The variational autoencoder".
//!
//! Write your code in `src/vae/exercise.rs`. Check it with `cargo test -p vae vae::`, and run it
//! with `cargo run -p vae -- vae` (pictures in `out/vae/`).
//! `src/vae/solution.rs` is a model solution: `cargo test -p vae --features solution vae::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p vae -- vae`.
pub mod run;
