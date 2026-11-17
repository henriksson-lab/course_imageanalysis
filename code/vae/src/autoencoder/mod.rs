//! Coding exercises for the unit "From PCA to autoencoders".
//!
//! Write your code in `src/autoencoder/exercise.rs`. Check it with `cargo test -p vae autoencoder::`, and run it
//! with `cargo run -p vae -- autoencoder` (pictures in `out/autoencoder/`).
//! `src/autoencoder/solution.rs` is a model solution: `cargo test -p vae --features solution autoencoder::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p vae -- autoencoder`.
pub mod run;
