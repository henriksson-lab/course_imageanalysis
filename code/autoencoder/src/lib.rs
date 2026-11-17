//! Coding exercises for the unit "From PCA to autoencoders".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p autoencoder`, and run it
//! with `cargo run -p autoencoder` (pictures in `out/autoencoder/`).
//! `src/solution.rs` is a model solution: `cargo test -p autoencoder --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
