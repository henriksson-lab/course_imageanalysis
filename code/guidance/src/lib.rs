//! Coding exercises for the unit "Conditioning, guidance and latent diffusion".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p guidance`, and run it
//! with `cargo run -p guidance` (pictures in `out/guidance/`).
//! `src/solution.rs` is a model solution: `cargo test -p guidance --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
