//! Coding exercises for the unit "Conditioning, guidance and latent diffusion".
//!
//! Write your code in `src/guidance/exercise.rs`. Check it with `cargo test -p day41 guidance::`, and run it
//! with `cargo run -p day41 -- guidance` (pictures in `out/guidance/`).
//! `src/guidance/solution.rs` is a model solution: `cargo test -p day41 --features solution guidance::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day41 -- guidance`.
pub mod run;
