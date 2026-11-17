//! Coding exercises for the unit "Generating and exploring images with a VAE".
//!
//! Write your code in `src/vae_gen/exercise.rs`. Check it with `cargo test -p day38 vae_gen::`, and run it
//! with `cargo run -p day38 -- vae-gen` (pictures in `out/vae-gen/`).
//! `src/vae_gen/solution.rs` is a model solution: `cargo test -p day38 --features solution vae_gen::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day38 -- vae-gen`.
pub mod run;
