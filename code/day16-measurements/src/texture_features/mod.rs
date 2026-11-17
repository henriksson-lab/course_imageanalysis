//! Coding exercises for the unit "Texture: co-occurrence matrices".
//!
//! Write your code in `src/texture_features/exercise.rs`. Check it with `cargo test -p day16 texture_features::`, and run it with
//! `cargo run -p day16 -- texture-features` (pictures in `out/texture-features/`).
//! `src/texture_features/solution.rs` is a model solution: `cargo test -p day16 --features solution texture_features::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day16 -- texture-features`.
pub mod run;
