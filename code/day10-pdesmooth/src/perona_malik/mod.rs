//! Coding exercises for the unit "Edge-stopping diffusion (Perona–Malik)".
//!
//! Write your code in `src/perona_malik/exercise.rs`. Check it with `cargo test -p day10 perona_malik::`, and run it on
//! the images with `cargo run -p day10 -- perona-malik` (pictures in `out/perona-malik/`).
//! `src/perona_malik/solution.rs` is a model solution: `cargo test -p day10 --features solution perona_malik::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day10 -- perona-malik`.
pub mod run;
