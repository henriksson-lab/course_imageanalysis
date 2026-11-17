//! Coding exercises for the unit "Filtered back-projection and sampling".
//!
//! Write your code in `src/fbp/exercise.rs`. Check it with `cargo test -p day09 fbp::`, and run it on
//! test images with `cargo run -p day09 -- fbp` (pictures in `out/fbp/`).
//! `src/fbp/solution.rs` is a model solution: `cargo test -p day09 --features solution fbp::`.

pub mod tomo;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day09 -- fbp`.
pub mod run;
