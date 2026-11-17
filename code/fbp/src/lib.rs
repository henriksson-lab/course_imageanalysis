//! Coding exercises for the unit "Filtered back-projection and sampling".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p fbp`, and run it on
//! test images with `cargo run -p fbp` (pictures in `out/fbp/`).
//! `src/solution.rs` is a model solution: `cargo test -p fbp --features solution`.

pub mod tomo;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
