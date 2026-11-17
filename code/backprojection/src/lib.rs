//! Coding exercises for the unit "Back-projection and the Fourier slice theorem".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p backprojection`, and run it on
//! test images with `cargo run -p backprojection` (pictures in `out/backprojection/`).
//! `src/solution.rs` is a model solution: `cargo test -p backprojection --features solution`.

pub mod tomo;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
