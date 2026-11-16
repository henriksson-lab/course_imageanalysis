//! Coding exercises for the unit "Images as arrays & geometric transforms".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p images`, and run it on
//! the micrograph with `cargo run -p images` (pictures in `out/images/`).
//! `src/solution.rs` is a model solution: `cargo test -p images --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
