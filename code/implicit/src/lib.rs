//! Coding exercises for the unit "Curves as level sets".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p implicit`, and run it
//! with `cargo run -p implicit` (pictures in `out/implicit/`).
//! `src/solution.rs` is a model solution: `cargo test -p implicit --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
