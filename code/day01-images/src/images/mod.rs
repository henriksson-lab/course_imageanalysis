//! Coding exercises for the unit "Images as arrays & geometric transforms".
//!
//! Write your code in `src/images/exercise.rs`. Check it with `cargo test -p day01 images::`, and run it on
//! the micrograph with `cargo run -p day01 -- images` (pictures in `out/images/`).
//! `src/images/solution.rs` is a model solution: `cargo test -p day01 --features solution images::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day01 -- images`.
pub mod run;
