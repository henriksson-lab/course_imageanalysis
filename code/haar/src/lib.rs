//! Coding exercises for the unit "Localised basis functions and the Haar transform".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p haar`, and run it
//! with `cargo run -p haar` (pictures in `out/haar/`).
//! `src/solution.rs` is a model solution: `cargo test -p haar --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
