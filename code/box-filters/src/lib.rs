//! Coding exercises for the unit "Box filters & local statistics".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p box-filters`, and run it on
//! the micrograph with `cargo run -p box-filters` (pictures in `out/box-filters/`).
//! `src/solution.rs` is a model solution: `cargo test -p box-filters --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
