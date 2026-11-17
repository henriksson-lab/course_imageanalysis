//! Coding exercises for the unit "From matches to a transform".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p feature-alignment`, and run it on
//! the micrograph with `cargo run -p feature-alignment` (pictures in `out/feature-alignment/`).
//! `src/solution.rs` is a model solution: `cargo test -p feature-alignment --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
