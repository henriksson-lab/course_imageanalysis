//! Coding exercises for the unit "Closed boundaries: DP in polar coordinates".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p polar-boundary`, and run it with
//! `cargo run -p polar-boundary` (pictures in `out/polar-boundary/`).
//! `src/solution.rs` is a model solution: `cargo test -p polar-boundary --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
