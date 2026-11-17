//! Coding exercises for the unit "Shape features".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p shape-features`, and run it on
//! synthetic nuclei with `cargo run -p shape-features` (pictures in `out/shape-features/`).
//! `src/solution.rs` is a model solution: `cargo test -p shape-features --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
