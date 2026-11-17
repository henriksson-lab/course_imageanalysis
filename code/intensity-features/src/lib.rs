//! Coding exercises for the unit "Intensity inside an object".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p intensity-features`, and run it with
//! `cargo run -p intensity-features` (pictures in `out/intensity-features/`).
//! `src/solution.rs` is a model solution: `cargo test -p intensity-features --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
