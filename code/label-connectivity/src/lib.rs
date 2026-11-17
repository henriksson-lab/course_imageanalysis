//! Coding exercises for the unit "Connectivity and labelling".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p label-connectivity`, and run it on
//! synthetic nuclei and the micrograph with `cargo run -p label-connectivity` (pictures in `out/label-connectivity/`).
//! `src/solution.rs` is a model solution: `cargo test -p label-connectivity --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
