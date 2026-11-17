//! Coding exercises for the unit "Stereo: from disparity to depth".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p stereo`, and run it on
//! a synthetic stereo pair with `cargo run -p stereo` (pictures in `out/stereo/`).
//! `src/solution.rs` is a model solution: `cargo test -p stereo --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
