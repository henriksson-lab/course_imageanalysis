//! Coding exercises for the unit "Hit-or-miss and thinning".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p skel-thinning`, and run it with
//! `cargo run -p skel-thinning` (pictures in `out/skel-thinning/`).
//! `src/solution.rs` is a model solution: `cargo test -p skel-thinning --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
