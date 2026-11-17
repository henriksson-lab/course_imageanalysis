//! Coding exercises for the unit "Least-squares unmixing".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p lsunmix`, and run it
//! with `cargo run -p lsunmix` (pictures in `out/lsunmix/`).
//! `src/solution.rs` is a model solution: `cargo test -p lsunmix --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
