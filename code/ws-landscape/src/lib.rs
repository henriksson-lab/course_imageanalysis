//! Coding exercises for the unit "Images as landscapes: watershed by flooding".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p ws-landscape`, and run it on
//! 1D profiles with `cargo run -p ws-landscape` (pictures in `out/ws-landscape/`).
//! `src/solution.rs` is a model solution: `cargo test -p ws-landscape --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
