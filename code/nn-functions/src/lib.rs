//! Coding exercises for the unit "Nearest neighbours and empty space: G, F and J".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p nn-functions`, and run it
//! with `cargo run -p nn-functions` (results in `out/nn-functions/`).
//! `src/solution.rs` is a model solution: `cargo test -p nn-functions --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
