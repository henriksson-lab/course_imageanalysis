//! Coding exercises for the unit "Zernike moments on the unit disc".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p zernike`, and run it
//! with `cargo run -p zernike` (results in `out/zernike/`).
//! `src/solution.rs` is a model solution: `cargo test -p zernike --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
