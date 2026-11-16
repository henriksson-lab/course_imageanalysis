//! Coding exercises for the unit "Images in frequency space".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p fourier`, and run it on
//! the micrograph with `cargo run -p fourier` (pictures in `out/fourier/`).
//! `src/solution.rs` is a model solution: `cargo test -p fourier --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
