//! Coding exercises for the unit "Multiresolution, filter banks and Daubechies wavelets".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p filterbank`, and run it
//! with `cargo run -p filterbank` (pictures in `out/filterbank/`).
//! `src/solution.rs` is a model solution: `cargo test -p filterbank --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
