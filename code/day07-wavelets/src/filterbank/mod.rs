//! Coding exercises for the unit "Multiresolution, filter banks and Daubechies wavelets".
//!
//! Write your code in `src/filterbank/exercise.rs`. Check it with `cargo test -p day07 filterbank::`, and run it
//! with `cargo run -p day07 -- filterbank` (pictures in `out/filterbank/`).
//! `src/filterbank/solution.rs` is a model solution: `cargo test -p day07 --features solution filterbank::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day07 -- filterbank`.
pub mod run;
