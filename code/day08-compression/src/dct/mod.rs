//! Coding exercises for the unit "Transform coding, the DCT and the KLT".
//!
//! Write your code in `src/dct/exercise.rs`. Check it with `cargo test -p day08 dct::`, and run it on
//! the micrograph with `cargo run -p day08 -- dct` (pictures in `out/dct/`).
//! `src/dct/solution.rs` is a model solution: `cargo test -p day08 --features solution dct::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day08 -- dct`.
pub mod run;
