//! Coding exercises for the unit "Opening, closing and the distance transform".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p morph-openclose`, and run it on
//! synthetic images with `cargo run -p morph-openclose` (pictures in `out/morph-openclose/`).
//! `src/solution.rs` is a model solution: `cargo test -p morph-openclose --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
