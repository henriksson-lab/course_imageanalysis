//! Coding exercises for the unit "Brightness constancy and the aperture problem".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p flow-constraint`, and run it on
//! the micrograph with `cargo run -p flow-constraint` (pictures in `out/flow-constraint/`).
//! `src/solution.rs` is a model solution: `cargo test -p flow-constraint --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
