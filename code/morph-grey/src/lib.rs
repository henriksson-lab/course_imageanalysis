//! Coding exercises for the unit "Grey-level morphology: top-hat and gradient".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p morph-grey`, and run it on
//! synthetic spots and the micrograph with `cargo run -p morph-grey` (pictures in `out/morph-grey/`).
//! `src/solution.rs` is a model solution: `cargo test -p morph-grey --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
