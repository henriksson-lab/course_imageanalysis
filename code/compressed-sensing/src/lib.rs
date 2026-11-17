//! Coding exercises for the unit "Compressed sensing: fewer measurements than unknowns".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p compressed-sensing`, and run it
//! on sparse test signals with `cargo run -p compressed-sensing` (pictures in `out/compressed-sensing/`).
//! `src/solution.rs` is a model solution: `cargo test -p compressed-sensing --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
