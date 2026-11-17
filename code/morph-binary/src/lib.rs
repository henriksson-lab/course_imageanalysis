//! Coding exercises for the unit "Erosion and dilation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p morph-binary`, and run it on
//! synthetic nuclei with `cargo run -p morph-binary` (pictures in `out/morph-binary/`).
//! `src/solution.rs` is a model solution: `cargo test -p morph-binary --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
