//! Coding exercises for the unit "Horn–Schunck".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p horn-schunck`, and run it on
//! the micrograph with `cargo run -p horn-schunck` (pictures in `out/horn-schunck/`).
//! `src/solution.rs` is a model solution: `cargo test -p horn-schunck --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
