//! Coding exercises for the unit "Horn–Schunck".
//!
//! Write your code in `src/horn_schunck/exercise.rs`. Check it with `cargo test -p day25 horn_schunck::`, and run it on
//! the micrograph with `cargo run -p day25 -- horn-schunck` (pictures in `out/horn-schunck/`).
//! `src/horn_schunck/solution.rs` is a model solution: `cargo test -p day25 --features solution horn_schunck::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day25 -- horn-schunck`.
pub mod run;
