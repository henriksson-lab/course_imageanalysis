//! Coding exercises for the unit "Active shape models".
//!
//! Write your code in `src/asm/exercise.rs`. Check it with `cargo test -p day30 asm::`, and run it
//! with `cargo run -p day30 -- asm` (pictures in `out/asm/`).
//! `src/asm/solution.rs` is a model solution: `cargo test -p day30 --features solution asm::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day30 -- asm`.
pub mod run;
