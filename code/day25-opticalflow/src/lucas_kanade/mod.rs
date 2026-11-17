//! Coding exercises for the unit "Lucas–Kanade".
//!
//! Write your code in `src/lucas_kanade/exercise.rs`. Check it with `cargo test -p day25 lucas_kanade::`, and run it on
//! the micrograph with `cargo run -p day25 -- lucas-kanade` (pictures in `out/lucas-kanade/`).
//! `src/lucas_kanade/solution.rs` is a model solution: `cargo test -p day25 --features solution lucas_kanade::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day25 -- lucas-kanade`.
pub mod run;
