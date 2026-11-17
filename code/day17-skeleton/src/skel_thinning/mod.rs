//! Coding exercises for the unit "Hit-or-miss and thinning".
//!
//! Write your code in `src/skel_thinning/exercise.rs`. Check it with `cargo test -p day17 skel_thinning::`, and run it with
//! `cargo run -p day17 -- skel-thinning` (pictures in `out/skel-thinning/`).
//! `src/skel_thinning/solution.rs` is a model solution: `cargo test -p day17 --features solution skel_thinning::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day17 -- skel-thinning`.
pub mod run;
