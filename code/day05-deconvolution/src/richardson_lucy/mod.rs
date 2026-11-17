//! Coding exercises for the unit "Richardson–Lucy, 3D, and the PSF you use".
//!
//! Write your code in `src/richardson_lucy/exercise.rs`. Check it with `cargo test -p day05 richardson_lucy::`, and run it on
//! the inverted micrograph with `cargo run -p day05 -- richardson-lucy` (pictures in `out/richardson-lucy/`).
//! `src/richardson_lucy/solution.rs` is a model solution: `cargo test -p day05 --features solution richardson_lucy::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day05 -- richardson-lucy`.
pub mod run;
