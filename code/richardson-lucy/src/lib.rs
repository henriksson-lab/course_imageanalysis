//! Coding exercises for the unit "Richardson–Lucy, 3D, and the PSF you use".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p richardson-lucy`, and run it on
//! the inverted micrograph with `cargo run -p richardson-lucy` (pictures in `out/richardson-lucy/`).
//! `src/solution.rs` is a model solution: `cargo test -p richardson-lucy --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
