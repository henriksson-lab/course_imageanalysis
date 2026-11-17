//! Coding exercises for the unit "Region properties and moments".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p regionprops`, and run it on
//! synthetic nuclei and digital discs with `cargo run -p regionprops` (pictures in `out/regionprops/`).
//! `src/solution.rs` is a model solution: `cargo test -p regionprops --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
