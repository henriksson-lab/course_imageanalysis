//! Coding exercises for the unit "Tracing fast: boxes, trees and grids".
//!
//! Write your code in `src/bvh/exercise.rs`. Check it with `cargo test -p day50 bvh::`, and run it on
//! random 2D scenes with `cargo run -p day50 -- bvh` (numbers on the terminal, pictures in `out/bvh/`).
//! `src/bvh/solution.rs` is a model solution: `cargo test -p day50 --features solution bvh::`.
//! `src/scene2d.rs` holds the boxes, discs, the tree type and the random scenes.

mod scene2d;
pub use scene2d::*;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day50 -- bvh`.
pub mod run;
