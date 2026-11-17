//! Coding exercises for the unit "Fine-tuning strategies".
//!
//! Write your code in `src/finetune_strategies/exercise.rs`. Check it with `cargo test -p finetuning finetune_strategies::`, and run it
//! with `cargo run -p finetuning -- finetune-strategies` (pictures in `out/finetune-strategies/`).
//! `src/finetune_strategies/solution.rs` is a model solution: `cargo test -p finetuning --features solution finetune_strategies::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p finetuning -- finetune-strategies`.
pub mod run;
