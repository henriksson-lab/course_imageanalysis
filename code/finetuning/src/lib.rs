//! Coding exercises for the day "Fine-tuning and transfer learning": one module per unit.
//!
//! - `transfer`: Transfer learning
//! - `finetune_strategies`: Fine-tuning strategies
//! - `few_labels`: Few labels: annotation and the human in the loop
//! - `domain_shift`: Domain shift and adaptation
//!
//! Run a unit's program with `cargo run -p finetuning -- <unit>`, check it with
//! `cargo test -p finetuning <unit>::` (see the course page "Coding exercises: getting started").
//! `common` holds what several units share: the toy patches, a small network and its pretraining.

pub mod common;
pub mod transfer;
pub mod finetune_strategies;
pub mod few_labels;
pub mod domain_shift;
