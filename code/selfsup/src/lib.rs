//! Coding exercises for the day "Learning without labels": one module per unit.
//!
//! - `pretext`: Pretext tasks and masked image modelling
//! - `contrastive`: Contrastive learning
//! - `self_distillation`: Self-distillation and foundation models
//! - `self_denoise`: Self-supervised denoising
//!
//! Run a unit's program with `cargo run -p selfsup -- <unit>`, check it with
//! `cargo test -p selfsup <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod pretext;
pub mod contrastive;
pub mod self_distillation;
pub mod self_denoise;
