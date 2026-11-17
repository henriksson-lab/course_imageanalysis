//! Coding exercises for the day "How GPUs work": one module per unit.
//!
//! - `gpu_parallel`: Many simple cores: throughput, warps and divergence
//! - `gpu_memory`: Memory is the bottleneck
//! - `gpu_patterns`: Data-parallel patterns
//! - `gpu_fit`: What GPUs are good and bad at
//!
//! Run a unit's program with `cargo run -p gpus -- <unit>`, check it with
//! `cargo test -p gpus <unit>::` (see the course page "Coding exercises: getting started").

pub mod gpu_parallel;
pub mod gpu_memory;
pub mod gpu_patterns;
pub mod gpu_fit;
