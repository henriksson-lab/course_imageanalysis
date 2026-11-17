//! Coding exercises for the day "Graph cuts": one module per unit.
//!
//! - `seg_energy`: Segmentation as energy minimisation
//! - `maxflow`: Graphs, cuts and maximum flow
//! - `graphcut_seg`: Graph-cut segmentation
//! - `ncut`: Spectral partitioning and normalized cuts
//!
//! Run a unit's program with `cargo run -p graphcuts -- <unit>`, check it with
//! `cargo test -p graphcuts <unit>::` (see the course page "Coding exercises: getting started").

pub mod common;
pub mod seg_energy;
pub mod maxflow;
pub mod graphcut_seg;
pub mod ncut;
