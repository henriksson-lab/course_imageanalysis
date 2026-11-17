//! Coding exercises for the day "Spectral unmixing": one module per unit.
//!
//! - `crosstalk`: Bleed-through and compensation
//! - `lsunmix`: Least-squares unmixing
//! - `endmembers`: Endmembers: the geometry of mixtures
//! - `stain_unmixing`: Stain separation: colour deconvolution
//!
//! Run a unit's program with `cargo run -p unmixing -- <unit>`, check it with
//! `cargo test -p unmixing <unit>::` (see the course page "Coding exercises: getting started").

pub mod crosstalk;
pub mod lsunmix;
pub mod endmembers;
pub mod stain_unmixing;
