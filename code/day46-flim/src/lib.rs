//! Coding exercises for day 46, "Fluorescence lifetime imaging (FLIM)": one module per unit.
//!
//! - `lifetime`: Lifetimes, decays and photon counting
//! - `decayfit`: Fitting decays with few photons
//! - `phasor`: Frequency domain and phasors
//! - `flim_fret`: FLIM-FRET and metabolic imaging
//!
//! Run a unit's program with `cargo run -p day46 -- <unit>`, check it with
//! `cargo test -p day46 <unit>::` (see the course page "Coding exercises: getting started").

pub mod lifetime;
pub mod decayfit;
pub mod phasor;
pub mod flim_fret;
