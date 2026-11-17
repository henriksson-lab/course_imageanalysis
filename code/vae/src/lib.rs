//! Coding exercises for the day "Variational autoencoders": one module per unit.
//!
//! - `autoencoder`: From PCA to autoencoders
//! - `vae`: The variational autoencoder
//! - `vae_gen`: Generating and exploring images with a VAE
//!
//! Run a unit's program with `cargo run -p vae -- <unit>`, check it with
//! `cargo test -p vae <unit>::` (see the course page "Coding exercises: getting started").

pub mod autoencoder;
pub mod vae;
pub mod vae_gen;
