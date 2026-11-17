//! Coding exercises for day 38, "Variational autoencoders": one module per unit.
//!
//! - `autoencoder`: From PCA to autoencoders
//! - `vae`: The variational autoencoder
//! - `vae_gen`: Generating and exploring images with a VAE
//!
//! Run a unit's program with `cargo run -p day38 -- <unit>`, check it with
//! `cargo test -p day38 <unit>::` (see the course page "Coding exercises: getting started").

pub mod autoencoder;
pub mod vae;
pub mod vae_gen;
