// The programs of this day, one per unit:  cargo run -p gmm -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Gaussian mixtures. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "gmm")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Feature spaces & k-means
    #[command(name = "kmeans")]
    Kmeans,
    /// The multivariate Gaussian & mixtures
    #[command(name = "gaussian-mixture")]
    GaussianMixture,
    /// Fitting a mixture: EM
    #[command(name = "em")]
    Em,
    /// Mixtures on images
    #[command(name = "gmm-images")]
    GmmImages,
}

fn main() {
    match Cli::parse().unit {
        Unit::Kmeans => gmm::kmeans::run::main(),
        Unit::GaussianMixture => gmm::gaussian_mixture::run::main(),
        Unit::Em => gmm::em::run::main(),
        Unit::GmmImages => gmm::gmm_images::run::main(),
    }
}
