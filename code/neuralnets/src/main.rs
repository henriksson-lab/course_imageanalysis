// The programs of this day, one per unit:  cargo run -p neuralnets -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Neural networks. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "neuralnets")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Neural networks: matrix, bend, matrix
    #[command(name = "neuralnet")]
    Neuralnet,
    /// Learning: loss, gradient descent and backpropagation
    #[command(name = "backprop")]
    Backprop,
    /// Convolutional networks
    #[command(name = "cnn")]
    Cnn,
    /// Segmentation networks
    #[command(name = "unet")]
    Unet,
}

fn main() {
    match Cli::parse().unit {
        Unit::Neuralnet => neuralnets::neuralnet::run::main(),
        Unit::Backprop => neuralnets::backprop::run::main(),
        Unit::Cnn => neuralnets::cnn::run::main(),
        Unit::Unet => neuralnets::unet::run::main(),
    }
}
