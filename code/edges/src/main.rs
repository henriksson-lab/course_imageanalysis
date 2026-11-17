// The programs of this day, one per unit:  cargo run -p edges -- <unit>
// (add --features solution before -- to run the model solution). `--help` lists the units.

use clap::{Parser, Subcommand};

/// Edges, blobs and ridges. Runs one unit's program; the pictures go to out/<unit>/.
#[derive(Parser)]
#[command(name = "edges")]
struct Cli {
    #[command(subcommand)]
    unit: Unit,
}

#[derive(Subcommand)]
enum Unit {
    /// Edges: Gaussian derivatives and Canny
    #[command(name = "canny")]
    Canny,
    /// Scale space, blobs and pyramids
    #[command(name = "scale-space")]
    ScaleSpace,
    /// The Hessian: blobs, ridges and saddles
    #[command(name = "hessian")]
    Hessian,
    /// The structure tensor: orientation and corners
    #[command(name = "structure-tensor")]
    StructureTensor,
}

fn main() {
    match Cli::parse().unit {
        Unit::Canny => edges::canny::run::main(),
        Unit::ScaleSpace => edges::scale_space::run::main(),
        Unit::Hessian => edges::hessian::run::main(),
        Unit::StructureTensor => edges::structure_tensor::run::main(),
    }
}
