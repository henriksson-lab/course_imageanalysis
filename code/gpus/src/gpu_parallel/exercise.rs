// Coding exercises for "Many simple cores: throughput, warps and divergence".
// Replace each `todo!()` with your code, then check it with `cargo test -p gpus gpu_parallel::`
// and look at `cargo run -p gpus -- gpu-parallel`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// The number of lanes in a warp.
pub const WARP: usize = 32;

/// C1. The cycles one warp needs for `if inside { A } else { B }`, where `inside[k]` says which
/// path lane `k` takes. A warp runs path A if any lane needs it (the others masked off), and
/// then path B if any lane needs it.
///
/// Hint: `inside.iter().any(|&b| b)` asks whether any lane takes path A.
pub fn warp_cycles(inside: &[bool], cost_a: usize, cost_b: usize) -> usize {
    todo!("C1: warp_cycles")
}

/// The result of running a branch over a whole image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Divergence {
    /// Number of warps, and how many of them ran both paths.
    pub warps: usize,
    pub diverged: usize,
    /// Cycles summed over all warps (as if they ran one after the other).
    pub cycles: usize,
    /// Useful lane-cycles ÷ issued lane-cycles (`WARP` × cycles).
    pub efficiency: f32,
}

/// C2. Run `if mask { A } else { B }` over the image with warps shaped as `warp_w` × `warp_h`
/// tiles (`warp_w · warp_h = 32`, both dividing the image size): count the warps, the
/// diverged ones, the cycles (C1 per warp) and the lane efficiency.
///
/// Hint: loop over the tiles with `(0..mask.height()).step_by(warp_h)` and
/// `(0..mask.width()).step_by(warp_w)`; collect the 32 lanes of a tile into a `Vec<bool>` and
/// call C1. A warp diverged if it took `cost_a + cost_b` cycles. Useful lane-cycles: `cost_a`
/// for every lane inside, `cost_b` for every lane outside.
pub fn divergence(mask: &Image<bool>, warp_w: usize, warp_h: usize, cost_a: usize, cost_b: usize) -> Divergence {
    todo!("C2: divergence")
}

/// C3. A loop with a different number of iterations per lane: the warp runs as long as its
/// slowest lane. Returns the warp's iterations and the lane efficiency (useful
/// lane-iterations ÷ lanes × the warp's iterations).
///
/// Hint: `iterations.iter().max()` and `iterations.iter().sum()`.
pub fn loop_efficiency(iterations: &[usize]) -> (usize, f32) {
    todo!("C3: loop_efficiency")
}

/// C4. Latency hiding on one multiprocessor that issues one instruction per cycle. Each of
/// `n_warps` warps repeats `loads` times: `k` compute instructions, then a memory load that
/// keeps it waiting `latency` cycles. Every cycle the multiprocessor issues one instruction
/// from a warp that is ready (if there is one), staying with the same warp until it has to
/// wait. Returns the fraction of cycles in which an
/// instruction was issued, from the start until the last load has arrived.
///
/// Hint: keep, per warp, the compute instructions left before its next load, the cycle at which
/// it is ready again, and its loads left. Loop over cycles while some warp has loads left: pick
/// a ready warp (the one served last if it is still ready, else the next ready one), issue one
/// instruction; after its `k`-th, it waits until `cycle + 1 + latency`. Count the busy cycles;
/// the end is when the last load arrives.
pub fn utilisation(n_warps: usize, k: usize, latency: usize, loads: usize) -> f32 {
    todo!("C4: utilisation")
}
