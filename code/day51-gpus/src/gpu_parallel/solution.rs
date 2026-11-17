// Model solution for the coding exercises of "Many simple cores: throughput, warps and divergence". The GPU here is a
// model with the rules of a real one (warps of 32 lanes, one instruction stream per warp,
// warps that wait for memory), simulated in plain Rust.

use imgcourse::Image;

/// The number of lanes in a warp.
pub const WARP: usize = 32;

/// C1. The cycles one warp needs for `if inside { A } else { B }`, where `inside[k]` says which
/// path lane `k` takes. A warp runs path A if any lane needs it (the others masked off), and
/// then path B if any lane needs it.
pub fn warp_cycles(inside: &[bool], cost_a: usize, cost_b: usize) -> usize {
    let any_a = inside.iter().any(|&b| b);
    let any_b = inside.iter().any(|&b| !b);
    let mut cycles = 0;
    if any_a {
        cycles += cost_a;
    }
    if any_b {
        cycles += cost_b;
    }
    cycles
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
pub fn divergence(mask: &Image<bool>, warp_w: usize, warp_h: usize, cost_a: usize, cost_b: usize) -> Divergence {
    let (mut warps, mut diverged, mut cycles, mut useful) = (0, 0, 0, 0);
    for ty in (0..mask.height()).step_by(warp_h) {
        for tx in (0..mask.width()).step_by(warp_w) {
            // The lanes of this warp: the pixels of one tile.
            let mut lanes = Vec::with_capacity(WARP);
            for y in ty..ty + warp_h {
                for x in tx..tx + warp_w {
                    lanes.push(mask[(x, y)]);
                }
            }
            let c = warp_cycles(&lanes, cost_a, cost_b);
            warps += 1;
            cycles += c;
            if c == cost_a + cost_b {
                diverged += 1;
            }
            useful += lanes.iter().map(|&a| if a { cost_a } else { cost_b }).sum::<usize>();
        }
    }
    Divergence { warps, diverged, cycles, efficiency: useful as f32 / (WARP * cycles) as f32 }
}

/// C3. A loop with a different number of iterations per lane: the warp runs as long as its
/// slowest lane. Returns the warp's iterations and the lane efficiency (useful
/// lane-iterations ÷ lanes × the warp's iterations).
pub fn loop_efficiency(iterations: &[usize]) -> (usize, f32) {
    let warp = iterations.iter().copied().max().unwrap_or(0);
    let useful: usize = iterations.iter().sum();
    (warp, useful as f32 / (iterations.len() * warp) as f32)
}

/// C4. Latency hiding on one multiprocessor that issues one instruction per cycle. Each of
/// `n_warps` warps repeats `loads` times: `k` compute instructions, then a memory load that
/// keeps it waiting `latency` cycles. Every cycle the multiprocessor issues one instruction
/// from a warp that is ready (if there is one), staying with the same warp until it has to
/// wait. Returns the fraction of cycles in which an
/// instruction was issued, from the start until the last load has arrived.
pub fn utilisation(n_warps: usize, k: usize, latency: usize, loads: usize) -> f32 {
    // Per warp: the compute instructions left before its next load, the cycle at which it
    // is ready again, and the loads it still has to do.
    let mut left = vec![k; n_warps];
    let mut ready_at = vec![0; n_warps];
    let mut loads_left = vec![loads; n_warps];
    let (mut cycle, mut busy, mut current) = (0, 0, 0);
    while loads_left.iter().any(|&l| l > 0) {
        // Stay with the current warp while it is ready; else take the next ready one.
        let ready = (0..n_warps).map(|i| (current + i) % n_warps).find(|&w| loads_left[w] > 0 && ready_at[w] <= cycle);
        if let Some(w) = ready {
            busy += 1;
            left[w] -= 1;
            if left[w] == 0 {
                // The last compute instruction is followed by a load: wait for it.
                ready_at[w] = cycle + 1 + latency;
                left[w] = k;
                loads_left[w] -= 1;
            }
            current = w;
        }
        cycle += 1;
    }
    let end = ready_at.iter().copied().max().unwrap_or(0).max(cycle);
    busy as f32 / end as f32
}
