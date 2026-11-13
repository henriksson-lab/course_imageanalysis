# Plan: a day on how GPUs work, and what they are good and bad at

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-12.

**Course map (2026-11-13, self-study restructure):** day 51 of 51, Part IX "3D and computing". Hard prerequisites: Bit depth and noise (day 3). Soft (see also): 3D graphics (day 48). Source: `days.yml`.

**Earlier placement (2026-11-12):** Part IX (row marked *requested, not
yet planned*; this plan answers it). Entry after day 3 (bit depth and noise); the 3D
graphics day helps (its `rasterise/` "GPU pipeline" slide, `gpuformats/` and
`volrender/`) but is not required. Links named in the course map: 3D graphics (the pipeline
GPUs come from; `gpuformats/`), integral images (parallel scan), neural networks.

## Goal of the day

After the day a student can:

1. explain the difference between a processor built for **latency** (few complex cores,
   big caches, branch prediction) and one built for **throughput** (thousands of simple
   lanes), and estimate from "time = start-up + amount / rate" when each one wins;
2. describe **SIMT** execution: threads grouped in **warps** of 32 lanes that run one
   instruction at a time, **branch divergence** (both sides of an `if` are run, with lanes
   masked) and its cost, and how a GPU **hides memory latency** by switching between many
   resident warps (**occupancy**);
3. draw the **memory hierarchy** (registers, shared memory, caches, global memory, and the
   link to the CPU), say why **memory bandwidth, not arithmetic, limits most image code**,
   count memory transactions for **coalesced** and strided access, and explain **tiling**
   (shared memory with a halo) for a stencil;
4. compute the **arithmetic intensity** of an operation (FLOPs per byte moved), place it on a
   **roofline** plot, and decide whether it is memory- or compute-bound; include
   **CPU ↔ GPU transfers** and launch overhead in an end-to-end time and find the image size
   where the GPU starts to pay off;
5. run the **data-parallel patterns** by hand on 8–16 numbers: **map**, **stencil**,
   **reduction** (a tree in $\log_2 n$ steps), **prefix scan** (Hillis–Steele and the
   work-efficient Blelloch scan), **histogram with atomic additions** (and contention), and
   state their work and depth;
6. explain **floating-point precision on GPUs** (float32 as the default, float64 often far
   slower, float16/bfloat16 halving the bytes, matrix units for low-precision matrix
   products, mixed precision), recalling the bit-depth day, and why a parallel sum can give
   a **different result each run**;
7. judge for a given image method whether it **maps well** to a GPU (filters, FFT,
   element-wise operations, neural networks, ray marching) or **badly** (watershed priority
   queues, union–find labelling, max-flow, dynamic programming with dependencies, small
   problems), name the property that decides it, and know the usual parallel workaround;
8. *(math)* derive the roofline bound, Little's law for the number of bytes in flight,
   the probability that a warp diverges, the work and depth of reduction and scan, Brent's
   bound, and Amdahl's law.

Why it fits this audience: almost everything is counting and a few one-line formulas (a
minimum of two lines on a log–log plot, a geometric series, a binomial probability), applied
to image operations they already know. It answers the question every student asks after the
first slow filter on a big stack ("would a GPU help?") with a method instead of a slogan, and
it explains why deep learning and GPUs grew up together. No programming is needed: the
machine is shown as pictures, simulations and counts, and one demo measures the real GPU in
the browser.

## Prerequisite gap

- **No computer architecture is assumed** (audience.md: programming is not needed). Unit 1
  opens with a three-slide primer: a processor runs instructions at a clock rate; data lives
  in memory far away and in small caches nearby; a memory access costs hundreds of clock
  cycles. Kept to pictures and numbers, no assembly.
- **Images as arrays, convolution, border rules** (`images/`, `kernels/`, day 1): the
  stencil pattern and the filter used for timing are the 3×3 and k×k kernels of `kernels/`.
  One-line recall.
- **Memory layout** (`highbit/`, day 3: "an image is a row of bytes", row stride, interleaved
  vs planar channels): unit 2 builds coalescing on it. One recall slide.
- **Floating point** (`floatingpoint/`, day 3: float32/float16/bfloat16 fields, spacing,
  "summing many pixels", pairwise vs naive summation; demo `floatbits`): unit 3's tree
  reduction **is** pairwise summation, and unit 4's precision slides recall the format
  table. Applied here, not re-taught.
- **FFT cost** (`fourier/`, day 1; `xcorr-fft/` "What it costs" on day 23): "FFT is
  $O(n\log n)$" is stated as a recall; the FFT's data-flow picture (butterflies in
  $\log_2 n$ passes) is drawn here in one slide.
- **3D graphics** (helps): `rasterise/` already says "every pixel is tested
  independently: why GPUs are fast" and has the slide "The GPU pipeline, conceptually";
  `gpuformats/` covers texture formats and filtering in hardware; `volrender/` ray-marches
  on the GPU with early termination. Unit 1 recalls the pipeline in one slide for those who
  took it, and stands on its own for those who did not.
- **Later days used as examples in unit 4** (watershed day 15, labelling day 14, graph cuts
  day 27, neural networks day 36, dynamic programming day): a student entering after day 3
  has not seen them. Each example is therefore one self-contained picture ("a queue that
  always serves the lowest pixel next", "follow parent pointers to a root") with a pointer
  forward ("the watershed day does this in full"); none needs the method's details.

## Overlaps / who owns what

This day **owns**: throughput vs latency, SIMT/warps/divergence, occupancy and latency
hiding, the GPU memory hierarchy, coalescing and tiling, arithmetic intensity and the
roofline, CPU ↔ GPU transfers and launch overhead, the data-parallel patterns (map, stencil,
reduction, **parallel scan**, histogram with atomics), work/depth and Brent's bound,
Amdahl's law, low-precision arithmetic **on GPUs** (rates, matrix units, mixed precision,
reproducibility of parallel sums), and the "good and bad on a GPU" judgement. Others recall.

| Topic | Owner | Here |
|---|---|---|
| IEEE 754 formats, spacing, naive/pairwise/Kahan summation | `floatingpoint/` (day 3) | recall; tree reduction = pairwise sum; GPU rates and mixed precision are new here |
| Row of bytes, stride, interleaved vs planar | `highbit/` (day 3) | recall; coalescing and "planar is faster on a GPU" new here |
| GPU graphics pipeline (vertex/fragment stages, rasterisation) | 3D graphics `rasterise/` | one recall slide: "the fragment stage is a map over pixels; GPUs became general by running it on anything" |
| Texture formats, hardware filtering, mipmaps | 3D graphics `gpuformats/` | pointer (texture caches and free bilinear interpolation, one line) |
| Ray marching, early termination | 3D graphics `volrender/` | example of divergence (rays stop at different steps), pointer |
| Hardware ray tracing (acceleration-structure traversal units) | ray-tracing day (requested, not planned) | one line in unit 4, pointer; that day points back here for divergence and incoherent memory access |
| Prefix sums, integral image, 2D table = scan rows then columns | integral images `integral-image/` | **parallel scan owned here** (proposed; see open questions); `integral-more/` (optional, not built) keeps a one-slide recall and embeds this day's `parscan` |
| Convolution cost: direct, separable, running sum, integral image | `kernels/`, integral `box-filters/` (`boxspeed`) | recall; arithmetic intensity of the same filters is new here |
| FFT and its cost | `fourier/`, `xcorr-fft/` | recall; FFT on the GPU (bandwidth-bound, $\log_2 n$ passes) one slide |
| Union–find, two-pass labelling | labelling `label-connectivity/` (demo `unionfind`) | recall picture; parallel label propagation and pointer jumping new here |
| Priority-queue flooding | watershed `ws-flooding/` | recall picture; why the global order is sequential |
| Max-flow, augmenting paths | graph cuts `maxflow/` | recall picture; push–relabel as the parallel alternative (one slide, pointer) |
| DP table filling | dynamic programming `dp-paths/` | recall picture; anti-diagonal wavefront new here |
| Convolutional layers, training | neural networks `cnn/`, `backprop/` | "a layer is a matrix product" and why it is compute-bound; int8 weights (`lib/q8.ts`) pointer |
| Distance transform | morphology `morph-openclose/` | example: separable exact EDT parallel per row; chamfer raster scan is not |
| Histograms | thresholding `histogram/` | the histogram as a parallel pattern (atomics) only |

## Units (one folder each: lec, quiz, exercise with #explore)

Four core units, about 1.5 h each including the exercise, plus one optional unit:
`gpu-parallel/`, `gpu-memory/`, `gpu-patterns/`, `gpu-fit/`; optional `gpu-irregular/`.
(Folder names checked against the built folders: none exists.) Every lecture leads with a
demo and a small numeric example; formulas follow on slides titled `## The math: … {.math}`;
pages put them in "The math" callouts; exercises have concept questions first and
`*(math)*` questions after. **No product, vendor or API names in student material**: "GPU",
"warp" (one line: "some vendors say wavefront"), "shared memory", "matrix units" (one line:
"often called tensor cores") are generic terms; the device presets in the demos are
"laptop CPU", "desktop GPU", "data-centre GPU" with order-of-magnitude numbers, labelled as
such.

### 1. `gpu-parallel/`: Many simple cores: throughput, warps and divergence

Concept layer:

- **Hook** (demo `gpufilter`, first slide after "About this lecture"): the same 9×9 blur of a
  2048×2048 image on the CPU (one thread of the browser) and on the GPU (WebGL2): the GPU is
  tens of times faster. Then the same at 128×128: the CPU wins. The day explains both
  results.
- **Primer** (three slides): instructions and a clock (~3·10⁹ cycles/s); memory is far: a
  read from main memory costs ~100 ns = hundreds of cycles; caches keep recently used data
  nearby. A CPU core spends most of its chip area on making **one** thread fast (caches,
  out-of-order execution, branch prediction): it minimises **latency**.
- **Throughput instead**: a GPU spends the area on thousands of simple arithmetic lanes and
  accepts that each single operation is slow. The analogy (one slide): a sports car vs a
  bus; the bus wins only when there are many passengers going the same way. Picture of chip
  area: CPU = big caches + few cores; GPU = a sea of small lanes.
- **Where GPUs come from** (one slide, recall `rasterise/` for those who took 3D graphics):
  drawing a frame is millions of independent pixels running the same small program; the
  fragment stage is a **map** over pixels. Image processing is the same shape of problem.
- **SIMT**: lanes are grouped in **warps** of 32 that execute the **same instruction** at
  the same time on different data (single instruction, multiple threads). Threads are
  grouped in **blocks** (one block runs on one multiprocessor and shares its fast memory);
  blocks are spread over the multiprocessors; a 2048×2048 image = 4·10⁶ threads = 131 072
  warps. The programmer writes the program for **one pixel**.
- **Branch divergence**: `if pixel > t: expensive else: cheap`. If the 32 lanes of a warp
  disagree, the warp runs **both** paths, masking the lanes that are not taking each one.
  Cost = sum of both paths, not the average. Divergence is a property of **warps**, not of
  pixels: an object boundary through a warp costs, a uniform region does not. Loops with
  data-dependent trip counts (iterate until converged, march a ray until opaque) wait for
  the slowest lane. Remedies: make branches uniform per warp (tiles aligned to warps, sort
  or compact work), replace a branch by arithmetic (`select`), split into two passes.
- **Latency hiding**: a warp that waits for memory is parked and another resident warp runs.
  The more warps resident (**occupancy**), the better the waiting is hidden. Occupancy is
  limited by registers and shared memory per thread: a big per-thread program means fewer
  warps in flight. "A GPU needs tens of thousands of threads to be busy": small images
  cannot fill it.
- **Concept summary**: GPUs are fast when there are many independent, identical, simple
  tasks; slow when tasks depend on each other, disagree on what to do, or are few.

The math layer:

- *The math: latency and throughput*: time for $n$ items $t(n) = t_0 + n/R$ (start-up plus
  rate); the break-even $n^* = (t_0^\text{GPU} - t_0^\text{CPU}) / (1/R_\text{CPU} - 1/R_\text{GPU})$.
- *The math: how much must be in flight (Little's law)*: items in flight = rate × latency.
  With bandwidth $\beta = 1$ TB/s and latency $L = 500$ ns, $\beta L = 500$ kB must be
  requested and not yet arrived; at 4 bytes per thread that is ~125 000 threads. Why
  occupancy matters, in one line of algebra.
- *The math: the probability that a warp diverges*: each lane takes the branch independently
  with probability $p$: $P(\text{mixed}) = 1 - p^{32} - (1-p)^{32}$; expected warp time
  $E = p^{32} c_A + (1-p)^{32} c_B + P(\text{mixed})(c_A + c_B)$. At $p = 0.1$ already 97 %
  of warps diverge; with spatially coherent pixels (a warp = 32 neighbouring pixels) the
  fraction is the fraction of warps cut by an object boundary, roughly perimeter/area.

Demo and exercises:

- **Demo `warpsim`** (new, SVG/canvas, simulated): tab *divergence*: a 64×32 synthetic image
  (seeded nuclei from `lib/synth.ts`, or noise, or a checkerboard) tiled into warps (choose
  32×1 rows or 8×4 tiles); a kernel with a branch "inside object: 100 cycles, outside: 10"
  (sliders for both costs and a threshold). Each warp is coloured by its state (all A, all
  B, mixed); a timeline below steps one warp through its instructions with the active-lane
  mask as 32 small squares. Readouts: total cycles, cycles if no divergence, efficiency
  (useful lane-cycles / issued lane-cycles). Toggle *loop until converged* (per-pixel trip
  count from a seeded map; the warp waits for its slowest lane). Tab *latency*: a
  multiprocessor with N resident warps (slider 1–32), each alternating "compute k cycles,
  load (latency L)"; a Gantt chart of issue slots shows idle cycles shrink as warps are
  added; readout: utilisation vs the Little's-law prediction. Presets: "uniform image: no
  divergence", "noise: every warp diverges", "nuclei with row warps vs 8×4 tiles", "one slow
  lane holds 31 hostage", "one warp cannot hide memory", "8 warps are enough", "too many
  registers: occupancy 25 %".
- **Demo `gpufilter`** (new, WebGL2, measured; also used in unit 2): see unit 2 for the full
  spec; here embedded with `data-tab="race"` (one button "run both", two times, the ratio).
- By hand: (a) a CPU with 8 cores at 3 GHz doing 16 float operations per cycle per core vs a
  GPU with 10 000 lanes at 1.5 GHz doing 2 per cycle (fused multiply–add): peak FLOP/s of
  each; (b) a warp over a row of 32 pixels of which 5 are inside an object, costs 100/10:
  cycles with and without divergence, efficiency; the same row tiled as 8×4 where the object
  covers one whole warp; (c) time = start-up + n/rate for given numbers: from which $n$ is
  the GPU faster? (d) a loop whose trip count per lane is given for 8 lanes: warp time vs
  average. *(math)* (e) $P(\text{mixed})$ for $p = 0.01, 0.1, 0.5$ and the expected cost;
  (f) Little's law: how many warps of 32 threads, each loading 8 bytes, keep a 2 TB/s memory
  busy at 600 ns latency?

### 2. `gpu-memory/`: Memory is the bottleneck: bandwidth, coalescing, transfers and the roofline

Concept layer:

- **The hierarchy** (one picture with sizes, latencies and bandwidths, order of magnitude):
  registers (per lane, ~1 cycle), shared memory / L1 (per multiprocessor, ~100 kB, tens of
  cycles), L2 cache (MBs), global memory (GBs, ~500 cycles, ~1 TB/s), the link to the CPU
  (~25 GB/s, 40 times slower than global memory), CPU main memory.
- **Image code is mostly memory-bound**: adding two float32 images does 1 addition per 12
  bytes moved. At 1 TB/s that is 83·10⁹ additions per second, a few per cent of what the
  lanes could do. Most of the GPU waits for memory most of the time; arithmetic is nearly
  free.
- **Coalescing**: memory is read in **lines** (128 bytes = 32 floats, in 32-byte sectors).
  When the 32 lanes of a warp read 32 consecutive floats, one line serves all: **coalesced**.
  When they read a **column** of a row-major image, every lane hits a different line: 32
  transactions, 1/8 of each used. Recall `highbit/`: an image is a row of bytes. Consequences:
  map threads to pixels along rows; **planar** channels (RRR…GGG…) coalesce, interleaved RGB
  makes each warp read three times the bytes it needs per channel; transposes go through
  shared memory.
- **Tiling with a halo** (the stencil's trick): to compute a 16×16 output tile with a 3×3
  kernel, a block loads an 18×18 input tile into shared memory once and every thread reads
  its 9 neighbours from there. Global reads per output pixel drop from 9 to $(18/16)^2 \approx
  1.27$. Larger kernels need larger halos; separable filters need two passes but far fewer
  operations (`kernels/`). (One line: the texture cache of `gpuformats/` does part of this
  automatically and gives bilinear interpolation for free.)
- **Arithmetic intensity** $I$ = operations per byte moved to and from global memory. Worked
  numbers: element-wise add 0.08; 3×3 convolution with ideal reuse 2.25; k×k convolution
  $\approx k^2/4$; matrix product of N×N matrices $N/6$; complex float32 FFT
  ($5n\log_2 n$ operations, 16 bytes per sample read and written) $\approx 0.3\log_2 n$ if the
  whole transform stays in fast memory, only ≈ 0.3 per pass if every pass goes through
  global memory: memory-bound either way (stated, not derived).
- **The roofline** (picture first): log–log plot of attainable operations per second vs $I$:
  a slanted line (bandwidth × $I$) and a flat roof (peak arithmetic). The **ridge point**
  $I^* = $ peak / bandwidth (~30 FLOP/byte for the desktop preset). Left of the ridge:
  memory-bound, speed up by moving fewer bytes (fuse operations, lower precision, reuse
  tiles); right: compute-bound, speed up by doing less arithmetic. A k×k convolution crosses
  the ridge around k ≈ 11; above that FFT-based convolution or separable filters win.
- **Fusing**: a pipeline "subtract background, divide by flat field, clip, threshold" as four
  passes moves the image through memory four times; fused into one kernel it moves it once:
  ~4× faster with identical arithmetic.
- **Transfers and launch overhead**: the image must cross the link to the GPU and back
  (~25 GB/s), and each kernel launch costs ~5–20 µs. End-to-end time = upload + kernels +
  download. For a single cheap operation the transfer is 10–100× the computation: **keep
  data on the GPU** for a whole pipeline, and only move results. Small images (≲ 256×256)
  are dominated by the fixed costs. Recall the hook: why the CPU won at 128×128.

The math layer:

- *The math: the roofline*: attainable $P(I) = \min(\pi, \beta I)$ with peak $\pi$ and
  bandwidth $\beta$; ridge $I^* = \pi/\beta$; time of a kernel
  $t = \max(W/\pi,\ Q/\beta)$ for work $W$ and traffic $Q$.
- *The math: intensity of a k×k stencil*: $2k^2$ operations per pixel; traffic per output
  pixel without reuse $4(k^2 + 1)$ bytes, with a $T\times T$ tile $4((T+k-1)^2/T^2 + 1)$; the
  limit $T\to\infty$ gives $I = 2k^2/8 = k^2/4$. Separable: $4k$ operations, two passes of 8
  bytes. Matrix product: $2N^3$ operations on $3\cdot 4N^2$ bytes, $I = N/6$.
- *The math: end to end*: $t = t_\text{launch}\,m + 2B/\beta_\text{link} + \sum_j \max(W_j/\pi, Q_j/\beta)$
  for $m$ kernels on an image of $B$ bytes; the break-even image size against a CPU time
  $c\cdot n$.

Demos and exercises:

- **Demo `coalesce`** (new, SVG, simulated): a warp of 32 lanes (top) above a strip of
  memory drawn as 128-byte lines split into 32-byte sectors. Access pattern selector:
  *row* (consecutive), *column of a W-wide row-major image* (W slider), *stride s*,
  *interleaved RGB, read R only*, *planar R*, *random* (seeded), *transpose via shared-memory
  tile*. Arrows from lanes to addresses; touched sectors light up; readouts: transactions,
  bytes moved, bytes used, efficiency. Tab *tile*: a 3×3/5×5/k×k stencil on a 32×32 grid;
  toggle *direct global reads* vs *shared-memory tile T×T with halo*; heat map of how often
  each input pixel is fetched from global memory, readout reads per output pixel vs the
  formula. Presets: "rows: 100 %", "columns: 12.5 %", "interleaved RGB: a third", "tile 16
  with a 3×3 halo", "a 31×31 kernel: the halo dominates small tiles".
- **Demo `roofline`** (new, SVG via `lib/plot.ts` with log axes): device preset (laptop
  CPU, desktop GPU, data-centre GPU; or sliders for peak and bandwidth, and a second roof
  for low-precision matrix units), and operations as points: element-wise add, threshold,
  4-step fused pipeline vs unfused, 3×3 / k×k convolution (k slider), separable Gaussian,
  FFT, matrix product (N slider), a convolutional layer (channels slider), each in float32
  or float16 (halves the bytes: the point moves right). Readout per point: memory- or
  compute-bound, attainable rate, time for a chosen image size. Tab *end to end*: image
  size slider (64² to 8192²), number of chained operations, *data stays on the GPU*
  toggle; stacked bars upload / launch / compute / download for GPU and a bar for the CPU;
  the crossover size marked. Presets: "adding images: far left", "k = 11 hits the ridge",
  "matrix products live under the roof", "float16 moves the point right", "one filter on a
  small image: transfer is everything", "a 10-step pipeline that stays on the GPU".
- **Demo `gpufilter`** (new, WebGL2, measured): the same direct k×k box or Gaussian filter
  (k = 3–31) on the CPU (TypeScript, `lib/imageops.ts` `correlate2d` with the same kernel,
  one thread) and on the GPU (a fragment shader over a float texture, rendered to a float
  texture; one pass, same border rule). Image: `cell256.png` tiled/upsampled to 256²–4096²
  (or synthetic). Tab *race*: two buttons and times (median of 5 runs after a warm-up run,
  the GPU result read back so the timing is honest). Tab *breakdown*: GPU time split into
  upload, compute, download (fence syncs between stages); a plot of time vs image size for
  CPU and GPU (both log axes) built as the student clicks sizes, with the crossover; a check
  that both outputs agree (max abs difference shown, float32 vs float16 texture as a
  toggle: the difference becomes visible). The CPU path is skipped above a work threshold
  (e.g. 2048² at k > 15) as `boxspeed` does, with the extrapolated time shown dashed. Labels
  "measured on this computer; your numbers will differ" and "CPU = one thread of the
  browser, not an optimised multi-core program: a tuned CPU code is 10–50× faster". Without
  WebGL2 or float render targets: a message and a recorded result table (static, from the
  author's machine, labelled as such).
- By hand: (a) bytes moved and arithmetic intensity of: add two float32 images; multiply by a
  constant in place; a 3×3 filter with and without reuse; (b) for the desktop preset
  (30·10¹² FLOP/s, 1 TB/s) which of them are memory-bound, and their time on a 4096² image;
  (c) a warp reads a column of a 1024-wide float32 image: lines touched and efficiency; the
  same image transposed; (d) reads per output pixel with a 16×16 tile and a 5×5 kernel;
  (e) a 4096² float32 image, link 25 GB/s: upload + download time vs the time of one
  element-wise kernel on the GPU; how many chained kernels before the computation equals the
  transfer? *(math)* (f) derive $I = k^2/4$ and the k at the ridge; (g) derive the
  break-even image size for given CPU rate, link bandwidth and launch overhead.

### 3. `gpu-patterns/`: Data-parallel patterns: map, stencil, reduction, scan and histogram

Concept layer:

- **Thinking in patterns** instead of loops: if every output can be computed without the
  others, it is parallel. Five patterns cover most of image processing.
- **Map** (one output per input, no neighbours): gain/offset, gamma, thresholding, colour
  conversion, element-wise arithmetic. Perfectly parallel; memory-bound (unit 2).
- **Stencil** (one output from a fixed neighbourhood): convolution, morphology with a small
  structuring element, finite differences, one step of a PDE (diffusion, level sets). Parallel
  per output pixel; tiles and halos (unit 2). An **iterated** stencil (many diffusion steps)
  needs one pass per step (a global barrier between steps).
- **Reduction** (many inputs → one number: sum, mean, max, count): a **tree**: 16 numbers in
  4 steps, $n$ numbers in $\log_2 n$ steps, still $n - 1$ additions. Blocks reduce their part
  in shared memory, then a second pass combines the block results. Picture: the ladder of
  pairwise sums. Recall `floatingpoint/` "summing many pixels": the tree **is** pairwise
  summation, so the GPU's mean of an image is more accurate than a naive loop. But the order
  of additions depends on how the work was split, and floating-point addition is not
  associative: two runs or two GPUs can give **different last digits**. Reproducibility needs
  a fixed order (or integer/compensated sums).
- **Scan** (prefix sum: every running total, owned here, recalled from
  `integral-image/`): looks sequential (each total needs the previous one) but runs in
  $\log_2 n$ steps. **Hillis–Steele**: at step $d$ every element adds the one $2^d$ to its
  left: simple, $n\log_2 n$ additions. **Blelloch** (work-efficient): an up-sweep (a
  reduction tree) and a down-sweep, $2(n-1)$ additions in $2\log_2 n$ steps. The integral
  image = scan along rows, then along columns. Uses beyond sums: **stream compaction** (keep
  only the foreground pixels: scan the 0/1 mask to get each kept pixel's output position),
  radix sort, building lists of objects. Compaction is also the cure for divergence (unit 1):
  gather the pixels that need the expensive branch into a dense list first.
- **Histogram** (many inputs → a few bins at data-dependent positions): each thread adds 1 to
  bin $h(I_p)$. Two threads incrementing the same bin at once would lose a count, so the
  addition must be **atomic** (read–modify–write as one indivisible step). Atomics to the same
  address are **serialised**: a dark image whose pixels all fall into bin 0 is the worst
  case (**contention**). Fix: **privatisation**: each block keeps its own histogram in shared
  memory, then the block histograms are summed (a reduction). Same idea for any scatter
  (accumulating votes in the Hough day's accumulator, splatting).
- **Gather vs scatter**: a thread reading many inputs (gather: convolution) needs no atomics;
  a thread writing to many outputs (scatter: histograms, voting, splatting, back-projection
  in tomography) does. Rewrite scatter as gather when possible (back-projection per output
  pixel instead of per projection ray; recall the radon day in one line).
- **FFT** in one slide: $\log_2 n$ passes of butterflies, each pass a map over pairs:
  regular, parallel, bandwidth-bound; the 2D FFT is 1D FFTs along rows then columns (with a
  transpose: coalescing again).

The math layer:

- *The math: work and depth*: work $W$ = total operations, depth $D$ = longest chain of
  dependent operations. Reduction $W = n - 1$, $D = \lceil\log_2 n\rceil$; Hillis–Steele scan
  $W = n\log_2 n - (n - 1)$, $D = \log_2 n$; Blelloch $W = 2(n-1)$, $D = 2\log_2 n$.
  **Brent's bound**: with $p$ lanes, $t_p \le W/p + D$. A sequential loop has $D = n$ and no
  lane count helps.
- *The math: why a tree needs associativity*: a reduction may regroup
  $((a+b)+c)+d = (a+b)+(c+d)$ only for an associative operation (sum, max, min, logical
  and/or, matrix product); floating-point $+$ is associative only up to rounding; error
  bound of pairwise summation $\approx \log_2 n\cdot u\sum|x_i|$ vs $n\cdot u\sum|x_i|$ naive
  (recalled from `floatingpoint/`, unit roundoff $u$).
- *The math: scan as a matrix*: the prefix sum is multiplication by the lower-triangular
  matrix of ones $L$; Hillis–Steele factorises $L = \prod_d (I + S_{2^d})$ restricted to
  valid indices (shift matrices); the integral image is $L_H\, I\, L_W^\top$ (rows and columns),
  a separable linear map.
- *The math: contention*: $n$ pixels into $b$ bins with probabilities $q_j$; the expected
  number of updates to the busiest bin $\approx n\max_j q_j$ sets the serial time of the
  atomic histogram; with privatisation over $m$ blocks it falls to $\approx n\max_j q_j/m$
  plus a reduction of $m\cdot b$ counts.

Demo and exercises:

- **Demo `parscan`** (new, SVG, step-through; the integral day's `integral-more/` can
  embed it with `data-mode="scan"`): 16 editable values (click ±, or presets) drawn as a row
  of boxes; `data-mode="reduce"`: the tree, one level per *step*; `"scan-hs"`: Hillis–Steele
  ladder, arrows for the additions of the current step; `"scan-blelloch"`: up-sweep then
  down-sweep; `"compact"`: a 0/1 mask, its exclusive scan, the kept elements moving to their
  output slots; `"histogram"`: 64 pixel values from a selectable image (uniform, dark,
  bimodal), 8 bins, warps of 8 threads issuing atomic increments: a timeline shows
  serialised updates per bin per cycle, toggle *privatised (per block)*. Counters
  everywhere: steps (depth), additions (work), vs the sequential loop. Toggle *float16
  values* in `reduce`: the tree sum and the sequential sum differ, and a *shuffle the split*
  button changes the tree's last digit (reproducibility). Presets: "16 numbers in 4 steps",
  "Hillis–Steele does more work", "Blelloch: same depth ×2, work n", "keep the foreground",
  "dark image: one bin, all wait", "private histograms fix it", "the sum changes with the
  order".
- By hand: (a) tree reduction of 8 given numbers, steps and additions; (b) Hillis–Steele and
  Blelloch scans of 8 given numbers, table per step; count additions; (c) stream compaction
  of a given 0/1 mask: exclusive scan and output positions; (d) a 4×4 image's integral image
  as two scans (rows, then columns); (e) an 8-bin histogram of 32 values with atomics: the
  longest queue on one bin; with 4 private histograms; (f) classify ten operations as map,
  stencil, reduction, scan, histogram/scatter (gain, median 3×3, image mean, integral image,
  Otsu's threshold from a histogram, Hough voting, diffusion step, maximum intensity
  projection, labelling the foreground pixels' output positions, back-projection). *(math)*
  (g) work and depth of both scans and Brent's bound for $n = 2^{20}$, $p = 10^4$; (h) show
  that max is associative and that $(10^8 + 1) - 10^8$ differs from $10^8 - 10^8 + 1$ in
  float32.

### 4. `gpu-fit/`: What GPUs are good and bad at, and precision

Concept layer:

- **Four questions** decide whether a method suits a GPU: (1) are there **many independent
  outputs** (≥ 10⁵–10⁶)? (2) do neighbouring threads **do the same thing** (no divergence)?
  (3) is memory access **regular** (coalesced, reused)? (4) is there **no global order**
  (no output waits for an unknown other output)? Plus (5) does the work justify the
  **transfer** (unit 2)?
- **Good** (each one slide with a picture and the pattern from unit 3):
  - element-wise operations and short fused pipelines (map; bandwidth-bound but trivially
    parallel);
  - filters and morphology with small kernels (stencil; tiles); separable filters; large
    kernels via the FFT;
  - FFT (regular butterflies; registration, deconvolution, large convolutions);
  - **neural networks**: a convolutional layer is a large **matrix product** (recall `cnn/`:
    a bank of kernels over all pixels and channels), with intensity $\sim N/6$: the one
    common image workload that is **compute-bound**. Why deep learning and GPUs grew up
    together (2012 onwards), and why training uses batches (more columns, more reuse);
  - **ray marching / volume rendering** (recall `volrender/`): one ray per pixel, independent;
    early ray termination makes rays in a warp stop at different steps (divergence: unit 1),
    which is why bricks/empty-space skipping are used; the texture cache gives trilinear
    interpolation for free. Hardware ray tracing (acceleration-structure traversal units):
    one line, pointer to the ray-tracing day;
  - per-pixel statistics, local thresholds, distance transform done **separably** (exact EDT
    row by row, then column by column: each row independent).
- **Bad** (each one slide: the recalled picture, the property that breaks, the usual parallel
  workaround and its price):
  - **watershed by priority-queue flooding** (`ws-flooding/`): the queue imposes a global
    order (always the lowest pixel next); workarounds: process all pixels of one grey level
    together (level-synchronous, parallel within a level, many passes), or arrow/steepest-
    descent formulations; results can differ on plateaus;
  - **union–find labelling** (`label-connectivity/`): pointer chasing (random memory access)
    and a sequential raster scan; workaround: **label propagation** (every pixel takes the
    smallest label among its neighbours; repeat until nothing changes): parallel per pixel,
    but the number of passes equals the longest path inside an object (a spiral or a long
    snake needs thousands); **pointer jumping** (each pixel replaces its parent by its
    grandparent) shortens chains in $\log$ passes;
  - **max-flow** (`maxflow/`): augmenting paths are found one after another; push–relabel is
    local and parallel per node, but irregular and needs many global iterations (one slide,
    pointer to the graph-cut day);
  - **dynamic programming with dependencies** (`dp-paths/`): each cell needs its
    predecessors; only cells on the same anti-diagonal (or the same row, for row-wise DP) are
    independent: a **wavefront** with $n + m - 1$ steps and little work per step;
  - **chamfer distance transforms and other raster-scan recursions**, recursive (IIR)
    filters, error-diffusion dithering (`quantisation/`: each pixel's error goes to its
    neighbours, a sequential dependency);
  - **small problems**: a 256×256 image has 65 536 pixels, too few to fill the GPU and too
    cheap to pay for transfer and launch; many small images are better batched into one
    launch;
  - **sorting-heavy or tree-heavy** steps (k-d trees, region adjacency graphs) in one line.
- **Amdahl's law** (picture: a bar of time, with the parallel part shrinking and the serial
  part not): if 10 % of a pipeline stays sequential (reading files, a union–find, a
  watershed), the whole pipeline can never be more than 10× faster, however fast the GPU.
  Measure where the time goes before moving anything.
- **Precision on GPUs** (recall the `floatingpoint/` format table in one slide):
  - **float32** is the default; **float64** runs on many consumer GPUs at only 1/32–1/64 of
    the float32 rate (data-centre GPUs: 1/2): avoid it unless the cancellation or summation
    problems of day 3 demand it, and fix those with better algorithms first (shift by a
    mean, pairwise sums);
  - **float16 / bfloat16** halve the bytes (the roofline point moves right, unit 2) and
    **matrix units** ("tensor cores") multiply small blocks of low-precision matrices at
    8–16× the float32 rate; **mixed precision**: multiply in 16 bit, **accumulate in
    float32** (recall: float16 sums get stuck at 64 for small values, `floatingpoint/`);
    bfloat16 keeps float32's range (no overflow at 65 504, why training prefers it), float16
    keeps more digits;
  - **8-bit integers** for inference with a scale per channel (recall: the course's own
    network weights are stored like this, one line);
  - for images: a 12- or 16-bit image in float16 loses its low bits and 65 535 overflows
    (recall `floatingpoint/`): store raw data in uint16 or float32 textures; float16 is fine
    for normalised intermediate images in [0, 1] and for network activations.
- **Closing slide**: a decision table (the four questions × the methods of the day) that
  students fill in during the exercise; and "a GPU makes a parallel algorithm fast; it does
  not make a sequential algorithm parallel: that is an algorithm change, with its own
  trade-offs".

The math layer:

- *The math: Amdahl's law*: serial fraction $s$, speed-up of the rest $k$:
  $S = 1/(s + (1-s)/k) \le 1/s$; with transfer added as a serial term; Gustafson's view
  (bigger images, the parallel part grows) in one line.
- *The math: passes of label propagation*: one pass moves the minimum label one pixel
  further; the number of passes is the largest **geodesic distance** from a component's
  minimum-label pixel to any of its pixels (≤ the component's geodesic diameter); for a
  spiral in an $n\times n$ image $O(n^2)$; pointer jumping halves the depth of every chain
  per pass: $\lceil\log_2 \ell\rceil$ passes for a chain of length $\ell$.
- *The math: the DP wavefront*: cell $(i, j)$ depends on $(i-1, j)$, $(i, j-1)$, $(i-1, j-1)$;
  all cells with $i + j = d$ are independent; depth $n + m - 1$, average parallelism
  $nm/(n + m - 1) \approx \min(n, m)/2$ for a square table.
- *The math: mixed-precision dot products*: a length-$K$ dot product with products rounded to
  float16 and sums in float16 has error growing like $K u_{16}$; accumulating in float32
  reduces the summation part to $K u_{32}$; the stuck-sum threshold $2^{11}\cdot$ (value
  spacing) from `floatingpoint/`.

Demos and exercises:

- **Demo `parlabel`** (new, grid step-through via `lib/gridui.ts` `gridStepper`): a binary
  image on a 24×24 grid (editable by clicking; presets: separate blobs, one long spiral, a
  comb, nuclei from `lib/synth.ts`). Algorithms: *sequential two-pass with union–find*
  (reference from `lib/label.ts`; counts steps as one pixel per step), *label propagation*
  (all pixels at once per pass; changed pixels highlighted), *propagation + pointer jumping*.
  Readouts: passes (depth), pixel updates (work), "time" on a model GPU (passes × launch +
  work / lanes) vs the sequential count, for the shown grid and extrapolated to 1024².
  `data-mode="wavefront"`: a DP table (`lib/dp.ts` costs on a small image) filled
  anti-diagonal by anti-diagonal with the number of busy lanes per step shown as bars.
  `data-mode="flood"`: priority-queue flooding (`lib/watershed.ts` trace) vs level-
  synchronous flooding, with the number of pixels processable in parallel per step.
  Presets: "blobs: 6 passes", "spiral: 300 passes", "pointer jumping: 9 passes", "the DP
  wavefront starts and ends with one lane", "flooding: one pixel at a time".
- **Demo `roofline`** again with `data-preset="cnn"` (a convolutional layer in float32,
  float16, and on matrix units: three points under three roofs) and `gpufilter` with the
  float16 texture toggle (measured difference to float32).
- By hand: (a) the decision table: for 12 methods of the course (Gaussian blur, median 3×3,
  FFT registration, Otsu, union–find labelling, watershed, exact EDT, chamfer DT, graph cut,
  a U-Net, MIP rendering, error-diffusion dithering) answer the four questions and give a
  verdict with the pattern or the obstacle; (b) Amdahl: a pipeline of load 2 s (serial),
  filter 6 s, label 1 s (serial), measure 1 s: speed-up if the filter becomes 50× faster;
  if everything but loading becomes 50× faster; (c) label propagation by hand on a given
  6×6 mask: passes until stable; the same mask with pointer jumping; (d) the anti-diagonals
  of a 5×7 DP table: steps and maximum parallel cells; (e) does 4097 survive float16? the
  sum of 10 000 values of 0.01 in float16 vs float16 products with float32 accumulation
  (recall numbers); (f) float64 at 1/64 rate: how long does a float64 matrix product take
  that runs in 10 ms in float32? *(math)* (g) derive Amdahl's bound and its limit;
  (h) prove the pass count of label propagation for a straight bar of length $\ell$ with the
  minimum label at one end and at the middle; (i) parallelism of the DP wavefront.

### 5. (optional / further reading) `gpu-irregular/`: Making irregular algorithms parallel

- **Jump flooding** (Rong & Tan 2006): an approximate distance transform / Voronoi diagram in
  $\log_2 n$ passes with step sizes $n/2, n/4, \ldots, 1$; where it errs; vs the exact
  separable EDT (morphology owns the distance transform; this is the parallel algorithm).
- **Parallel connected components** in full: block-wise labelling (label tiles in shared
  memory, then merge across tile borders with a small union–find), and why most GPU labelling
  codes are hybrids.
- **Parallel watershed** variants (level-synchronous, cellular-automaton / path-based) and
  why they can disagree with the sequential result on plateaus (link to `ws-flooding/`'s
  plateau slide).
- **Push–relabel max-flow** on a grid (Goldberg–Tarjan, grid versions used for graph cuts) as
  a stencil-like local rule with global relabels.
- **Out-of-core and multi-GPU**: tiling a stack larger than GPU memory, halos between tiles,
  overlap of transfer and compute (streams in one line, no API names).
- **Demo**: `parlabel` with `data-mode="jumpflood"` (seeds on a 32×32 grid, step sizes
  shown, error map vs exact EDT from `lib/distance.ts`) and `data-mode="blocks"` (tile-wise
  labels, then the border merge).

## Shared engineering

- **Reused libraries** (from `demos/src/lib/README.md`): `ui.ts` (sliders, buttons,
  presets, `autoMount`, `rng`), `plot.ts` (`Plot` with `xlog`/`ylog` for the roofline and
  time-vs-size plots, `bars`, `step`, `legend`; `seriesColor`), `gridui.ts` (`gridStepper`,
  `GridAlgorithm`, `GridView`, `N4`/`N8` for `parlabel`), `imageops.ts` (`correlate2d`,
  `gaussianKernel`, `gaussianBlur`, `resize`, `toRGBA` for the CPU path of `gpufilter`),
  `cell.ts` (`loadCell`, `CELL_CREDIT`, shown), `synth.ts` (`nuclei`, `spots`,
  `checkerboard` for `warpsim`, `parlabel`, `parscan`), `label.ts` (`label2d`,
  `UnionFind`: the sequential reference in `parlabel`), `distance.ts` (`edt` for the
  jump-flooding error map), `watershed.ts` (`watershedTrace` for the flood mode), `dp.ts`
  (owner dynprog day; cost table for the wavefront mode), `histogram.ts` (reference
  counts for `parscan`), `gl3d.ts` (`webgl2Available` only; `gpufilter` does not use the
  scene renderer), `colour.ts` (`viridis`, `diverging` for heat maps and difference maps),
  `random.ts`. The float16 rounding for `parscan`'s precision toggle comes from
  `floatbits/fp.ts` (`encode`/`decode` with `F16`); if that module has moved into a shared
  lib by build time, import from there (do not copy).
- **New lib files owned by this day** (names checked against `demos/src/lib/`: none exists):
  - `lib/gpusim.ts` (no DOM): the simulated machine. Warp execution with an active-lane mask
    over a per-thread instruction trace (branch costs, data-dependent loop counts) →
    cycles, efficiency; a latency-hiding scheduler (N resident warps, compute/load phases,
    latency L, round-robin issue) → issue-slot Gantt and utilisation; memory transactions
    for a list of 32 addresses (line and sector size as parameters) → transactions, bytes
    moved/used; tile/halo read counts for a k×k stencil; roofline `attainable(I, peak,
    bw)`, `ridge`, `kernelTime(W, Q)`; end-to-end time model (launch, link, kernels);
    Amdahl; device presets (generic labels, order-of-magnitude numbers, one table so all
    demos agree). Used by `warpsim`, `coalesce`, `roofline`, `parlabel`.
  - `lib/parallel.ts` (no DOM): instrumented data-parallel patterns returning per-step
    frames and work/depth counters: tree reduction (with an optional rounding function to
    emulate float16), Hillis–Steele and Blelloch scans (inclusive/exclusive), stream
    compaction, atomic histogram with per-bin serialisation and privatised variant, label
    propagation and pointer jumping on a grid (as `GridAlgorithm`s), DP wavefront schedule,
    level-synchronous flooding, jump flooding. Used by `parscan` and `parlabel`; the
    integral day's `integral-more/` may import the scans.
  - `lib/glcompute.ts` (WebGL2, DOM): minimal "compute through a fragment shader" helper:
    create an offscreen WebGL2 context only while the demo is visible (same pattern as
    `gl3d.ts`: IntersectionObserver, release off-screen), check
    `EXT_color_buffer_float`/`R16F`/`R32F` support, upload a `Float32Array` as a texture,
    run a full-screen-triangle program into a float framebuffer (ping-pong for chained
    passes), read back, and **timing helpers** (warm-up run, `fenceSync` + `clientWaitSync`
    between upload/compute/download, `EXT_disjoint_timer_query_webgl2` when present,
    median of runs). Used by `gpufilter`; available to later demos (e.g. a ray-tracing
    day's progressive tracer). It does not replace `gl3d.ts` (scene rendering).
- **New demos** (names checked against `demos/src/`: none exists): `warpsim`, `coalesce`,
  `roofline`, `gpufilter`, `parscan`, `parlabel`. Settings through `data-*`
  (`data-tab`, `data-mode`, `data-preset`, `data-device`, `data-k`, `data-size`).
- **Honest simulation**: WebGL2 has no compute shaders, shared memory or atomics, so warps,
  coalescing, scans, histograms and labelling are **simulated** (and say so in the demo
  caption: "a model of a GPU, with the rules of a real one"); only `gpufilter` measures the
  real GPU, on the one pattern a fragment shader expresses directly (a stencil/map). Device
  numbers are order-of-magnitude and labelled; no product names.
- **CSS**: `demo.css` loads late (build-issues): prefix every per-demo selector with
  `.qb-demo[data-demo="<name>"]`; text in slide embeds ≥ 18 px (the 32-lane masks and
  memory sectors are drawn large enough that their labels stay ≥ 18 px; lane numbers shown
  only on the exercise page via `data-layout="full"`). Fit 1150×560 on a slide.
- **Speed**: all simulations are tiny (≤ 64×32 pixels, ≤ 1024 addresses, a 24×24 grid) and
  stay in TypeScript. `gpufilter`'s CPU path is deliberately the plain TypeScript filter
  (the subject is the comparison, not a fast CPU); it runs on a button press, not on a
  slider, so the 30 ms live rule does not apply, and work above the cap is extrapolated, not
  run. No Rust/Wasm.
- **Tests** (`demos/test/gpus-test.ts`, run by `make test`): divergence efficiency on
  uniform / fully mixed warps equals the closed form; transactions for consecutive,
  column and stride access match hand counts (1, 32, ...); tile read counts equal
  $(T+k-1)^2/T^2$; roofline time equals $\max(W/\pi, Q/\beta)$; reduction and both scans equal
  the sequential prefix sums on random integers, with work/depth counters equal to the
  formulas; compaction output positions match a sequential filter; atomic histogram counts
  equal `lib/histogram.ts`; label propagation and pointer jumping give the same partition as
  `label2d` (up to relabelling) and the pass count of a bar of length $\ell$ equals the
  formula; DP wavefront visits every cell once with all dependencies satisfied; Amdahl.
  `glcompute.ts` is not testable in node; `gpufilter` checks GPU vs CPU output agreement at
  runtime and shows it.
- Follow `demos/README.md`: no network, fixed seeds, presets + reset, keyboard back to
  reveal.js (pointer only), static figures in decks and exercises as the print fallback
  (a static roofline, a warp mask diagram, a scan ladder).

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 `gpu-parallel/`: `gpufilter` race as hook, lecture, divergence by hand, `warpsim` |
| 10:45–12:15 | Unit 2 `gpu-memory/`: lecture, intensity and transactions by hand, `coalesce`, `roofline`, `gpufilter` breakdown |
| 13:15–14:45 | Unit 3 `gpu-patterns/`: lecture, reduction/scan/compaction by hand, `parscan` |
| 15:00–16:30 | Unit 4 `gpu-fit/`: lecture, the decision table and Amdahl by hand, `parlabel`; quizzes as wrap-up |

## Open questions

- **Ownership of the parallel scan**: the integral day's optional `integral-more/` (not
  built) planned a "parallel prefix sums on GPUs" slide and a `satstep` `data-mode="scan"`
  (not implemented in the built `satstep`). Proposal: this day owns parallel scan;
  `integral-more/` keeps one recall slide and embeds `parscan` (`data-mode="scan-hs"`), and
  the scan mode is not added to `satstep`. Needs a line in the course map's owners table and
  in `plan/integral-day.md` (not edited here).
- **Course-map row**: mark "Computing: GPUs" as planned (this file) and add the owners
  above (throughput/SIMT/roofline/patterns/GPU precision) to the owners table. Not edited
  here.
- **Entry point vs examples**: entry after day 3, but unit 4's examples come from days 14,
  10, 13, 15 and the DP day. Keep them as self-contained one-picture recalls (this
  plan), or recommend the day after Part II? Proposal: entry stays after day 3; unit 4's
  examples are written as forward pointers.
- **WebGPU**: compute shaders, shared memory and atomics would let `parscan`, the histogram
  and `parlabel` run for real on the GPU. Not in the demo rules (WebGL2 for 3D); browser
  support is still uneven. Proposal: simulate now; revisit as an optional "real" tab later
  with a WebGL2/simulation fallback.
- **Timing in the browser**: precise GPU timer queries are often disabled (privacy), so
  `gpufilter` uses fences and wall-clock medians; numbers vary by machine and with power
  saving. Is a measured demo worth the support burden, or should the recorded table be the
  default with "measure on this computer" as a button? Proposal: measured, with the
  recorded table shown next to it.
- **A multi-threaded CPU path** (Web Workers from inline blob URLs) would make the CPU
  comparison fairer (cores × threads). Optional; check it works from `file://`.
- **Terminology**: "warp" and "shared memory" are one vendor's words but the most common in
  textbooks; "tensor cores" likewise. Proposal: use "warp", "shared memory" and
  "matrix units", with one line each naming the alternatives, and no API or product names
  (teacher references below may name them).
- **Ray-tracing day** (requested, not planned): it should recall divergence and
  incoherent access from units 1–2 and use `lib/glcompute.ts` for its progressive tracer;
  hardware ray-tracing units are one line here.
- Is unit 4 too dense (good/bad survey + Amdahl + precision)? Fallback: move the precision
  slides into unit 2 after the roofline ("fewer bytes move the point right; matrix units
  raise the roof"), which is where float16 first appears.
- Code: a short optional "the same filter as a GPU program" listing would make the SIMT
  model concrete, but design principle 0 keeps code out of the core material; pseudo-math
  only ("for every pixel p in parallel: …").

## References

- J. L. Hennessy & D. A. Patterson (2017), *Computer Architecture: A Quantitative Approach*,
  6th ed., Morgan Kaufmann (ch. 4: data-level parallelism, GPUs; memory hierarchy).
- S. Williams, A. Waterman & D. Patterson (2009), *Roofline: an insightful visual performance
  model for multicore architectures*, CACM 52(4).
- D. B. Kirk & W. W. Hwu (2022), *Programming Massively Parallel Processors*, 4th ed., Morgan
  Kaufmann (SIMT, coalescing, tiling, reduction, scan, histogram privatisation, stencils).
- M. McCool, A. D. Robison & J. Reinders (2012), *Structured Parallel Programming: Patterns
  for Efficient Computation*, Morgan Kaufmann (map, stencil, reduction, scan, work/depth).
- W. D. Hillis & G. L. Steele (1986), *Data parallel algorithms*, CACM 29(12).
- G. E. Blelloch (1990), *Prefix sums and their applications*, CMU-CS-90-190.
- R. P. Brent (1974), *The parallel evaluation of general arithmetic expressions*, JACM 21(2).
- G. M. Amdahl (1967), *Validity of the single processor approach to achieving large scale
  computing capabilities*, AFIPS; J. L. Gustafson (1988), *Reevaluating Amdahl's law*, CACM.
- J. D. C. Little (1961), *A proof for the queuing formula L = λW*, Operations Research 9(3).
- E. Lindholm, J. Nickolls, S. Oberman & J. Montrym (2008), *A unified graphics and computing
  architecture*, IEEE Micro 28(2) (SIMT, teacher background).
- J. D. Owens et al. (2008), *GPU computing*, Proc. IEEE 96(5).
- N. J. Higham (1993), *The accuracy of floating point summation*, SIAM J. Sci. Comput. 14(4)
  (pairwise summation; recalled from the bit-depth day).
- P. Micikevicius et al. (2018), *Mixed precision training*, ICLR (float16 with float32
  accumulation).
- A. Krizhevsky, I. Sutskever & G. E. Hinton (2012), *ImageNet classification with deep
  convolutional neural networks*, NeurIPS (the GPU turning point; one slide).
- G. Rong & T.-S. Tan (2006), *Jump flooding in GPU with applications to Voronoi diagram and
  distance transform*, I3D.
- O. Kalentev, A. Rai, S. Kemnitz & R. Schneider (2011), *Connected component labeling on a
  2D grid using CUDA*, J. Parallel Distrib. Comput. 71(4) (label propagation; teacher
  background).
- V. Vineet & P. J. Narayanan (2008), *CUDA cuts: fast graph cuts on the GPU*, CVPR workshops
  (push–relabel on grids; teacher background).
- Teacher background only (not named in student material): vendor programming guides for
  the GPU compute APIs (warp size, memory transactions, occupancy calculators) and the
  WebGL2 specification (float render targets, `EXT_color_buffer_float`,
  `EXT_disjoint_timer_query_webgl2`).

## Build notes and deviations (2026-11-13)

Built: the four core units `gpu-parallel/`, `gpu-memory/`, `gpu-patterns/`, `gpu-fit/` (lec, quiz,
exercise); demos `warpsim`, `coalesce`, `roofline`, `gpufilter`, `parscan`, `parlabel`; libs
`lib/gpusim.ts`, `lib/parallel.ts`, `lib/glcompute.ts`; test `demos/test/gpus-test.ts`; figures
from `tools/figs/gpus/figs.py`. The optional unit `gpu-irregular/` (jump flooding, block labelling)
was not built.

- Layout recall: the "row of bytes / stride" slides live in `gpuformats/`, so `gpu-memory/` recalls
  that unit (not `highbit/`).
- Latency-hiding scheduler issues greedily (same warp until it waits), as real schedulers do; round
  robin gave bursty, Little's-law-violating utilisation.
- `gpufilter`: sizes 128²–2048² (not 4096²: read-back memory); CPU path above 4·10⁸ multiply–adds is
  extrapolated from a 256² crop. The hook "the CPU wins at 128²" did not hold on the author's machine
  (CPU 1.6 ms vs GPU 1.9 ms, order flips between runs): the slide says "no gain". Recorded table
  (shown without WebGL2) measured with a desktop GPU in headless Chrome (Vulkan backend): 2048², 9×9:
  CPU 1230 ms, GPU 145 ms (upload 25, compute 14, download 106). No renderer/product name is shown.
- Pointer jumping (one propagation + one jump per pass) needs 41 passes on the spiral, not ~log:
  regions that settle on a local minimum still have to be crossed pass by pass. Units say so honestly.
- `parscan` float16 example uses 2000 + 15 small values (exact 2025.5, sequential 2025, tree
  2025–2027 depending on the split); with 2048 + ones the shuffle did not change the sum.
- `parlabel` flood mode uses its own level-synchronous / queue flooding in `lib/parallel.ts`
  (not `lib/watershed.ts` traces); the wavefront mode shows the schedule only (no `lib/dp.ts` costs).
- In the end-to-end model a single 3×3 filter never wins on the GPU against the laptop-CPU preset
  (the CPU filters faster than the link moves the image); the units teach this explicitly.
- `float16` rounding imported from `floatbits/fp.ts` (`roundF16`), not copied.
