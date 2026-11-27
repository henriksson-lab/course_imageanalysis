# Plan: a day on matching shapes in Fourier space (descriptors, rotation, docking)

Internal working doc (plan/ is not rendered). Status: draft for discussion, 2026-11-27.

**Course map (proposed):** a new day directly after FFT registration (day 23), in Part V
"Aligning images and motion"; it becomes day 24 and the later days move up by one (crates and
folders are named by id, so nothing is renamed). Day id `shapematch`, title "Shape matching in
Fourier space". Hard prerequisites: FFT registration (`fftreg`, which brings `images`).
Soft (see also): labelling (`shape-features/`, `regionprops/`), moments (`moment-invariants/`,
`zernike/`), dynamic programming (`polar-boundary/`, forward), snakes (`shape-models/`, forward).
Not yet in `days.yml`: add it when the build starts.

**Why a day of its own** (discussed 2026-11-27): FFT registration only finds **shifts**. Three
topics that are each too small for a day answer the same next question, *how do we match shapes
when the rotation is unknown too?*, and all three reuse the correlation theorem of
`xcorr-fft/`:

- Fourier descriptors of an outline (today one slide in `shape-features/`);
- rotation and scale as shifts in polar and log-polar coordinates (Fourier–Mellin; until now the
  optional, unbuilt unit 5 of `plan/fftregistration-day.md`, which **moves here**);
- FFT docking (Katchalski-Katzir): one FFT correlation scores every translation of a ligand
  against a receptor, and a loop runs over rotations. The biological payoff of the day.

**Notation** (course conventions, as on the FFT registration day): "taper" for the Hann
weighting; positive angles turn clockwise on screen (`code/imgcourse`); an outline is a complex
signal $z_n = x_n + i\,y_n$, $n = 0 \ldots N-1$; its descriptors are $c_k$, $k$ signed
(`freq()` in `lib/fft.ts`); the concept layer says "outline A" and "outline B", "receptor" and
"ligand".

## Goal of the day

After the day a student can:

1. write a closed outline as a periodic complex signal, compute its **Fourier descriptors** and
   say what translation, scale, rotation, a new **starting point** and mirroring do to them;
   tell a rotation from a change of starting point by hand;
2. **match two outlines** with unknown rotation and starting point: one FFT gives the score for
   every starting point at once, and the phase of the peak gives the rotation (a complex
   least-squares fit); know when magnitudes $|c_k|$ alone are enough and when they are not;
3. explain why a rotation about a known centre is a **circular shift in polar coordinates**, and
   why the magnitude spectrum turns an unknown rotation and scale into shifts on a
   **log-polar** grid (**Fourier–Mellin**); handle its practical issues (180° ambiguity,
   tapering, high-pass weighting, interpolation);
4. explain **FFT docking**: receptor and ligand on a grid with a surface layer and an interior,
   a score that rewards contact and punishes overlap, one correlation per ligand rotation,
   and the cost of the rotation loop in 2D and 3D;
5. *(optional)* say how the same idea extends to 3D (rotation sampling, spherical harmonics) and
   to cryo-EM particle alignment, and what rigid docking cannot do.

Why it fits this audience: it is one theorem (the shift theorem) used three more times, now on
an outline, on an angle and on a grid of molecules. Everything has a 1D version small enough to
do by hand (four-point outlines, eight-cell receptors). Docking connects the course to
structural biology, which many students know from the other side.

## Prerequisite gap

- **`xcorr-fft/` and `phasecorr/` must have been seen** (hard prerequisite): correlation =
  $\overline F G$ in Fourier space, circular vs linear correlation and zero-padding, peaks and
  peak ratios. Recalled in one line each, not re-derived.
- **Complex numbers in polar form**, $e^{i\alpha}$ as a rotation: recalled on the unit 1 "About
  this lecture" slide. Multiplying by $e^{i\alpha}$ rotates a point; the concept layer says this
  with a picture.
- **Boundary tracing** (`regionprops/`, chain code and perimeter): the outline is taken from
  there. Resampling to equal arc length uses linear interpolation along the polygon (new, one
  slide).
- **Bilinear interpolation, inverse mapping** (`images/`): for polar and log-polar resampling
  and for rotating the ligand grid. Recall + pointer.
- **Rotation invariance by magnitudes** (`zernike/`, moments day; `moment-invariants/` for the
  orientation from complex moments): the same fact, $e^{im\theta}$ picks up a phase under rotation.
  Recalled, not retaught.

## Overlaps with other days (who owns what)

| Topic | Owner | Elsewhere |
|---|---|---|
| Fourier descriptors: definition, invariances in one bullet each, smoothing by truncation, demo overlay | `shape-features/` (labelling day): **one slide**, stays as it is | this day unit 1 **expands** it (matching, starting point, elliptic descriptors, pitfalls); `shape-features/` gets a forward pointer |
| Correlation theorem, circular vs linear, padding, phase correlation, tapers, peak ratio, sub-pixel peaks | `fftreg` (`xcorr-fft/`, `phasecorr/`, `stitching/`) | this day: recalls; reuses `lib/registration.ts` |
| **Fourier–Mellin** (rotation and scale by log-polar resampling of $\lvert F\rvert$) | **this day**, unit 2 (moved from fftreg unit 5) | `plan/fftregistration-day.md` unit 5 and the "limits" slide of `stitching/` point here |
| Polar resampling about a centre, $r(\theta)$ of star-shaped objects | `polar-boundary/` (dynamic programming day, `lib/polar.ts`) | this day unit 2 uses `lib/polar.ts`; DP day is later in the course, so the pointer is forward. Unit 1 contrasts: $z(t)$ works for any outline, $r(\theta)$ only for star-shaped ones |
| Rotation invariants from moments, Zernike magnitudes | moments day (`moment-invariants/`, `zernike/`) | unit 1 and 2: one recall line each ("the same $e^{im\theta}$ trick") |
| Landmark alignment (Procrustes), shape PCA | snakes day (`shape-models/`) | unit 1 "matching" is Procrustes with the correspondence unknown up to a cyclic shift: one forward-pointer slide |
| Point-set alignment without correspondences (ICP) | keypoints day (`icp/`) | unit 1: contrast slide (ICP needs a starting guess, FD matching searches all starts) |
| Generalised Hough (R-table, rotation by looping over angles) | Hough day (`ght/`) | unit 3: same rotation loop, different evidence; pointer |
| Template matching with NCC | `fftreg` (`matching/`) | unit 3: docking is template matching with a score that is not NCC; pointer |
| Distance transforms (surface layers) | morphology day | unit 3: the surface layer = dilation minus the object; recall |
| GPUs for batches of FFTs | GPU day | unit 4: the rotation loop is the textbook GPU workload; pointer |
| 3D volumes, voxel spacing | marching-cubes day (`volumes3d/`) | unit 4: pointer only |

## Units (one folder each: lec, quiz, exercise with #explore and ## Code)

Three core units, about 1.5 h each including the exercise, plus one shorter optional unit.
Layering as in README.md "Marking the math layer".

### 1. `fourier-descriptors/`: Outlines as periodic signals

- Recall the `shape-features/` slide in one picture: the star rebuilt from $|k| \le K$.
- **Sampling the outline**: trace the boundary (`regionprops/`), resample to $N$ points equally
  spaced in **arc length** (otherwise the staircase of pixel steps leaks into every $c_k$);
  $N$ a power of two. Picture: pixel corners vs resampled points.
- **What a change does to the coefficients** (each with a picture, concept first):
  translation moves only $c_0$; scaling multiplies all $c_k$ by $s$; a rotation by $\alpha$
  multiplies all $c_k$ by $e^{i\alpha}$; a new starting point $m$ steps later multiplies $c_k$ by
  $e^{2\pi i k m/N}$ (a phase *ramp*: different for every $k$); walking the other way round swaps
  $c_k$ and $c_{-k}$; a mirror image conjugates and swaps.
- **Each harmonic is an ellipse**: $c_k$ and $c_{-k}$ together trace an ellipse $|k|$ times; $c_{\pm1}$
  is the best-fitting ellipse, semi-axes $|c_1| \pm |c_{-1}|$. Picture: epicycles drawing the
  outline (the demo animates them).
- **The math: invariances** {.math}: $c_k = \frac1N\sum_n z_n e^{-2\pi i k n/N}$; the five rules
  above from the shift theorem and linearity; $|c_k|/|c_1|$ is invariant to everything except
  mirroring and loses the relative phases (two different shapes can share all magnitudes: example
  in the demo). Invariants that keep some relative phase (e.g. $c_k c_{-k}/c_1 c_{-1}$, see the math
  questions) as one line of further reading.
- **Matching with unknown rotation and starting point**: compare outline B with every rotation
  and every starting point of outline A. For a fixed starting point the best rotation and scale
  are one complex number found by least squares; the score for **all** starting points at once
  is a circular correlation of the two complex outlines, one FFT (recall `xcorr-fft/`). The peak
  says where to start, its phase says the angle. Picture: two leaf outlines, the score curve
  over $m$, the overlay at the peak.
- **The math: complex Procrustes over cyclic shifts** {.math}: minimise
  $\sum_k |c'_k - \lambda\, e^{2\pi i k m/N} c_k|^2$ over $\lambda = s e^{i\alpha} \in \mathbb C$ and $m$;
  for fixed $m$, $\lambda^* = \frac{\sum_k \overline{c_k} e^{-2\pi i k m/N} c'_k}{\sum_k |c_k|^2}$; the residual is
  smallest where $|\sum_k \overline{c_k}\,c'_k\,e^{-2\pi i k m/N}|$ is largest, an inverse DFT of
  $\overline{c_k}\,c'_k$: the correlation theorem again. Mirror images: repeat with the conjugated,
  reversed outline.
- **Elliptic Fourier descriptors** (Kuhl & Giardina 1982): $x(t)$ and $y(t)$ as two real series,
  four numbers per harmonic, normalised by the first ellipse; the standard in morphometrics
  (leaf, shell, nucleus and cell outlines). One slide: same information, different bookkeeping.
- Pitfalls: one closed outline per object (holes, touching objects: watershed first); high $k$
  is pixel noise; very concave or spiky shapes need many coefficients; descriptors of a
  segmentation inherit its errors.
- **Demo `fdmatch`** (new): two outlines (presets from `lib/shapes.ts` plus traced masks of
  nuclei from `cell256.png`), B a rotated, rescaled, re-started, noisy copy of A or a different
  shape. Panels: A and B with their epicycle reconstruction for a slider $K$; the coefficient
  magnitudes and phases (stem plots); the match score over the starting point $m$ with the peak,
  the recovered angle and scale, and the overlay. Toggles: arc-length resampling on/off, allow
  mirror, magnitudes only. Presets: "rotation vs new starting point", "a square sampled at its
  corners is a circle", "staircase noise without resampling", "same magnitudes, different shape",
  "mirror image", "nuclei: which two are the same shape?". Builds on `fourierOutline()` in
  `demos/src/moments/features.ts` (move it to `lib/shapes.ts` or a new `lib/contour.ts`).
- By hand: (a) $z = (1, i, -1, -i)$: only $c_1 = 1$: a square sampled at its four corners has the
  coefficients of a circle (sampling!). (b) $z = (2, i, -2, -i)$: $c_1 = 3/2$, $c_{-1} = 1/2$, the
  others 0; semi-axes 2 and 1. (c) Rotate (b) by 90° (multiply by $i$) and, separately, start one
  point later: $c_1 \to \tfrac32 i$ in both, but $c_{-1} \to \tfrac12 i$ for the rotation and
  $-\tfrac12 i$ for the new start. How do you tell them apart? (d) Concept: why does matching on
  $|c_k|$ alone fail to tell a shape from some of its rearrangements, and which measurement
  would?
- Math questions: prove the starting-point rule; show that $c_1 c_{-1}$ does not change with the
  starting point and turns by $2\alpha$ under a rotation; derive $\lambda^*$.

### 2. `fourier-mellin/`: Rotation and scale become shifts

- Rotation about a **known** centre: resample the image (or an object's mask) on a polar grid
  $(r, \theta)$; a rotation is a circular shift along $\theta$, found by a 1D correlation per ring,
  summed over rings (one FFT along $\theta$). Picture: a cell and its polar unwrap, rotated and
  shifted. Recall `polar-boundary/` (forward) and Zernike's $e^{im\theta}$.
- Unknown centre too: the magnitude spectrum $|F|$ ignores shifts (`fourier/` B4) but turns with
  the image and shrinks when the image grows. On a **log-polar** grid of $|F|$, rotation and scale
  are shifts: phase correlation finds them; undo them, then phase-correlate once more for the
  translation (Reddy & Chatterji 1996; De Castro & Morandi 1987 for rotation).
- Practicalities: the 180° ambiguity ($|F|$ of a real image is point-symmetric: test $\theta$ and
  $\theta + 180°$ by the final translation peak); high-pass weighting of $|F|$ (low frequencies
  dominate and carry little angle information); tapering is essential (the border cross does
  not rotate, `phasecorr/`); log-polar resampling needs interpolation (`images/`) and has few
  samples at small radius; usable scale range ≈ 0.5–2.
- **The math: Fourier–Mellin** {.math}: $g(\mathbf x) = f(sR\mathbf x + \mathbf t)$ ⇒
  $|G(\mathbf k)| = s^{-2}|F(s^{-1}R\mathbf k)|$ (continuous FT); in log-polar coordinates
  $(\rho, \theta) = (\log|\mathbf k|, \arg\mathbf k)$: $|G|(\rho, \theta) = s^{-2}|F|(\rho - \log s, \theta - \alpha)$:
  a shift. The Mellin transform is the Fourier transform in $\log r$.
- **The math: rotational correlation** {.math}: with $f(r, \theta) = \sum_m f_m(r) e^{im\theta}$,
  $\int f(r, \theta)\, g(r, \theta + \alpha)\,d\theta = 2\pi\sum_m \overline{f_m(r)}\, g_m(r)\, e^{im\alpha}$: every
  angle at once by one inverse FFT in $\alpha$ (the 2D case of the spherical-harmonic trick of unit 4).
- Applications (one slide): aligning rotated tiles or slide scans; **cryo-EM 2D classification**:
  thousands of noisy particle images aligned in rotation and translation to class averages by
  exactly these polar correlations; orientation of cells before averaging.
- **Demo `fmellin`** (new; spec from `plan/fftregistration-day.md` unit 5): image B = A rotated,
  scaled, shifted (sliders) with noise; panels $|F_A|$, $|F_B|$, their log-polar resamplings, the
  phase-correlation surface in $(\log r, \theta)$ with the peak → angle and scale, the translation
  step and the overlay. Toggles: Hann taper, high-pass on $|F|$, both 180° candidates; a "known
  centre" tab with the plain polar unwrap and the per-ring correlation. Presets: "pure rotation
  30°", "scale 1.5 + shift", "180° ambiguity", "no taper: the cross dominates", "scale 3: out of
  range", "known centre: one 1D correlation".
- By hand: a 30° rotation moves the log-polar peak by how many $\theta$ bins (256 bins over 180°)?
  (≈ 42.7.) A scaling by 2 with 64 $\log r$ bins spanning $r = 1 \ldots 128$: $\ln 2 / (\ln 128 / 64) \approx 9.1$
  bins. Concept: why does Fourier–Mellin fail on two images that only half overlap?

### 3. `fft-docking/`: Docking by correlation

- The problem, in pictures: two proteins (or any two rigid pieces) fit together; find the
  relative position and angle where the surfaces touch over a large area without overlapping.
  Six unknowns in 3D, three in 2D (two shifts, one angle). In this unit 2D shapes stand in for
  molecules: honest caption.
- **The grid score** (Katchalski-Katzir et al. 1992), concept first: put the receptor on a grid
  as a thin **surface layer** (value 1) around an **interior** (a large negative value $\rho$); the
  ligand as 1 on its cells. Slide the ligand: each ligand cell on the receptor's surface adds 1
  (contact), each one inside the receptor adds $\rho$ (a clash). The score is a correlation, so
  **one FFT** gives it for every shift; repeat for every ligand angle; keep the best peaks.
  Picture: the two grids, the score map at the right angle and at a wrong one.
- **The math: docking as correlation** {.math}: $S_\theta(\mathbf d) = \sum_{\mathbf x} a(\mathbf x)\, b_\theta(\mathbf x - \mathbf d)$,
  $\hat S_\theta = \hat a\,\overline{\hat b_\theta}$; the receptor's FFT is computed once, the ligand's once per
  angle. Zero-padding to avoid wrap-around (recall `xcorr-fft/`). Cost $\mathcal O(R\, M \log M)$ for
  $R$ angles and $M$ grid cells, against $\mathcal O(R\, M^2)$ by brute force.
- **Two scores for one FFT**: real and imaginary parts carry two grids, e.g. surface + $i\cdot$core
  for both molecules; the real part of the product is surface·surface − core·core: contact
  minus clash in one complex correlation (the idea used by FTDock and ZDOCK). Further terms
  (electrostatics: receptor potential × ligand charges; desolvation) are more correlations
  added to the score. One slide, concept only.
- What it gets wrong: proteins are not rigid (side chains move, loops fold), so a strict clash
  penalty rejects the true pose. Softer scores (thinner core, smaller $|\rho|$, coarser grid)
  tolerate this at the price of more false positives; real pipelines therefore keep thousands of
  poses and **re-rank** them with finer models (clustering, energy refinement). Grid spacing
  and angle step set the resolution: a 10° step moves a ligand atom 20 Å from the centre by about
  3.5 Å.
- **Demo `dock2d`** (new): a receptor with a pocket and a ligand (shape presets, or draw them on
  a grid with `lib/paint.ts`), receptor surface layer and core shown in colour. Drag and turn the
  ligand by hand with a live score readout; button "search": the score map for the current
  angle, the best score per angle as a curve, top-5 poses overlaid. Sliders: $\rho$, surface
  thickness, grid spacing, angle step, a "flexible" bump on the ligand (the true fit then
  clashes). Readouts: FFTs done, time vs brute force. Presets: "key and lock", "the right angle,
  the wrong angle", "$\rho$ too mild: the ligand sinks in", "surface too thin: no contact signal",
  "coarse angle step misses the fit", "a flexible loop: re-ranking needed".
- By hand: receptor $a = (\rho, \rho, \rho, 1, 0, 0, 0, 0)$ with $\rho = -9$ (core 0–2, surface
  layer 3), ligand of two cells placed at $d, d+1$: scores $S(1) = 2\rho = -18$, $S(2) = \rho + 1 = -8$,
  $S(3) = 1$, $S(4) = 0$: the best pose touches the surface. Then via the length-8 DFT (or argue
  with the correlation theorem that it must agree). Concept: what changes if the ligand is
  given surface and core layers as well?
- Math questions: show that the complex score's real part is surface·surface − core·core; count
  flops for a 2D search ($128^2$ grid, 5° steps) both ways.

### 4. (optional / further reading) `docking-3d/`: Rotations in 3D and beyond rigid shapes

- Rotations in 3D: three angles; sampling them evenly is not a 3D grid of Euler angles (it
  crowds near the poles). Number of rotations ≈ $8\pi^2 / (\Delta\theta)^3$: a few thousand at 15°,
  ≈ 50 000–70 000 at 6°. With a $128^3$ grid one FFT is ≈ $2\cdot10^8$ flops, so a fine search is
  ≈ $10^{13}$ flops: hours on one CPU core, minutes on a GPU (pointer to the GPU day).
- **Spherical harmonics**: the 3D analogue of unit 1 and unit 2's angular Fourier series; a
  rotation mixes coefficients of the same degree (Wigner matrices) instead of multiplying them
  by a phase, and correlations over rotations can again be computed by fast transforms (Hex:
  Ritchie & Kemp 2000; SO(3) FFT: Kostelec & Rockmore 2008). Concept and one picture only.
- Related uses of the same machine: template matching in cryo-electron tomograms (a known
  complex searched in rotation and translation), fitting atomic models into cryo-EM maps,
  ligand docking into pockets (where flexible, non-FFT methods dominate).
- Limits and where the field went: flexibility, scoring, and learned methods that predict
  complexes directly from sequence; FFT docking stays as a fast, exhaustive first pass. One
  slide, no tool names beyond the classic references.
- Figures only (rotation sampling on a sphere, a spherical-harmonic picture); no demo.

## Shared engineering

- **`demos/src/lib/registration.ts`** (owner `fftreg`): reuse `correlate2d`, `crossPower`,
  `taper2d`, `findPeaks`, `peakRatio`; add log-polar resampling (on top of `lib/polar.ts`) and a
  1D complex circular correlation (for unit 1).
- **`demos/src/lib/contour.ts`** (new, **owned by this day**, no DOM, runs under node): boundary
  to polygon (from `lib/label.ts` / `lib/shapes.ts`), arc-length resampling, complex DFT of an
  outline (`lib/fft.ts`), invariance helpers, matching over starting point and rotation,
  elliptic descriptors, epicycle reconstruction. `fourierOutline()` from
  `demos/src/moments/features.ts` moves here and `moments` imports it.
- **`demos/src/lib/docking.ts`** (new, owned by this day): grid encoding (surface layer by
  dilation from `lib/morph.ts`, core, complex encoding), rotation of a grid (bilinear, from
  `lib/imageops.ts`, or exact rotation of the shape polygon then rasterisation), scoring over
  angles with the receptor FFT cached, top-k poses with non-maximum suppression across angles.
- Reuse: `lib/polar.ts` (`toPolar`, `fromPolar`), `lib/fft.ts`, `lib/ui.ts`, `lib/plot.ts`,
  `lib/colour.ts`, `lib/paint.ts`, `lib/random.ts`, `lib/imageops.ts` (`cell256.png`).
- Performance: `dock2d` at $128^2$ padded to $256^2$ and 5° steps is 72 FFTs of $256^2$, tens of ms:
  fine on a button, animate the angle loop so the cost is visible. `fdmatch` is 1D, trivial.
- Tests (`make test`, `demos/test/*-test.ts`): the five invariance rules on random outlines
  (error < 1e-10); matching recovers a known rotation, scale and starting point exactly for a
  resampled copy; the by-hand values of units 1 and 3; Fourier–Mellin recovers 30° and scale
  1.5 within 1° / 2 % (test moved from the fftreg plan); docking finds the planted pose of the
  "key and lock" preset; FFT docking score equals brute force on small grids.
- **Rust code** (`plan/code-plan.md`): crate `code/shapematch/`, modules `fourier_descriptors`
  (arc-length resampling, descriptors, invariants, matching over starting points with an FFT),
  `fourier_mellin` (log-polar resampling, phase correlation copied into `common.rs` from the
  fftreg pattern), `fft_docking` (grids, correlation per angle, best poses); `docking_3d` has
  none. Uses `imgcourse::fft` and `Complex` as the fftreg crate does.
- Follow `demos/README.md`: fixed seeds, presets + reset, keyboard back to reveal.js, fits
  1150×560 on a slide, static figure as print fallback.

## Suggested timetable

| Time | |
|---|---|
| 09:00–10:30 | Unit 1 lecture + by-hand descriptors, `fdmatch` |
| 10:45–12:15 | Unit 2 lecture + exercise (`fmellin`) |
| 13:15–14:45 | Unit 3 lecture + exercise (`dock2d`) |
| 15:00–15:45 | Unit 4 (optional) slides; quizzes as wrap-up |

## Changes elsewhere when this is built

- `days.yml`: insert the day after `fftreg`, renumber the following days (`number:`), run
  `make check-prereqs` and `make sidebar`; `plan/coursemap.md` Part V row and day count.
- `plan/fftregistration-day.md`: unit 5 → "moved to the shape-matching day"; `stitching/`
  "limits" slide: rotation/scale pointer to `fourier-mellin/`.
- `shape-features/lec.qmd` "Contour descriptors" slide: forward pointer to
  `fourier-descriptors/`; `regionprops/exercise.qmd` "where this leads" may mention it.
- `moment-invariants/`, `zernike/`: optional one-line forward pointer ("the same phase trick
  aligns outlines and docks molecules").
- `plan/review-gaps.md` gap 13 (contour descriptors): mark as covered.

## Open questions

- Day id and title: `shapematch` / "Shape matching in Fourier space", or a title that names
  docking ("Rotations, outlines and docking")?
- Placement: right after FFT registration (proposed; it needs only `fftreg`), or after the
  moments day? The latter is earlier in the course and would lack phase correlation.
- Is unit 4 worth a folder, or three slides at the end of `fft-docking/`?
- A real 2D test case for docking: a slice through a known protein complex rasterised to a grid
  (licence: PDB entries are public domain), or synthetic shapes only (proposed for the demo, with
  a real slice as a figure)?
- Morphometrics data for unit 1 (leaf or nucleus outlines with a licence), or traced nuclei from
  `cell256.png` only?

## References

- Zahn & Roskies (1972), *Fourier descriptors for plane closed curves*, IEEE Trans. Computers.
- Granlund (1972), *Fourier preprocessing for hand print character recognition*, IEEE Trans. Computers.
- Persoon & Fu (1977), *Shape discrimination using Fourier descriptors*, IEEE Trans. SMC.
- Kuhl & Giardina (1982), *Elliptic Fourier features of a closed contour*, Computer Graphics and Image Processing.
- De Castro & Morandi (1987), *Registration of translated and rotated images using finite Fourier transforms*, PAMI.
- Reddy & Chatterji (1996), *An FFT-based technique for translation, rotation, and scale-invariant image registration*, IEEE Trans. Image Processing.
- Katchalski-Katzir, Shariv, Eisenstein, Friesem, Aflalo & Vakser (1992), *Molecular surface recognition: determination of geometric fit between proteins and their ligands by correlation techniques*, PNAS.
- Gabb, Jackson & Sternberg (1997), *Modelling protein docking using shape complementarity, electrostatics and biochemical information*, J. Mol. Biol.
- Chen, Li & Weng (2003), *ZDOCK: an initial-stage protein-docking algorithm*, Proteins.
- Ritchie & Kemp (2000), *Protein docking using spherical polar Fourier correlations*, Proteins.
- Kostelec & Rockmore (2008), *FFTs on the rotation group*, J. Fourier Analysis and Applications.
