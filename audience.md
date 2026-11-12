# Audience

University students who have completed about **three years** of study (typically
a bachelor's degree in a science or engineering subject). Not published on the site.

## What we can expect them to know

### Mathematics
- **Linear algebra:** vectors, matrices, matrix–vector and matrix–matrix products,
  linear transformations, inverses, determinants, eigenvalues and eigenvectors,
  bases and change of basis.
- **Calculus:** derivatives, partial derivatives and the gradient, integrals
  (single and multiple), Taylor expansions, basic series.
- **Basic probability and statistics:** mean, variance, the normal distribution.

### What we do not assume
- **No signal processing:** convolution, the Fourier transform, sampling and
  aliasing are new. Introduce them from what they know: a weighted sum, an
  integral, a change of basis.
- **No prior image analysis**, and no machine learning.
- **Programming** is not needed for the core material. Code examples may be added
  later as a secondary layer *(language to be decided)*.

## Consequences for the material

- The full mathematics is part of the course: they can read matrix notation, sums
  and integrals. But students differ in mathematical strength, so the material is
  **layered**: the demo and the picture carry the idea, and the math is a clearly
  marked layer on top that weaker students can skim and still get the concept
  (see design principle 0).
- Recall linear algebra in one line when we lean on it ("the Fourier transform is a
  change of basis"), rather than re-teaching it.
- New notions (convolution, frequency, sampling) get the slow treatment: a small
  example by hand first, then the demo, then the general formula.
