# Design principles

The principles below govern how we build course material in this repository. Check
every new lecture, exercise and demo against them. See [audience.md](audience.md)
for who we are designing for.

## 1. Interactive first

The course teaches through **interactive demos**. Every unit has at least one demo
that lets the student change the input and see the effect immediately.

- A demo answers one question ("what does this kernel do?", "which frequencies
  carry the edges?"). Prefer several small demos over one that does everything.
- The demo is embedded both in the **lecture** (to show it live) and in the
  **exercise** under `## Explore {#explore}`, with concrete questions to answer
  with it. A demo without questions is a toy, not an exercise.
- Every demo has **presets** that take the student straight to the interesting
  cases, and a way back to the start.
- Demos use **fixed random seeds**, so a recording and a student see the same thing.

## 2. Picture, small example, then the general formula

Build intuition before notation: a picture or demo first, then a small case worked
by hand (a 5 × 5 image, a 3 × 3 kernel), then the general formula. The students
know linear algebra and calculus, so the general formula should come; it is just
not where we start.

## 3. Connect to what they already know

Most of image analysis is linear algebra and calculus in disguise. Name the link
explicitly: an image is a matrix, a geometric transform is a matrix acting on
coordinates, a filter is a linear map, a derivative is a difference of neighbours,
the Fourier transform is a change of basis.

## 4. Real images, honestly labelled

Use real images (microscopy, photographs, medical images) alongside synthetic test
patterns. Synthetic patterns isolate an effect; real images show that it matters.
Always credit the source and licence of an image in the demo or caption.

## 5. Exercises over stretch goals

A good number of core exercises every student is expected to complete, rather
than a few with long optional tails. A short "Where this leads" note is fine.

## 6. Keep slide decks small: split, don't cut

Each deck is a self-contained snippet on one topic (roughly 10–18 content slides).
When a deck grows too big, split it into its own unit folder rather than cutting
good material.

## 7. Demo engineering rules

See [demos/README.md](demos/README.md): no network access, plain DOM + SVG/canvas,
give the keyboard back to reveal.js, labels readable on a projector, settings from
`data-*` attributes so one demo serves several units.
