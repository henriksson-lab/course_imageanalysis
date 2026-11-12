# Design principles

The principles below govern how we build course material in this repository. Check
every new lecture, exercise and demo against them. See [audience.md](audience.md)
for who we are designing for.

## 0. Theory and intuition, not tools

The course teaches the **ideas** behind image analysis methods: what a method does,
why it works, and when it fails. It is not about any specific library, program or
API.

- **Two layers: concept and math.** All the math stays in the course, complete
  and correct. But it is layered so that a weaker student can **skip past the math
  and still get the concept**:
  - The **concept layer** (pictures, demos, plain-language explanation, small
    worked examples with numbers) must stand on its own. Reading only this layer,
    a student understands what the method does, why, and when it fails.
  - The **math layer** (formulas, derivations, proofs) follows the concept it
    formalises, clearly marked so it can be skipped: in slides, a separate slide or
    block after the intuition; on pages, a "The math" callout or section. Never
    make the concept layer depend on having followed a derivation.
  - **Demos train intuition**: lead with them, and do not hammer students with
    formulas before they have seen the effect.
- **Exercises cover both layers**: concept questions (**predict and explain**:
  "what happens to the boundary if λ doubles?", small calculations with numbers)
  that every student should do, and math questions (derive, prove, generalise)
  marked as such for students who want the full treatment.
- **No tool names** in student material beyond, at most, a one-line pointer
  ("image libraries call this X"). Comparisons of specific tools stay in teacher
  notes in `plan/`.
- **Code examples are secondary.** They may be added later, separately from the
  core material; nothing in a unit may depend on them.

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

## 2. Picture, small example, then (briefly) the general formula

Build intuition before notation: a picture or demo first, then a small case worked
by hand (a 5 × 5 image, a 3 × 3 kernel), then the general formula. The students
know linear algebra and calculus, so the formula can be stated compactly when it
helps; it is the end point of the explanation, not the starting point, and it sits
in the skippable math layer (see 0).

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
`data-*` attributes so one demo serves several units. Demos are written in
**TypeScript**; 3D uses **WebGL**; computation that is too slow for live
interaction moves to **Rust compiled to WebAssembly**.
