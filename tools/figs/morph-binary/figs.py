"""Figures for morph-binary/lec.qmd (run from the repo root: python3 -I tools/figs/morph-binary/figs.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gridfig import (Svg, grid, erode, dilate, diff_colours, se_offsets, CROSS, SQUARE,
                     OBJ, BG, RED, GREEN, ORANGE)

OUT = "morph-binary"

# the 7 x 7 image and L-shaped SE of fig-erode-dilate.svg
A7 = grid(["0000000", "0111100", "0111100", "0110000", "0111110", "0000110", "0000000"])
L_SE = grid(["000", "011", "010"])
L = se_offsets(L_SE, (1, 1))                    # (0,0), (0,1), (1,0)
L_REF = [(-dy, -dx) for dy, dx in L]


def fig_defects():
    """A thresholded mask with the four typical defects, annotated."""
    rows = [
        "00000000000000000000000",
        "01111110000000000000010",
        "01111110000111100111100",
        "01110110000111111111100",
        "01111110010111100111100",
        "01111110000111100111100",
        "00000000000000000000000",
        "00000100000000000000000",
        "00111111111011111111000",
        "00111111111011111111000",
        "00000000000000000000100",
    ]
    a = grid(rows)
    cell = 30
    h, w = a.shape
    x0, y0 = 10, 10
    W, H = x0 * 2 + w * cell, y0 + h * cell + 70
    s = Svg(W, H, "a binary mask with a hole, specks, a thin bridge and a gap")
    s.grid(x0, y0, a, cell)
    # annotate
    s.outline(x0, y0, [(3, 4)], cell)                      # hole
    s.outline(x0, y0, [(3, 16), (3, 17), (3, 18)], cell)   # bridge
    s.outline(x0, y0, [(8, 11), (9, 11)], cell)            # gap
    for p in [(4, 9), (1, 21), (7, 5), (10, 20)]:          # specks
        s.outline(x0, y0, [p], cell)
    yl = y0 + h * cell + 32
    s.text(x0 + 4.5 * cell, yl, "hole", size=22, fill=ORANGE, weight="600")
    s.text(x0 + 9.5 * cell, yl, "specks", size=22, fill=ORANGE, weight="600")
    s.text(x0 + 13.5 * cell, yl, "gap", size=22, fill=ORANGE, weight="600")
    s.text(x0 + 17.5 * cell, yl, "bridge", size=22, fill=ORANGE, weight="600")
    s.save(f"{OUT}/fig-defects.svg")


def fig_se_shapes():
    """Five structuring elements with their origins."""
    ses = [
        ("cross", grid(["010", "111", "010"]), (1, 1)),
        ("3 × 3 square", grid(["111", "111", "111"]), (1, 1)),
        ("disc, radius 2", grid(["01110", "11111", "11111", "11111", "01110"]), (2, 2)),
        ("line", grid(["00001", "00010", "00100", "01000", "10000"]), (2, 2)),
        ("L-shape", L_SE, (1, 1)),
    ]
    cell = 30
    gap = 60
    widths = [se.shape[1] * cell for _, se, _ in ses]
    W = sum(widths) + gap * (len(ses) - 1) + 40
    H = 5 * cell + 80
    s = Svg(W, H, "five structuring elements: cross, square, disc, line, L-shape, each with its origin")
    x = 20
    for (name, se, org), wd in zip(ses, widths):
        yoff = 50 + (5 - se.shape[0]) * cell / 2
        s.grid(x, yoff, se, cell, origin=org)
        s.text(x + wd / 2, 32, name, size=22, weight="600")
        x += wd + gap
    s.save(f"{OUT}/fig-se-shapes.svg")


def fig_dilate_L():
    """Dilation as 'paste B on every pixel', and the reflected window at one pixel."""
    a = grid(["000000", "000000", "010000", "000000", "000100", "000000"])
    d = dilate(a, L)
    assert d.sum() == 6
    cell = 38
    s = Svg(2 * 6 * cell + 60 + 40, 6 * cell + 70, "a two-pixel image dilated by the L: a copy of L on each pixel; the mirrored L placed at a pixel x hits A")
    y0 = 50
    x1 = 20
    s.text(x1 + 3 * cell, 32, "A ⊕ B: copies of B", size=22, weight="600")
    s.grid(x1, y0, d, cell, colours=diff_colours(a, d))
    x2 = x1 + 6 * cell + 60
    s.text(x2 + 3 * cell, 32, "window B̌ at x hits A", size=22, weight="600")
    s.grid(x2, y0, d, cell, colours=diff_colours(a, d))
    # pixel x = (3, 1): the window x + B̌ = {(3,1), (3,0), (2,1)} hits A at (2,1)
    win = [(3 + dy, 1 + dx) for dy, dx in L_REF]
    s.outline(x2, y0, win, cell)
    s.add(f'<circle cx="{x2 + 1.5 * cell}" cy="{y0 + 3.5 * cell}" r="7" fill="{ORANGE}"/>')
    s.text(x2 + 1.5 * cell, y0 + 4.75 * cell, "x", size=24, weight="600", fill=ORANGE)
    s.save(f"{OUT}/fig-dilate-L.svg")


def fig_minmax_row():
    """Erosion and dilation of one row of a mask by a 3-wide window: min and max."""
    a = np.array([[0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]], dtype=np.uint8)
    row = [(0, -1), (0, 0), (0, 1)]
    e = erode(a, row, outside=1)
    d = dilate(a, row, outside=0)
    cell = 36
    w = a.shape[1]
    lab = 150
    W, H = lab + w * cell + 10, 3 * (cell + 30) + 10
    s = Svg(W, H, "one row of a mask, its erosion (min over 3) and dilation (max over 3)")
    rows = [("A", a, None), ("min over 3", e, diff_colours(a, e)), ("max over 3", d, diff_colours(a, d))]
    for k, (name, v, col) in enumerate(rows):
        y = 20 + k * (cell + 30)
        s.text(lab - 12, y + cell * 0.68, name, size=21, anchor="end", weight="600")
        s.grid(lab, y, v, cell, colours=col, numbers=[[str(int(t)) for t in v[0]]])
    # the window at position 5
    s.add(f'<rect x="{lab + 4 * cell}" y="{18}" width="{3 * cell}" height="{cell + 4}" fill="none" stroke="{ORANGE}" stroke-width="4"/>')
    s.arrow(lab + 5.5 * cell, 20 + cell + 4, lab + 5.5 * cell, 20 + (cell + 30) - 2, colour=ORANGE)
    s.save(f"{OUT}/fig-minmax-row.svg")


def fig_duality():
    """(A ⊖ B)ᶜ = Aᶜ ⊕ B̌ for the 7 x 7 example."""
    e = erode(A7, L, outside=0)
    ac = 1 - A7
    dd = dilate(ac, L_REF, outside=1)
    assert e.sum() == 7 and ((1 - dd) == e).all()
    cell = 26
    pw = 7 * cell
    gx, gy = 70, 80
    W, H = 2 * pw + gx + 40, 2 * pw + gy + 60
    s = Svg(W, H, "duality: the erosion of A equals the complement of the dilation of the background by the reflected L")
    pos = [(20, 50), (20 + pw + gx, 50), (20, 50 + pw + gy), (20 + pw + gx, 50 + pw + gy)]
    panels = [("Aᶜ (background)", ac, None), ("Aᶜ ⊕ B̌", dd, diff_colours(ac, dd)),
              ("A ⊖ B", e, None), ("(Aᶜ ⊕ B̌)ᶜ", 1 - dd, None)]
    for (x, y), (name, v, col) in zip(pos, panels):
        s.text(x + pw / 2, y - 14, name, size=22, weight="600")
        s.grid(x, y, v, cell, colours=col)
    s.arrow(20 + pw + 8, 50 + pw / 2, 20 + pw + gx - 8, 50 + pw / 2)
    s.arrow(20 + pw + gx + pw / 2, 50 + pw + 8, 20 + pw + gx + pw / 2, 50 + pw + gy - 42)
    s.text(20 + pw + gx / 2, 50 + pw + gy + pw / 2 + 14, "=", size=40, weight="600")
    s.save(f"{OUT}/fig-duality.svg")


def fig_row_col():
    """Dilation by a row of 3, then a column of 3 = dilation by the 3 x 3 square."""
    a = grid(["0000000", "0000000", "0010000", "0000000", "0000100", "0001000", "0000000"])
    row = [(0, -1), (0, 0), (0, 1)]
    col = [(-1, 0), (0, 0), (1, 0)]
    r1 = dilate(a, row)
    r2 = dilate(r1, col)
    sq = dilate(a, SQUARE)
    assert (r2 == sq).all()
    cell = 26
    pw = 7 * cell
    gap = 70
    W, H = 3 * pw + 2 * gap + 40, pw + 70
    s = Svg(W, H, "dilating by a row of three, then by a column of three, gives the dilation by the 3 by 3 square")
    xs = [20, 20 + pw + gap, 20 + 2 * (pw + gap)]
    names = ["A", "⊕ row", "⊕ column"]
    vals = [(a, None), (r1, diff_colours(a, r1)), (r2, diff_colours(r1, r2))]
    for x, nm, (v, c) in zip(xs, names, vals):
        s.text(x + pw / 2, 34, nm, size=22, weight="600")
        s.grid(x, 50, v, cell, colours=c)
    for k in range(2):
        xa = xs[k] + pw + 10
        s.arrow(xa, 50 + pw / 2, xa + gap - 20, 50 + pw / 2)
    s.grid(xs[0] + pw + 16, 50 + pw / 2 - 30, np.ones((1, 3), np.uint8), 16, origin=(0, 1))
    s.grid(xs[1] + pw + 32, 50 + pw / 2 - 58, np.ones((3, 1), np.uint8), 16, origin=(1, 0))
    print("row-col counts", int(a.sum()), int(r1.sum()), int(r2.sum()))
    s.save(f"{OUT}/fig-row-col.svg")


def fig_boundaries():
    """Inner boundary A minus (A eroded) with the cross and with the square."""
    a = grid(["0000000000", "0001111000", "0011111100", "0111111110", "0111111110",
              "0111111100", "0011111100", "0001111000", "0000110000", "0000000000"])
    bc = a & (1 - erode(a, CROSS))
    bs = a & (1 - erode(a, SQUARE))
    cell = 26
    pw = 10 * cell
    gap = 60
    W, H = 2 * pw + gap + 40, pw + 64
    s = Svg(W, H, "the inner boundary with the cross (thin, 8-connected) and with the square (thicker, 4-connected)")
    for k, (nm, b) in enumerate([("with the cross", bc), ("with the 3 × 3 square", bs)]):
        x = 20 + k * (pw + gap)
        col = [[ORANGE if b[y, xx] else ("#c9d0df" if a[y, xx] else "") for xx in range(a.shape[1])] for y in range(a.shape[0])]
        s.text(x + pw / 2, 34, nm, size=22, weight="600")
        s.grid(x, 50, a, cell, colours=col)
    s.save(f"{OUT}/fig-boundaries.svg")
    return int(bc.sum()), int(bs.sum())


def fig_border():
    """Erosion by the square of the 'border convention' preset, outside = 0 vs outside = 1."""
    a = grid(["111110000000", "111110000000", "111110000000", "111110000000", "000000000000", "000000000000",
              "000000011110", "000000011110", "000000011110", "000000011110", "000000000000", "000000000000"])
    e0 = erode(a, SQUARE, outside=0)
    e1 = erode(a, SQUARE, outside=1)
    assert a.sum() == 36 and e0.sum() == 10 and e1.sum() == 16, (a.sum(), e0.sum(), e1.sum())
    cell = 20
    pw = 12 * cell
    gap = 40
    W, H = 2 * pw + gap + 40, pw + 70
    s = Svg(W, H, "erosion by the 3 by 3 square: with outside = 0 the corner object loses its edge rows, with outside = 1 it does not")
    for k, (nm, e) in enumerate([("outside = 0: 10 left", e0), ("outside = 1: 16 left", e1)]):
        x = 20 + k * (pw + gap)
        s.text(x + pw / 2, 34, nm, size=22, weight="600")
        s.add(f'<rect x="{x - 6}" y="{44}" width="{pw + 12}" height="{pw + 12}" fill="{"#ffffff" if k == 0 else OBJ}" opacity="{1 if k == 0 else 0.35}"/>')
        s.grid(x, 50, a, cell, colours=diff_colours(a, e))
    s.save(f"{OUT}/fig-border.svg")


if __name__ == "__main__":
    fig_defects()
    fig_se_shapes()
    fig_dilate_L()
    fig_minmax_row()
    fig_duality()
    fig_row_col()
    print("boundaries (cross, square):", fig_boundaries())
    fig_border()
