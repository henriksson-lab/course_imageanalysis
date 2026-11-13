"""Small helpers for the pixel-grid figures of the morphology day (SVG written by hand, in the
style of the existing fig-erode-dilate.svg: dark object cells, red = removed, green = added,
orange origin dot). Also a tiny set of binary morphology operations on numpy arrays with an
explicit border convention, used to compute what the figures show.

Used by tools/figs/morph-*/figs.py (they add this folder to sys.path).
"""
import numpy as np

OBJ = "#29334d"      # object pixel
BG = "#ffffff"       # background pixel
RED = "#ed6b66"      # removed
GREEN = "#80d17a"    # added
ORANGE = "#ff7f0e"   # origin, markers, highlights
GRID = "#bbb"
FONT = 'font-family="system-ui, sans-serif"'


def grid(rows):
    """'0110' strings -> uint8 array."""
    return np.array([[1 if c == "1" else 0 for c in r] for r in rows], dtype=np.uint8)


def se_offsets(se, origin):
    """Offsets (dy, dx) of the 1-cells of a small SE array relative to its origin (row, col)."""
    oy, ox = origin
    return [(y - oy, x - ox) for y, x in zip(*np.nonzero(se))]


def _shift_get(a, dy, dx, outside):
    """b[y, x] = a[y + dy, x + dx], with `outside` beyond the array."""
    h, w = a.shape
    b = np.full_like(a, outside)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    if ys0 < ys1 and xs0 < xs1:
        b[ys0:ys1, xs0:xs1] = a[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
    return b


def erode(a, offs, outside=0):
    out = np.ones_like(a)
    for dy, dx in offs:
        out &= _shift_get(a, dy, dx, outside)
    return out


def dilate(a, offs, outside=0):
    out = np.zeros_like(a)
    for dy, dx in offs:
        out |= _shift_get(a, -dy, -dx, outside)
    return out


CROSS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
SQUARE = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


class Svg:
    def __init__(self, w, h, label):
        self.w, self.h = w, h
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" {FONT} role="img" aria-label="{label}">',
                      f'<rect width="{w}" height="{h}" fill="#fff"/>']

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=20, anchor="middle", weight="normal", fill="#222", italic=False):
        st = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}"{st}>{s}</text>')

    def grid(self, x0, y0, a, cell=34, colours=None, numbers=None, num_size=None, frame=True, origin=None):
        """Draw array a (0/1) at (x0, y0). colours: optional array of colour strings overriding
        the default (OBJ for 1, BG for 0). numbers: optional array of strings drawn in the cells."""
        h, w = a.shape
        for y in range(h):
            for x in range(w):
                c = colours[y][x] if colours is not None and colours[y][x] else (OBJ if a[y, x] else BG)
                self.add(f'<rect x="{x0 + x * cell}" y="{y0 + y * cell}" width="{cell}" height="{cell}" fill="{c}" stroke="{GRID}" stroke-width="1"/>')
                if numbers is not None and numbers[y][x] != "":
                    dark = c in (OBJ, "#4d6fa8")
                    ns = num_size or int(cell * 0.55)
                    self.text(x0 + (x + 0.5) * cell, y0 + (y + 0.5) * cell + ns * 0.36, numbers[y][x], size=ns, fill="#fff" if dark else "#222")
        if frame:
            self.add(f'<rect x="{x0}" y="{y0}" width="{w * cell}" height="{h * cell}" fill="none" stroke="#444" stroke-width="1.5"/>')
        if origin is not None:
            oy, ox = origin
            self.add(f'<circle cx="{x0 + (ox + 0.5) * cell}" cy="{y0 + (oy + 0.5) * cell}" r="{max(4, cell * 0.18):.1f}" fill="{ORANGE}"/>')

    def outline(self, x0, y0, cells, cell, colour=ORANGE, width=3.5):
        """Outline the boundary of a set of (row, col) cells."""
        s = set(cells)
        for (y, x) in s:
            X, Y = x0 + x * cell, y0 + y * cell
            if (y - 1, x) not in s:
                self.add(f'<line x1="{X}" y1="{Y}" x2="{X + cell}" y2="{Y}" stroke="{colour}" stroke-width="{width}" stroke-linecap="round"/>')
            if (y + 1, x) not in s:
                self.add(f'<line x1="{X}" y1="{Y + cell}" x2="{X + cell}" y2="{Y + cell}" stroke="{colour}" stroke-width="{width}" stroke-linecap="round"/>')
            if (y, x - 1) not in s:
                self.add(f'<line x1="{X}" y1="{Y}" x2="{X}" y2="{Y + cell}" stroke="{colour}" stroke-width="{width}" stroke-linecap="round"/>')
            if (y, x + 1) not in s:
                self.add(f'<line x1="{X + cell}" y1="{Y}" x2="{X + cell}" y2="{Y + cell}" stroke="{colour}" stroke-width="{width}" stroke-linecap="round"/>')

    def arrow(self, x1, y1, x2, y2, colour="#444", width=2.5):
        import math
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" stroke-width="{width}"/>')
        a = math.atan2(y2 - y1, x2 - x1)
        p1 = (x2 - 12 * math.cos(a - 0.4), y2 - 12 * math.sin(a - 0.4))
        p2 = (x2 - 12 * math.cos(a + 0.4), y2 - 12 * math.sin(a + 0.4))
        self.add(f'<polygon points="{x2},{y2} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{colour}"/>')

    def save(self, path):
        self.parts.append("</svg>")
        with open(path, "w") as f:
            f.write("\n".join(self.parts) + "\n")


def diff_colours(before, after):
    """Colour array: OBJ where in both, RED where removed, GREEN where added."""
    h, w = before.shape
    c = [["" for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if before[y, x] and after[y, x]:
                c[y][x] = OBJ
            elif before[y, x]:
                c[y][x] = RED
            elif after[y, x]:
                c[y][x] = GREEN
    return c
