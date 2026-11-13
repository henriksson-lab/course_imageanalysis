"""Static figures for xcorr-fft/ (cross-correlation with the FFT).

Run from the repository root:  python3 -I tools/figs/xcorr-fft/figs.py
Writes xcorr-fft/fig-*.svg|png. Pure numpy; correlation convention as in the unit and in
demos/src/lib/registration.ts: c(t) = sum_x A(x+t) B(x), DFT of c = A_k conj(B_k).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

plt.rcParams.update({"font.size": 16, "axes.titlesize": 16, "svg.fonttype": "none",
                     "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, RED, GREEN, GREY, PURPLE = "#1f77b4", "#ff7f0e", "#d62728", "#2a9d4b", "#888888", "#cc79a7"
CELL = np.asarray(Image.open("demos/src/fourier2d/cell256.png")).astype(float)
OUT = "xcorr-fft/"


def bumps(N, seed):
    r = np.random.default_rng(seed); x = np.arange(N); s = np.zeros(N)
    for _ in range(4):
        c, h, w = r.uniform(0, N), r.uniform(1, 3), r.uniform(1, 2.5)
        for p in (-N, 0, N):
            s += h * np.exp(-(x - c - p) ** 2 / (2 * w * w))
    return s


# ------------------------------------------------------------ flip = conjugate
def flip_conj():
    f = np.array([0, 1, 3, 2, 0, 0, 1, 0], float)
    N = len(f); x = np.arange(N)
    g = f[(-x) % N]                      # flipped (circularly): g(x) = f(-x)
    F, G = np.fft.fft(f), np.fft.fft(g)
    assert np.allclose(G, np.conj(F))
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 4.8), gridspec_kw={"hspace": 0.75, "wspace": 0.45})
    for a, s, c, lab in ((ax[0, 0], f, BLUE, "image f"), (ax[1, 0], g, ORANGE, "f flipped")):
        a.stem(x, s, linefmt=c, markerfmt="o", basefmt=" ")
        a.set_title(lab, color=c); a.set_xticks(x); a.set_yticks([0, 3]); a.set_ylim(0, 3.6)
        a.set_xlabel("x")
    k = np.arange(N)
    ax[0, 1].plot(k, np.abs(F), "o", color=BLUE, ms=11, label="f")
    ax[0, 1].plot(k, np.abs(G), "x", color=ORANGE, ms=11, mew=3, label="flipped")
    ax[0, 1].set_title("size |F|: the same"); ax[0, 1].set_xticks(k); ax[0, 1].set_xlabel("frequency k")
    ph = np.angle(F); ph[np.abs(F) < 1e-9] = 0
    ax[1, 1].axhline(0, color=GREY, lw=1)
    ax[1, 1].plot(k, ph, "o", color=BLUE, ms=11)
    ax[1, 1].plot(k, -ph, "x", color=ORANGE, ms=11, mew=3)
    ax[1, 1].set_title("phase: opposite"); ax[1, 1].set_xticks(k); ax[1, 1].set_xlabel("frequency k")
    ax[1, 1].set_yticks([-np.pi, 0, np.pi]); ax[1, 1].set_yticklabels(["−π", "0", "π"])
    fig.savefig(OUT + "fig-flip-conj.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ waves vote
def waves_vote():
    N, t0 = 32, 5
    A = bumps(N, 3); A -= A.mean()
    B = np.roll(A, -t0)                  # B(x) = A(x + t0): B belongs at t0 on A
    Ak, Bk = np.fft.fft(A), np.fft.fft(B)
    P = Ak * np.conj(Bk)
    t = np.linspace(-N / 2, N / 2, 801)
    ks = [1, 2, 3, 4, 5]
    contrib = lambda k: (2 / N) * np.real(P[k] * np.exp(2j * np.pi * k * t / N))
    total = sum((2 / N) * np.real(P[k] * np.exp(2j * np.pi * k * t / N)) for k in range(1, N // 2)) \
        + np.real(P[N // 2] * np.exp(1j * np.pi * t)) / N
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    off = 0; gap = 1.0
    scale = max(np.abs(contrib(k)).max() for k in ks)
    cols = [BLUE, ORANGE, GREEN, PURPLE, "#0072b2"]
    for i, k in enumerate(ks):
        y = contrib(k) / scale * 0.45
        ax.plot(t, y + off, color=cols[i], lw=2.2)
        ax.text(N / 2 + 0.6, off, "k = %d" % k, va="center", fontsize=15, color=cols[i])
        off -= gap
    off -= 0.35
    y = total / np.abs(total).max() * 0.9
    ax.plot(t, y + off, color="k", lw=2.6)
    ax.text(N / 2 + 0.6, off, "sum of\nall waves", va="center", fontsize=15)
    ax.axvline(t0, color=GREEN, ls="--", lw=2)
    ax.text(t0, 0.75, "true shift t = %d" % t0, ha="center", color=GREEN, fontsize=15)
    ax.set_xlim(-N / 2, N / 2); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("shift t")
    fig.savefig(OUT + "fig-waves-vote.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    tt = np.arange(-N // 2, N // 2)
    c = np.real(np.fft.ifft(P))
    print("waves: argmax of c", tt[np.argmax(c[tt % N])])


# ------------------------------------------------------------ circular vs linear
def circular():
    N, t = 8, 3
    A = np.array([60, 90, 200, 230, 150, 120, 80, 170], float)
    B = np.array([110, 180, 220, 140, 70, 90, 200, 160], float)
    cm = plt.get_cmap("gray")
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    h = 0.9

    def cell(x, y, v, ec="k", ls="-", alpha=1.0):
        if v is None:
            ax.add_patch(Rectangle((x, y), 1, h, fc="white", ec=GREY, lw=1, ls=":"))
            ax.text(x + 0.5, y + h / 2, "0", ha="center", va="center", fontsize=14, color=GREY)
        else:
            ax.add_patch(Rectangle((x, y), 1, h, fc=cm(v / 255), ec=ec, lw=1.2, ls=ls, alpha=alpha))

    def label(y, s, c):
        ax.text(-0.3, y + h / 2, s, ha="right", va="center", fontsize=18, color=c, weight="bold")
    # top: circular
    yA, yB = 6.6, 5.4
    ax.text(-1.2, 8.0, "circular (length N = 8), shift t = 3", fontsize=16, va="center")
    label(yA, "A", BLUE); label(yB, "B", ORANGE)
    for i in range(N):
        cell(i, yA, A[i])
    ax.add_patch(Rectangle((0, yA), N, h, fill=False, ec=BLUE, lw=3))
    for i in range(N - t):
        cell(t + i, yB, B[i])
    for i in range(t):
        cell(i, yB, B[N - t + i])                     # wrapped in on the left
        cell(N + i, yB, B[N - t + i], ec=RED, ls="--", alpha=0.25)   # where it was pushed out
    ax.add_patch(Rectangle((t, yB), N - t, h, fill=False, ec=ORANGE, lw=3))
    ax.add_patch(Rectangle((0, yB), t, h, fill=False, ec=RED, lw=3))
    ax.annotate("", xy=(t / 2, yB - 0.1), xytext=(N + t / 2, yB - 0.1),
                arrowprops=dict(arrowstyle="->", color=RED, lw=2.4, connectionstyle="arc3,rad=-0.25"))
    ax.text(N / 2 + 1, yB - 1.8, "pushed out on the right, back in on the left:\npairs that never lie on top of each other",
            ha="center", va="center", fontsize=15, color=RED)
    # bottom: padded
    yA, yB = 0.3, -0.9
    ax.text(-1.2, 1.7, "zero-padded to 2N, same shift", fontsize=16, va="center")
    label(yA, "A", BLUE); label(yB, "B", ORANGE)
    for i in range(2 * N):
        cell(i * 0.5 * 2 if False else i, yA, A[i] if i < N else None)
    ax.add_patch(Rectangle((0, yA), N, h, fill=False, ec=BLUE, lw=3))
    for i in range(2 * N):
        j = i - t
        cell(i, yB, B[j] if 0 <= j < N else None)
    ax.add_patch(Rectangle((t, yB), N, h, fill=False, ec=ORANGE, lw=3))
    ax.text(N, yB - 0.75, "the end of B meets zeros: the linear correlation", ha="center", va="center", fontsize=15, color=GREEN)
    ax.set_xlim(-1.3, 2 * N + 0.2); ax.set_ylim(-2.1, 8.5); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(OUT + "fig-circular.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ cost
def cost():
    Ns = 2.0 ** np.arange(5, 13)
    M = 4 * Ns ** 2
    fft = 3 * 5 * M * np.log2(M) + 6 * M
    brute = 2 * Ns ** 4
    win = lambda w: 2 * ((2 * w + 1) * Ns - w * (w + 1)) ** 2
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.loglog(Ns, brute, "o-", color=RED, lw=2.4, label="brute force, every shift")
    ax.loglog(Ns, win(32), "s--", color=ORANGE, lw=2.2, label="brute force, ±32 px")
    ax.loglog(Ns, fft, "^-", color=BLUE, lw=2.4, label="FFT route (padded)")
    ax.set_xticks(Ns); ax.set_xticklabels([str(int(n)) for n in Ns], fontsize=14)
    ax.minorticks_off()
    ax.set_xlabel("tile side N (pixels)"); ax.set_ylabel("operations")
    ax.legend(frameon=False, fontsize=14, loc="upper left")
    i = list(Ns).index(1024)
    ax.annotate("1.4·10⁹", (1024, fft[i]), xytext=(1400, 1.2e8), color=BLUE, fontsize=15, ha="left",
                arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.annotate("2.2·10¹²", (1024, brute[i]), xytext=(1400, 4e11), color=RED, fontsize=15, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_ylim(1e5, 1e15)
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT + "fig-fft-cost.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("cost N=1024: fft %.3g brute %.3g win32 %.3g win12 %.3g" % (fft[i], brute[i], win(32)[i], win(12)[i]))


# ------------------------------------------------------------ local sums
def local_sums():
    A = CELL[64:192, 64:192]
    n = 32
    # sum of the pixels of A under a 32 x 32 block placed at every shift (linear, block inside A)
    from numpy.lib.stride_tricks import sliding_window_view as sw
    S = sw(A, (n, n)).sum(axis=(2, 3))
    full = np.full(A.shape, np.nan); full[:S.shape[0], :S.shape[1]] = S / (n * n)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.35})
    ax[0].imshow(A, cmap="gray", vmin=0, vmax=255); ax[0].set_title("image A")
    box = np.zeros_like(A); box[:n, :n] = 1
    ax[1].imshow(box, cmap="gray_r", vmin=0, vmax=1.6)
    ax[1].add_patch(Rectangle((-0.5, -0.5), 128, 128, fill=False, ec=GREY, lw=1))
    ax[1].set_title("ones, the size of B")
    ax[1].text(n / 2, n + 14, "32 × 32", ha="center", fontsize=14)
    cm = plt.get_cmap("gray").copy(); cm.set_bad("#e8e8e8")
    ax[2].imshow(np.ma.masked_invalid(full), cmap=cm, vmin=0, vmax=255); ax[2].set_title("local mean of A")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.text(0.355, 0.5, "★", fontsize=30, ha="center", va="center")
    fig.text(0.655, 0.5, "=", fontsize=30, ha="center", va="center")
    fig.savefig(OUT + "fig-local-sums.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    flip_conj(); waves_vote(); circular(); cost(); local_sums()
