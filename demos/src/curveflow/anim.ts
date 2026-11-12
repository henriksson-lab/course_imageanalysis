// Play/pause loop for the level-set demos (curveflow, gac, chanvese). Runs `frame()` once per
// animation frame while playing, and pauses by itself when the demo is not visible: page tab
// hidden, scrolled off screen, or on a reveal.js slide that is not the current one.
// (Local helper; a candidate for lib/ui.ts if other days need the same.)

export class Animator {
  playing = false;
  private raf = 0;
  private onScreen = true;
  constructor(private host: HTMLElement, private frame: () => boolean | void, private onState?: (playing: boolean) => void) {
    if (typeof IntersectionObserver !== "undefined") {
      new IntersectionObserver(es => { this.onScreen = es.some(e => e.isIntersecting); this.kick(); }).observe(host);
    }
    document.addEventListener("visibilitychange", () => this.kick());
    const R = (window as unknown as { Reveal?: { on?: (ev: string, f: () => void) => void } }).Reveal;
    R?.on?.("slidechanged", () => this.kick());
  }

  /** Is the demo visible right now? */
  visible(): boolean {
    if (document.hidden || !this.onScreen) return false;
    const sec = this.host.closest("section");
    if (sec && document.querySelector(".reveal") && !sec.classList.contains("present")) return false;
    return true;
  }

  play(): void { this.playing = true; this.onState?.(true); this.kick(); }
  pause(): void { this.playing = false; this.onState?.(false); }
  toggle(): void { if (this.playing) this.pause(); else this.play(); }

  private kick(): void {
    if (this.raf || !this.playing || !this.visible()) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      if (!this.playing || !this.visible()) return;
      const more = this.frame();
      if (more === false) { this.pause(); return; }
      this.kick();
    });
  }
}
