// Queues for flooding, shortest paths and propagation (no DOM).
//  - PriorityQueue: binary min-heap of integer items (pixel indices) with numeric
//    priorities; ties come out in insertion order (FIFO), which watershed flooding needs.
//  - Heap<T>: generic binary heap with a comparator, for anything else.
//  - BucketQueue: integer priorities 0..levels-1, FIFO inside each level, O(1) push/pop.
//  - Fifo: growable ring buffer of integers (breadth-first propagation, reconstruction).

/** Min-heap of int items keyed by a number; equal keys pop in insertion (FIFO) order. */
export class PriorityQueue {
  private items: Int32Array;
  private keys: Float64Array;
  private order: Float64Array;
  private n = 0;
  private counter = 0;

  constructor(capacity = 64) {
    this.items = new Int32Array(capacity);
    this.keys = new Float64Array(capacity);
    this.order = new Float64Array(capacity);
  }

  get size(): number { return this.n; }
  isEmpty(): boolean { return this.n === 0; }
  clear(): void { this.n = 0; this.counter = 0; }

  push(item: number, priority: number): void {
    if (this.n === this.items.length) this.grow();
    let i = this.n++;
    const ord = this.counter++;
    // sift up
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.keys[p] < priority || (this.keys[p] === priority && this.order[p] < ord)) break;
      this.items[i] = this.items[p]; this.keys[i] = this.keys[p]; this.order[i] = this.order[p];
      i = p;
    }
    this.items[i] = item; this.keys[i] = priority; this.order[i] = ord;
  }

  /** Item with the smallest priority (undefined behaviour when empty: check size first). */
  peek(): number { return this.items[0]; }
  peekPriority(): number { return this.keys[0]; }

  /** Remove and return the item with the smallest priority (-1 when empty). */
  pop(): number {
    if (this.n === 0) return -1;
    const top = this.items[0];
    const n = --this.n;
    if (n > 0) {
      const it = this.items[n], k = this.keys[n], o = this.order[n];
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && (this.keys[c + 1] < this.keys[c] || (this.keys[c + 1] === this.keys[c] && this.order[c + 1] < this.order[c]))) c++;
        if (k < this.keys[c] || (k === this.keys[c] && o < this.order[c])) break;
        this.items[i] = this.items[c]; this.keys[i] = this.keys[c]; this.order[i] = this.order[c];
        i = c;
      }
      this.items[i] = it; this.keys[i] = k; this.order[i] = o;
    }
    return top;
  }

  private grow(): void {
    const c = this.items.length * 2;
    const it = new Int32Array(c); it.set(this.items); this.items = it;
    const k = new Float64Array(c); k.set(this.keys); this.keys = k;
    const o = new Float64Array(c); o.set(this.order); this.order = o;
  }
}

/** Generic binary heap: `less(a, b)` true when a must come out before b. */
export class Heap<T> {
  private a: T[] = [];
  constructor(private less: (a: T, b: T) => boolean) {}
  get size(): number { return this.a.length; }
  isEmpty(): boolean { return this.a.length === 0; }
  peek(): T | undefined { return this.a[0]; }
  push(x: T): void {
    const a = this.a;
    let i = a.length;
    a.push(x);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(x, a[p])) break;
      a[i] = a[p]; i = p;
    }
    a[i] = x;
  }
  pop(): T | undefined {
    const a = this.a;
    if (a.length === 0) return undefined;
    const top = a[0], last = a.pop()!;
    if (a.length > 0) {
      let i = 0;
      const n = a.length;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && this.less(a[c + 1], a[c])) c++;
        if (!this.less(a[c], last)) break;
        a[i] = a[c]; i = c;
      }
      a[i] = last;
    }
    return top;
  }
}

/** Growable FIFO ring buffer of integers. */
export class Fifo {
  private buf: Int32Array;
  private head = 0;
  private n = 0;
  constructor(capacity = 64) { this.buf = new Int32Array(Math.max(1, capacity)); }
  get size(): number { return this.n; }
  isEmpty(): boolean { return this.n === 0; }
  clear(): void { this.head = 0; this.n = 0; }
  push(x: number): void {
    if (this.n === this.buf.length) {
      const b = new Int32Array(this.buf.length * 2);
      for (let i = 0; i < this.n; i++) b[i] = this.buf[(this.head + i) % this.buf.length];
      this.buf = b; this.head = 0;
    }
    this.buf[(this.head + this.n) % this.buf.length] = x;
    this.n++;
  }
  /** Remove and return the oldest item (-1 when empty). */
  shift(): number {
    if (this.n === 0) return -1;
    const x = this.buf[this.head];
    this.head = (this.head + 1) % this.buf.length;
    this.n--;
    return x;
  }
  peek(): number { return this.n ? this.buf[this.head] : -1; }
}

/**
 * Bucket queue for integer priorities 0..levels-1 (e.g. 8-bit grey levels):
 * O(1) push and pop, FIFO within a level. pop() returns an item of the lowest
 * non-empty level; pushing below the current level is allowed.
 */
export class BucketQueue {
  private buckets: Fifo[];
  private cur = 0;
  private n = 0;
  constructor(public readonly levels: number) {
    this.buckets = Array.from({ length: levels }, () => new Fifo(16));
    this.cur = levels;
  }
  get size(): number { return this.n; }
  isEmpty(): boolean { return this.n === 0; }
  /** The lowest non-empty level (levels when empty). */
  get level(): number { this.advance(); return this.cur; }
  push(item: number, priority: number): void {
    const p = Math.max(0, Math.min(this.levels - 1, Math.floor(priority)));
    this.buckets[p].push(item);
    if (p < this.cur) this.cur = p;
    this.n++;
  }
  pop(): number {
    if (this.n === 0) return -1;
    this.advance();
    this.n--;
    return this.buckets[this.cur].shift();
  }
  clear(): void { for (const b of this.buckets) b.clear(); this.n = 0; this.cur = this.levels; }
  private advance(): void {
    if (this.n === 0) { this.cur = this.levels; return; }
    if (this.cur >= this.levels) this.cur = 0;
    while (this.buckets[this.cur].isEmpty()) this.cur++;
  }
}
