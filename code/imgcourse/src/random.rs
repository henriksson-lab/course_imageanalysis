/// A small random number generator (xorshift), so that every run with the same
/// seed gives the same numbers. Good enough for noise and sampling, not for security.
///
/// ```
/// use imgcourse::Rng;
/// let mut rng = Rng::new(42);
/// let u = rng.uniform();         // between 0 and 1
/// let g = rng.normal();          // mean 0, standard deviation 1
/// let k = rng.below(10);         // 0, 1, ..., 9
/// assert!((0.0..1.0).contains(&u) && g.is_finite() && k < 10);
/// ```
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // splitmix64 of the seed, so that small seeds still give a well-mixed state
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Rng { state: (z ^ (z >> 31)) | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Normal (Gaussian), mean 0 and standard deviation 1 (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = 1.0 - self.uniform(); // in (0, 1], so the log is finite
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// A whole number in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}
