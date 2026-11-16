use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// A complex number `re + i·im`, for Fourier transforms.
///
/// ```
/// use imgcourse::Complex;
/// let w = Complex::from_angle(std::f32::consts::PI / 2.0);   // e^{iπ/2} = i
/// assert!((w - Complex::new(0.0, 1.0)).abs() < 1e-6);
/// assert!((w * w + Complex::from(1.0)).abs() < 1e-6);        // i² = −1
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

    pub fn new(re: f32, im: f32) -> Self {
        Complex { re, im }
    }

    /// `e^{iθ} = cos θ + i sin θ`: the point at angle `theta` on the unit circle.
    pub fn from_angle(theta: f32) -> Self {
        Complex { re: theta.cos(), im: theta.sin() }
    }

    /// `r e^{iθ}`.
    pub fn from_polar(r: f32, theta: f32) -> Self {
        Complex { re: r * theta.cos(), im: r * theta.sin() }
    }

    /// The complex conjugate `re − i·im`.
    pub fn conj(self) -> Self {
        Complex { re: self.re, im: -self.im }
    }

    /// The size `|z|`.
    pub fn abs(self) -> f32 {
        self.re.hypot(self.im)
    }

    /// `|z|²`.
    pub fn norm_sqr(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// The angle (phase) in radians, between −π and π.
    pub fn arg(self) -> f32 {
        self.im.atan2(self.re)
    }
}

impl From<f32> for Complex {
    fn from(re: f32) -> Self {
        Complex { re, im: 0.0 }
    }
}

impl Add for Complex {
    type Output = Complex;
    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, o: Complex) {
        self.re += o.re;
        self.im += o.im;
    }
}

impl Sub for Complex {
    type Output = Complex;
    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl Neg for Complex {
    type Output = Complex;
    fn neg(self) -> Complex {
        Complex::new(-self.re, -self.im)
    }
}

impl Mul for Complex {
    type Output = Complex;
    fn mul(self, o: Complex) -> Complex {
        Complex::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}

impl Mul<f32> for Complex {
    type Output = Complex;
    fn mul(self, s: f32) -> Complex {
        Complex::new(self.re * s, self.im * s)
    }
}

impl Mul<Complex> for f32 {
    type Output = Complex;
    fn mul(self, z: Complex) -> Complex {
        z * self
    }
}

impl Div<f32> for Complex {
    type Output = Complex;
    fn div(self, s: f32) -> Complex {
        Complex::new(self.re / s, self.im / s)
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let p = f.precision().unwrap_or(2);
        // Values that round to zero are printed as 0, not as -0.
        let scale = 10f32.powi(p as i32);
        let tidy = |v: f32| if (v * scale).round() == 0.0 { 0.0 } else { v };
        let (re, im) = (tidy(self.re), tidy(self.im));
        let sign = if im < 0.0 { '-' } else { '+' };
        write!(f, "{re:.p$}{sign}{:.p$}i", im.abs())
    }
}
