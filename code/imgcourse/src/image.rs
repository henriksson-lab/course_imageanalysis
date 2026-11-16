use crate::Border;
use std::fmt;
use std::ops::{Index, IndexMut};

/// A rectangular grid of pixels of type `T` (`f32` for grey values, `[f32; 3]` for colour,
/// `u32` for labels, `Complex` for spectra).
///
/// Pixel `(x, y)` is column `x` (from the left) and row `y` (from the top). The pixels
/// are stored row by row in one `Vec`: pixel `(x, y)` is element `y * width + x`.
#[derive(Clone, Debug, PartialEq)]
pub struct Image<T> {
    width: usize,
    height: usize,
    data: Vec<T>,
}

impl<T: Copy> Image<T> {
    /// An image with every pixel set to `value`.
    pub fn new(width: usize, height: usize, value: T) -> Self {
        Image { width, height, data: vec![value; width * height] }
    }

    /// An image whose pixel `(x, y)` is `f(x, y)`.
    pub fn from_fn(width: usize, height: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                data.push(f(x, y));
            }
        }
        Image { width, height, data }
    }

    /// An image from its pixels, row by row (`data.len()` must be `width * height`).
    pub fn from_vec(width: usize, height: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), width * height, "from_vec: {} values for a {width} × {height} image", data.len());
        Image { width, height, data }
    }

    /// A small image written out row by row, handy for examples done by hand:
    /// `Image::from_rows(&[[1.0, 2.0], [3.0, 4.0]])` is 2 wide and 2 high.
    pub fn from_rows<const W: usize>(rows: &[[T; W]]) -> Self {
        let data = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Image { width: W, height: rows.len(), data }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// The pixel at column `x`, row `y`. Panics outside the image; see `get_border`.
    pub fn get(&self, x: usize, y: usize) -> T {
        self[(x, y)]
    }

    pub fn set(&mut self, x: usize, y: usize, value: T) {
        self[(x, y)] = value;
    }

    /// Is `(x, y)` inside the image? Takes signed numbers, so that `x - 1` may go below 0.
    pub fn contains(&self, x: isize, y: isize) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }

    /// The pixel at `(x, y)`, which may lie outside the image: then `border` decides
    /// what is read (zero, the nearest edge pixel, a mirror image, or the opposite side).
    pub fn get_border(&self, x: isize, y: isize, border: Border) -> T
    where
        T: Default,
    {
        match (border.index(x, self.width), border.index(y, self.height)) {
            (Some(x), Some(y)) => self[(x, y)],
            _ => T::default(),
        }
    }

    /// All pixel positions `(x, y)`, row by row: `for (x, y) in img.coords() { ... }`.
    pub fn coords(&self) -> impl Iterator<Item = (usize, usize)> {
        let w = self.width;
        (0..self.height).flat_map(move |y| (0..w).map(move |x| (x, y)))
    }

    /// A new image with `f` applied to every pixel value.
    pub fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> Image<U> {
        Image { width: self.width, height: self.height, data: self.data.iter().map(|&v| f(v)).collect() }
    }

    /// A new image with `f(x, y, value)` at every pixel.
    pub fn map_xy<U: Copy>(&self, f: impl Fn(usize, usize, T) -> U) -> Image<U> {
        Image::from_fn(self.width, self.height, |x, y| f(x, y, self[(x, y)]))
    }

    /// Combine two images of the same size pixel by pixel: `a.zip_map(&b, |p, q| p - q)`.
    pub fn zip_map<U: Copy, V: Copy>(&self, other: &Image<U>, f: impl Fn(T, U) -> V) -> Image<V> {
        assert_eq!((self.width, self.height), (other.width, other.height), "zip_map: images of different sizes");
        let data = self.data.iter().zip(&other.data).map(|(&a, &b)| f(a, b)).collect();
        Image { width: self.width, height: self.height, data }
    }

    /// The `w × h` piece whose top-left corner is `(x0, y0)`.
    pub fn crop(&self, x0: usize, y0: usize, w: usize, h: usize) -> Image<T> {
        Image::from_fn(w, h, |x, y| self[(x0 + x, y0 + y)])
    }

    /// All pixel values, row by row.
    pub fn pixels(&self) -> &[T] {
        &self.data
    }

    pub fn pixels_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Row `y` as a slice.
    pub fn row(&self, y: usize) -> &[T] {
        &self.data[y * self.width..(y + 1) * self.width]
    }
}

impl Image<f32> {
    pub fn min(&self) -> f32 {
        self.data.iter().copied().fold(f32::INFINITY, f32::min)
    }

    pub fn max(&self) -> f32 {
        self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.data.len() as f32
    }
}

impl<T> Index<(usize, usize)> for Image<T> {
    type Output = T;
    fn index(&self, (x, y): (usize, usize)) -> &T {
        assert!(x < self.width && y < self.height, "pixel ({x}, {y}) is outside the {} × {} image", self.width, self.height);
        &self.data[y * self.width + x]
    }
}

impl<T> IndexMut<(usize, usize)> for Image<T> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut T {
        assert!(x < self.width && y < self.height, "pixel ({x}, {y}) is outside the {} × {} image", self.width, self.height);
        &mut self.data[y * self.width + x]
    }
}

/// Prints the image as a table, one row per line: `println!("{img:.2}")` for 2 decimals.
/// Meant for small images worked out by hand.
impl<T: fmt::Display> fmt::Display for Image<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let cells: Vec<String> = self
            .data
            .iter()
            .map(|v| match f.precision() {
                Some(p) => format!("{v:.p$}"),
                None => format!("{v}"),
            })
            .collect();
        let w = cells.iter().map(|c| c.chars().count()).max().unwrap_or(0);
        for y in 0..self.height {
            for x in 0..self.width {
                write!(f, "{:>w$}", cells[y * self.width + x], w = w + 1)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
