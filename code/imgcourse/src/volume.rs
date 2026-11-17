use std::ops::{Index, IndexMut};

/// A 3D grid of voxels: `vol[(x, y, z)]`, with `x` the column, `y` the row and `z` the slice,
/// all counted from 0. Stored slice by slice, each slice row by row.
#[derive(Clone, Debug, PartialEq)]
pub struct Volume<T> {
    width: usize,
    height: usize,
    depth: usize,
    data: Vec<T>,
}

impl<T: Copy> Volume<T> {
    pub fn new(width: usize, height: usize, depth: usize, value: T) -> Self {
        Volume { width, height, depth, data: vec![value; width * height * depth] }
    }

    /// A volume whose voxel `(x, y, z)` is `f(x, y, z)`.
    pub fn from_fn(width: usize, height: usize, depth: usize, mut f: impl FnMut(usize, usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(width * height * depth);
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    data.push(f(x, y, z));
                }
            }
        }
        Volume { width, height, depth, data }
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Is `(x, y, z)` inside the volume? Signed, so that `x - 1` may go below 0.
    pub fn contains(&self, x: isize, y: isize, z: isize) -> bool {
        x >= 0 && y >= 0 && z >= 0 && (x as usize) < self.width && (y as usize) < self.height && (z as usize) < self.depth
    }

    /// All voxel positions, slice by slice.
    pub fn coords(&self) -> impl Iterator<Item = (usize, usize, usize)> {
        let (w, h, d) = (self.width, self.height, self.depth);
        (0..d).flat_map(move |z| (0..h).flat_map(move |y| (0..w).map(move |x| (x, y, z))))
    }

    pub fn map<U: Copy>(&self, mut f: impl FnMut(T) -> U) -> Volume<U> {
        Volume { width: self.width, height: self.height, depth: self.depth, data: self.data.iter().map(|&v| f(v)).collect() }
    }

    /// Slice `z` as an image.
    pub fn slice(&self, z: usize) -> crate::Image<T> {
        crate::Image::from_fn(self.width, self.height, |x, y| self[(x, y, z)])
    }

    pub fn voxels(&self) -> &[T] {
        &self.data
    }
}

impl<T> Index<(usize, usize, usize)> for Volume<T> {
    type Output = T;
    fn index(&self, (x, y, z): (usize, usize, usize)) -> &T {
        assert!(x < self.width && y < self.height && z < self.depth, "voxel ({x}, {y}, {z}) is outside the {} × {} × {} volume", self.width, self.height, self.depth);
        &self.data[(z * self.height + y) * self.width + x]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for Volume<T> {
    fn index_mut(&mut self, (x, y, z): (usize, usize, usize)) -> &mut T {
        assert!(x < self.width && y < self.height && z < self.depth, "voxel ({x}, {y}, {z}) is outside the {} × {} × {} volume", self.width, self.height, self.depth);
        &mut self.data[(z * self.height + y) * self.width + x]
    }
}
