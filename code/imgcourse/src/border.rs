/// What to read outside the image (the "border rule" of the filtering unit).
///
/// For an image `1 2 3` (width 3), the positions −2 −1 | 0 1 2 | 3 4 read:
///
/// | rule     | −2 | −1 | 0 1 2 | 3 | 4 |
/// |----------|----|----|-------|---|---|
/// | `Zero`   | 0  | 0  | 1 2 3 | 0 | 0 |
/// | `Clamp`  | 1  | 1  | 1 2 3 | 3 | 3 |
/// | `Mirror` | 2  | 1  | 1 2 3 | 3 | 2 |
/// | `Wrap`   | 2  | 3  | 1 2 3 | 1 | 2 |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Border {
    /// Zeros outside the image.
    Zero,
    /// Copy the nearest edge pixel ("replicate").
    Clamp,
    /// Mirror the image at its edge (the edge pixel itself is repeated).
    Mirror,
    /// Wrap around to the opposite side (periodic, as the Fourier transform assumes).
    Wrap,
}

impl Border {
    /// The position inside `0..n` that position `i` reads, or `None` for a zero.
    pub fn index(self, i: isize, n: usize) -> Option<usize> {
        let n = n as isize;
        if (0..n).contains(&i) {
            return Some(i as usize);
        }
        match self {
            Border::Zero => None,
            Border::Clamp => Some(i.clamp(0, n - 1) as usize),
            Border::Wrap => Some(i.rem_euclid(n) as usize),
            Border::Mirror => {
                // The mirrored image repeats with period 2n: 0 1 .. n-1 n-1 .. 1 0
                let j = i.rem_euclid(2 * n);
                Some(if j < n { j } else { 2 * n - 1 - j } as usize)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_in_the_docs() {
        let read = |b: Border| (-2..5).map(|i| b.index(i, 3).map(|j| j + 1).unwrap_or(0)).collect::<Vec<_>>();
        assert_eq!(read(Border::Zero), [0, 0, 1, 2, 3, 0, 0]);
        assert_eq!(read(Border::Clamp), [1, 1, 1, 2, 3, 3, 3]);
        assert_eq!(read(Border::Mirror), [2, 1, 1, 2, 3, 3, 2]);
        assert_eq!(read(Border::Wrap), [2, 3, 1, 2, 3, 1, 2]);
    }
}
