//! A simulated field of cells imaged in the three channels of a sensitised-emission
//! measurement, with photon (Poisson) noise, and the controls.
//! Not part of the exercise: read it if you like.

use imgcourse::{samples, Image, Rng};

/// True bleed-through, cross-excitation and instrument factor of the simulated microscope.
pub const D_TRUE: f32 = 0.5;
pub const A_TRUE: f32 = 0.1;
pub const G_TRUE: f32 = 2.0;
/// Camera offset added to every image (photons).
pub const OFFSET: f32 = 100.0;

/// The kinds of cells in the field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    DonorOnly,
    AcceptorOnly,
    NoFret,
    Fusion,
    Binding,
}

impl Kind {
    /// The paired fraction f and the efficiency E of a paired donor.
    pub fn f_and_e(self) -> (f32, f32) {
        match self {
            Kind::Fusion => (1.0, 0.3),
            Kind::Binding => (0.5, 0.4),
            _ => (0.0, 0.0),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Kind::DonorOnly => "donor only",
            Kind::AcceptorOnly => "acceptor only",
            Kind::NoFret => "no FRET",
            Kind::Fusion => "fusion",
            Kind::Binding => "binding",
        }
    }
}

/// The three images of a field: I_DD, I_DA, I_AA (photon counts, including the offset).
pub struct Field {
    pub idd: Image<f32>,
    pub ida: Image<f32>,
    pub iaa: Image<f32>,
    /// The cells: 0 background, 1, 2, … the cells.
    pub labels: Image<u32>,
    /// The kind of each cell (index = label − 1).
    pub kinds: Vec<Kind>,
}

/// A field of `n` cells of the given kinds (repeated in turn), `photons` the typical donor
/// signal per pixel. `acceptor_scale` multiplies the acceptor amount of the FRET cells
/// (fusion and binding), e.g. to add free acceptor. Same seed, same field.
pub fn field(kinds: &[Kind], n: usize, photons: f32, acceptor_scale: f32, seed: u64) -> Field {
    let (_, labels) = samples::nuclei(128, n, seed);
    let mut rng = Rng::new(seed + 1);
    let cell_kinds: Vec<Kind> = (0..n).map(|i| kinds[i % kinds.len()]).collect();
    // amounts of donor and acceptor per cell (photon scale)
    let amounts: Vec<(f32, f32)> = cell_kinds
        .iter()
        .map(|&k| {
            let nd = photons * (0.5 + rng.uniform());
            let na = photons * (0.8 + 0.8 * rng.uniform());
            match k {
                Kind::DonorOnly => (nd, 0.0),
                Kind::AcceptorOnly => (0.0, na),
                Kind::NoFret => (nd, na),
                _ => (nd, na * acceptor_scale),
            }
        })
        .collect();
    let mut idd = Image::new(128, 128, 0.0f32);
    let mut ida = Image::new(128, 128, 0.0f32);
    let mut iaa = Image::new(128, 128, 0.0f32);
    for (x, y) in labels.coords() {
        let l = labels[(x, y)] as usize;
        let (mut m_dd, mut m_da, mut m_aa) = (0.0, 0.0, 0.0);
        if l > 0 {
            let (nd, na) = amounts[l - 1];
            let (f, e) = cell_kinds[l - 1].f_and_e();
            m_dd = nd * (1.0 - f * e);
            m_aa = na;
            let fc = G_TRUE * nd * f * e;
            m_da = D_TRUE * m_dd + A_TRUE * m_aa + fc;
        }
        idd[(x, y)] = rng.poisson(m_dd + OFFSET);
        ida[(x, y)] = rng.poisson(m_da + OFFSET);
        iaa[(x, y)] = rng.poisson(m_aa + OFFSET);
    }
    Field { idd, ida, iaa, labels, kinds: cell_kinds }
}

/// The mask of the cells of a field (label > 0).
pub fn cell_mask(labels: &Image<u32>) -> Image<bool> {
    labels.map(|l| l > 0)
}
