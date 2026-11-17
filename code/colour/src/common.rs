//! Helpers shared by the units of the day: the sRGB transfer function (written in the unit
//! `rgb_linear`, used ready-made by the later units).

/// sRGB decoding: stored value in [0, 1] → linear light.
pub fn srgb_decode(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB encoding: linear light in [0, 1] → stored value.
pub fn srgb_encode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}
