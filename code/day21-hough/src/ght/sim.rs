//! Moving polygons and the key template for the generalised Hough transform (test scenes, edge
//! points and drawing are in `src/common.rs`). Not part of the exercise: read it if you like.

/// Move polygons: rotate by `phi` and scale by `scale` about `centre`, then shift by `shift`.
pub fn transform(polys: &[Vec<[f32; 2]>], centre: [f32; 2], phi: f32, scale: f32, shift: [f32; 2]) -> Vec<Vec<[f32; 2]>> {
    let (c, s) = (phi.cos(), phi.sin());
    polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|p| {
                    let (dx, dy) = (p[0] - centre[0], p[1] - centre[1]);
                    [centre[0] + shift[0] + scale * (c * dx - s * dy), centre[1] + shift[1] + scale * (s * dx + c * dy)]
                })
                .collect()
        })
        .collect()
}

/// A key as polygons in a 64 × 64 frame: a round bow on the left, a shaft and two teeth.
pub fn key() -> Vec<Vec<[f32; 2]>> {
    let bow: Vec<[f32; 2]> = (0..36)
        .map(|i| {
            let a = i as f32 * std::f32::consts::TAU / 36.0;
            [16.0 + 11.0 * a.cos(), 32.0 + 11.0 * a.sin()]
        })
        .collect();
    vec![
        bow,
        vec![[25.0, 28.0], [58.0, 28.0], [58.0, 35.0], [25.0, 35.0]],
        vec![[42.0, 35.0], [47.0, 35.0], [47.0, 43.0], [42.0, 43.0]],
        vec![[51.0, 35.0], [56.0, 35.0], [56.0, 41.0], [51.0, 41.0]],
    ]
}
