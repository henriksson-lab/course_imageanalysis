// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Basins a–e are leaves 0–4; the 3 × 3 grid of B3 is numbered row by row.
// Run: cargo test -p day29 merge_trees::

use day29::merge_trees::*;

/// B1: minima 1, 3, 0, 4, 2; areas 10, 2, 8, 3, 6; passes a|b 5, b|c 4, c|d 7, d|e 6.
const MIN: [f32; 5] = [1.0, 3.0, 0.0, 4.0, 2.0];
const AREA: [f32; 5] = [10.0, 2.0, 8.0, 3.0, 6.0];
fn passes() -> Vec<Edge> {
    vec![(0, 1, 5.0), (1, 2, 4.0), (2, 3, 7.0), (3, 4, 6.0)]
}

/// The regions as letters a, b, c, … in order of first appearance.
fn pattern(labels: &[usize]) -> String {
    let mut seen: Vec<usize> = Vec::new();
    labels
        .iter()
        .map(|l| {
            let k = seen.iter().position(|s| s == l).unwrap_or_else(|| {
                seen.push(*l);
                seen.len() - 1
            });
            (b'a' + k as u8) as char
        })
        .collect()
}

/// The merges as "leaves of a | leaves of b @ level", e.g. "b|c@4".
fn describe(n: usize, merges: &[Merge]) -> Vec<String> {
    let names = ["a", "b", "c", "d", "e", "f", "g", "h", "i"];
    let mut leaves: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut out = Vec::new();
    for m in merges {
        let (mut la, mut lb) = (leaves[m.a].clone(), leaves[m.b].clone());
        la.sort();
        lb.sort();
        let show = |l: &[usize]| l.iter().map(|&i| names[i]).collect::<String>();
        let (x, y) = if la[0] < lb[0] { (show(&la), show(&lb)) } else { (show(&lb), show(&la)) };
        out.push(format!("{x}|{y}@{}", (m.level * 10.0).round() / 10.0));
        leaves.push([la, lb].concat());
    }
    out
}

#[test]
fn c1_altitude_b1a() {
    let t = altitude_tree(5, &passes());
    assert_eq!(describe(5, &t), ["b|c@4", "a|bc@5", "d|e@6", "abc|de@7"], "B1(a): the altitude tree");
}

#[test]
fn c2_dynamics_b1b() {
    let t = attribute_tree(5, &passes(), &MIN, &AREA, Attribute::Dynamics);
    assert_eq!(describe(5, &t), ["b|c@1", "d|e@2", "a|bc@4", "abc|de@5"], "B1(b): the dynamics tree");
}

#[test]
fn c2_area_b1c() {
    let t = attribute_tree(5, &passes(), &MIN, &AREA, Attribute::Area);
    assert_eq!(describe(5, &t), ["b|c@2", "d|e@3", "bc|de@9", "a|bcde@10"], "B1(c): the area tree");
}

#[test]
fn c2_volume_b1d() {
    let t = attribute_tree(5, &passes(), &MIN, &AREA, Attribute::Volume);
    let levels: Vec<f32> = t.iter().map(|m| m.level).collect();
    assert_eq!(levels, [2.0, 6.0, 40.0, 45.0], "B1(d): the volume values in order");
}

#[test]
fn c3_cuts_b1_b2() {
    assert_eq!(pattern(&cut_first(5, &altitude_tree(5, &passes()), 2)), "aaabc", "B1(a): two merges by altitude");
    let dynamics = attribute_tree(5, &passes(), &MIN, &AREA, Attribute::Dynamics);
    assert_eq!(pattern(&cut_first(5, &dynamics, 2)), "abbcc", "B1(b): two merges by dynamics");
    assert_eq!(pattern(&cut(5, &dynamics, 3.0)), "abbcc", "B2(b): threshold 3");
    assert_eq!(pattern(&cut(5, &dynamics, 4.5)), "aaabb", "B2(c): threshold 4.5 = the h-minima with h = 4.5");
    assert_eq!(pattern(&cut_first(5, &dynamics, 0)), "abcde", "no merges: every leaf alone");
}

#[test]
fn c4_ucm_b2a() {
    let dynamics = attribute_tree(5, &passes(), &MIN, &AREA, Attribute::Dynamics);
    let u = ucm(5, &dynamics, &passes());
    assert_eq!(u, [4.0, 1.0, 5.0, 2.0], "B2(a): UCM of the boundaries a|b, b|c, c|d, d|e");
    let u = ucm(5, &altitude_tree(5, &passes()), &passes());
    assert_eq!(u, [5.0, 4.0, 7.0, 6.0], "altitude tree: each boundary disappears at its own pass (no lower path)");
}

#[test]
fn c4_ultrametric_of_far_leaves() {
    // u(a, e) for the altitude tree is 7: a and e first share a region at the top
    let t = altitude_tree(5, &passes());
    let u = ucm(5, &t, &[(0, 4, 0.0), (0, 2, 0.0), (3, 4, 0.0)]);
    assert_eq!(u, [7.0, 5.0, 6.0], "u(a, e), u(a, c), u(d, e) in the altitude tree");
}

#[test]
fn c5_rag_b3() {
    let means = [10.0, 12.0, 30.0, 11.0, 25.0, 31.0, 40.0, 26.0, 50.0];
    let mut pairs = Vec::new();
    for y in 0..3 {
        for x in 0..3 {
            if x < 2 {
                pairs.push((3 * y + x, 3 * y + x + 1));
            }
            if y < 2 {
                pairs.push((3 * y + x, 3 * y + x + 3));
            }
        }
    }
    let merges = rag_merge(&means, &[1.0; 9], &pairs);
    let levels: Vec<f32> = merges.iter().map(|m| (m.level * 10.0).round() / 10.0).collect();
    assert_eq!(levels, [1.0, 1.0, 1.0, 1.5, 5.0, 12.0, 19.4, 26.9], "B3: the differences of the 8 merges");
    let before_last_two = cut_first(9, &merges, 6);
    assert_eq!(pattern(&before_last_two), "aababbbbc", "B3: {{10, 11, 12}}, {{25, 26, 30, 31, 40}}, {{50}} before the last two merges");
}
