//! High-precision preflop all-in equity table for the 169 canonical classes.
//!
//! Replaces the legacy Monte-Carlo `preflop_169.bin`: we regenerate the table
//! ourselves with a fixed seed and documented precision (binomial SE =
//! sqrt(0.25/sims) per matchup), and validate the evaluator itself against an
//! independent exact numpy enumeration in tier A. Antisymmetry E[j][i] = 1 -
//! E[i][j] holds exactly by construction; the diagonal is set analytically
//! (same-class split, E = 0.5 minus the impossible-mass correction = exactly 0.5).
use crate::card::{hand_169_to_cards, Card, CardMask};
use crate::eval::eval_nlh_river;
use rayon::prelude::*;

pub const N: usize = 169;
pub type EquityTable = [[f32; N]; N];

#[derive(Clone, Copy)]
struct FastRng(u64);

impl FastRng {
    #[inline(always)]
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x5450_504f_4b45_5221 } else { seed })
    }
    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    #[inline(always)]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// All specific combos belonging to canonical class `idx`.
pub fn class_combos(idx: usize) -> Vec<[Card; 2]> {
    let cards = hand_169_to_cards(idx);
    let (r1, r2, suited) = (cards[0].rank(), cards[1].rank(), cards[0].suit() == cards[1].suit());
    let mut out = Vec::new();
    for s1 in 0..4u8 {
        for s2 in 0..4u8 {
            let ok = if r1 == r2 {
                s1 < s2 // pairs: C(4,2) = 6 combos
            } else if suited {
                s1 == s2 // suited: 4 combos
            } else {
                s1 != s2 // offsuit: 12 combos
            };
            if ok {
                out.push([Card::new(r1, s1), Card::new(r2, s2)]);
            }
        }
    }
    out
}

fn disjoint(h: &[Card; 2], v: &[Card; 2]) -> bool {
    h[0] != v[0] && h[0] != v[1] && h[1] != v[0] && h[1] != v[1]
}

/// Monte-Carlo equity of class i vs class j (fixed seed; SE = sqrt(0.25/sims)).
fn simulate(i: usize, j: usize, sims: u64, seed: u64) -> f32 {
    if i == j {
        return 0.5; // symmetric class split (ties included)
    }
    let combos_i = class_combos(i);
    let combos_j = class_combos(j);
    let mut rng = FastRng(seed ^ (i as u64).wrapping_mul(26_544_357_614) ^ (j as u64).wrapping_mul(40_503));
    let mut score = 0.0f64;
    let mut remaining: Vec<Card> = Vec::with_capacity(48);
    for _ in 0..sims {
        let h = combos_i[rng.below(combos_i.len())];
        let v = loop {
            let cand = combos_j[rng.below(combos_j.len())];
            if disjoint(&h, &cand) {
                break cand;
            }
        };
        // remaining deck (48 cards) via mask
        let mut dead = CardMask::EMPTY;
        dead.insert(h[0]);
        dead.insert(h[1]);
        dead.insert(v[0]);
        dead.insert(v[1]);
        remaining.clear();
        let mut alive = CardMask::FULL;
        alive.0 &= !dead.0;
        while let Some(c) = alive.pop_card() {
            remaining.push(c);
        }
        // partial Fisher-Yates: draw 5 board cards
        let mut board = [Card(0); 5];
        for k in 0..5 {
            let pick = k + rng.below(remaining.len() - k);
            remaining.swap(k, pick);
            board[k] = remaining[k];
        }
        let rh = eval_nlh_river(h, board);
        let rv = eval_nlh_river(v, board);
        score += if rh.0 < rv.0 {
            1.0
        } else if rh.0 == rv.0 {
            0.5
        } else {
            0.0
        };
    }
    (score / sims as f64) as f32
}

/// Generate the full 169x169 table (parallel over unordered pairs).
pub fn generate(sims: u64, seed: u64) -> EquityTable {
    let pairs: Vec<(usize, usize)> = (0..N).flat_map(|i| (i..N).map(move |j| (i, j))).collect();
    let equities: Vec<f32> = pairs
        .par_iter()
        .map(|&(i, j)| simulate(i, j, sims, seed))
        .collect();
    let mut table = [[0.5f32; N]; N];
    for (k, &(i, j)) in pairs.iter().enumerate() {
        table[i][j] = equities[k];
        table[j][i] = 1.0 - equities[k];
    }
    table
}

/// Serialize as little-endian u16 (equity * 65535, rounded).
pub fn save(table: &EquityTable, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut buf = Vec::with_capacity(N * N * 2);
    for row in table {
        for &e in row {
            let v = (e.clamp(0.0, 1.0) * 65535.0).round() as u16;
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::File::create(path)?.write_all(&buf)
}

/// Load a table saved by `save`.
pub fn load(path: &str) -> std::io::Result<EquityTable> {
    let bytes = std::fs::read(path)?;
    assert_eq!(bytes.len(), N * N * 2, "equity table size mismatch");
    let mut table = [[0.5f32; N]; N];
    for i in 0..N {
        for j in 0..N {
            let o = (i * N + j) * 2;
            table[i][j] = u16::from_le_bytes([bytes[o], bytes[o + 1]]) as f32 / 65535.0;
        }
    }
    Ok(table)
}
