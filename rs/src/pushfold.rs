//! # Exact 10 BB Push/Fold HUNL Benchmark & Exploitability Engine
//!
//! Provides mathematically exact Best Response, NashConv, and Nash Equilibrium solving
//! across all 169 canonical preflop hand classes with full card removal (blockers).
//!
//! Payoffs:
//! - BTN posts 0.5 BB (SB), BB posts 1.0 BB.
//! - BTN actions: Fold (-0.5 BB to BTN) or Push (All-in to 10 BB).
//! - If BTN Pushes, BB actions: Fold (-1.0 BB to BB, +1.0 BB to BTN) or Call (Showdown for 20 BB pot).
//! - At Showdown: BTN EV = 20 * E(i, j) - 10 BB, BB EV = 10 - 20 * E(i, j) BB.

use crate::card::{hand_169_index, Card};

pub const NUM_CANONICAL_HANDS: usize = 169;

/// Number of card combinations for each of the 169 canonical hands:
/// - Pairs (0..13): 6 combos
/// - Suited (13..91): 4 combos
/// - Offsuit (91..169): 12 combos
pub fn hand_combos(idx: usize) -> usize {
    if idx < 13 {
        6
    } else if idx < 91 {
        4
    } else {
        12
    }
}

/// Returns the human-readable hand name (e.g. "AA", "AKs", "72o") and combo count.
pub fn hand_name(idx: usize) -> (&'static str, usize) {
    static NAMES: [&str; 169] = [
        "22", "33", "44", "55", "66", "77", "88", "99", "TT", "JJ", "QQ", "KK", "AA",
        // Suited 13..91
        "32s", "42s", "43s", "52s", "53s", "54s", "62s", "63s", "64s", "65s",
        "72s", "73s", "74s", "75s", "76s", "82s", "83s", "84s", "85s", "86s", "87s",
        "92s", "93s", "94s", "95s", "96s", "97s", "98s",
        "T2s", "T3s", "T4s", "T5s", "T6s", "T7s", "T8s", "T9s",
        "J2s", "J3s", "J4s", "J5s", "J6s", "J7s", "J8s", "J9s", "JTs",
        "Q2s", "Q3s", "Q4s", "Q5s", "Q6s", "Q7s", "Q8s", "Q9s", "QTs", "QJs",
        "K2s", "K3s", "K4s", "K5s", "K6s", "K7s", "K8s", "K9s", "KTs", "KJs", "KQs",
        "A2s", "A3s", "A4s", "A5s", "A6s", "A7s", "A8s", "A9s", "ATs", "AJs", "AQs", "AKs",
        // Offsuit 91..169
        "32o", "42o", "43o", "52o", "53o", "54o", "62o", "63o", "64o", "65o",
        "72o", "73o", "74o", "75o", "76o", "82o", "83o", "84o", "85o", "86o", "87o",
        "92o", "93o", "94o", "95o", "96o", "97o", "98o",
        "T2o", "T3o", "T4o", "T5o", "T6o", "T7o", "T8o", "T9o",
        "J2o", "J3o", "J4o", "J5o", "J6o", "J7o", "J8o", "J9o", "JTo",
        "Q2o", "Q3o", "Q4o", "Q5o", "Q6o", "Q7o", "Q8o", "Q9o", "QTo", "QJo",
        "K2o", "K3o", "K4o", "K5o", "K6o", "K7o", "K8o", "K9o", "KTo", "KJo", "KQo",
        "A2o", "A3o", "A4o", "A5o", "A6o", "A7o", "A8o", "A9o", "ATo", "AJo", "AQo", "AKo",
    ];
    (NAMES[idx], hand_combos(idx))
}

/// Exact Push/Fold Game Model with precomputed transition tensors.
pub struct PushFoldGame {
    pub stack_bb: f32,
    pub small_blind_bb: f32,
    pub big_blind_bb: f32,
    /// Prior probability of each canonical hand P(i) = combos(i) / 1326
    pub prior: [f32; NUM_CANONICAL_HANDS],
    /// Conditional probability P(j | i) of opponent having hand j given Hero has hand i
    pub prob_opp_given_hero: [[f32; NUM_CANONICAL_HANDS]; NUM_CANONICAL_HANDS],
    /// Canonical showdown equity E(i, j) of hand i vs hand j
    pub equity: [[f32; NUM_CANONICAL_HANDS]; NUM_CANONICAL_HANDS],
}

impl PushFoldGame {
    /// Constructs a new PushFoldGame with exact precomputed blockers and equities.
    pub fn with_equity(
        stack_bb: f32,
        small_blind_bb: f32,
        big_blind_bb: f32,
        equity: [[f32; NUM_CANONICAL_HANDS]; NUM_CANONICAL_HANDS],
    ) -> Self {
        let mut prior = [0.0f32; NUM_CANONICAL_HANDS];
        for i in 0..NUM_CANONICAL_HANDS {
            prior[i] = hand_combos(i) as f32 / 1326.0;
        }

        // 1. Build all 1326 specific hole card pairs
        let mut all_combos = Vec::with_capacity(1326);
        for u1 in 0..52u8 {
            for u2 in (u1 + 1)..52u8 {
                let c1 = Card(u1);
                let c2 = Card(u2);
                let idx = hand_169_index(c1, c2);
                all_combos.push((c1, c2, idx));
            }
        }
        assert_eq!(all_combos.len(), 1326);

        // 2. Count joint unblocked combinations C(i, j)
        let mut blocker_counts = [[0u32; NUM_CANONICAL_HANDS]; NUM_CANONICAL_HANDS];
        for a in 0..1326 {
            let (c1, c2, idx1) = all_combos[a];
            for b in (a + 1)..1326 {
                let (c3, c4, idx2) = all_combos[b];
                if c1 != c3 && c1 != c4 && c2 != c3 && c2 != c4 {
                    blocker_counts[idx1][idx2] += 1;
                    blocker_counts[idx2][idx1] += 1;
                }
            }
        }

        let mut prob_opp_given_hero = [[0.0f32; NUM_CANONICAL_HANDS]; NUM_CANONICAL_HANDS];
        for i in 0..NUM_CANONICAL_HANDS {
            let hero_combos = hand_combos(i) as f32;
            let total_opp_combos = hero_combos * 1225.0;
            for j in 0..NUM_CANONICAL_HANDS {
                prob_opp_given_hero[i][j] = blocker_counts[i][j] as f32 / total_opp_combos;
            }
        }

        // 3. Equity matrix is supplied by the caller (kalay_rs::equitygen)
        let equity = equity;

        Self {
            stack_bb,
            small_blind_bb,
            big_blind_bb,
            prior,
            prob_opp_given_hero,
            equity,
        }
    }

    /// Evaluates expected payoffs for (BTN, BB) given strategies (p = BTN Push%, q = BB Call%).
    /// Returns (u_btn, u_bb). Due to zero-sum: u_btn + u_bb == 0.
    pub fn evaluate_payoffs(&self, p: &[f32; 169], q: &[f32; 169]) -> (f32, f32) {
        let mut u_btn = 0.0f32;
        let pot_showdown = 2.0 * self.stack_bb;

        for i in 0..NUM_CANONICAL_HANDS {
            let p_push = p[i].clamp(0.0, 1.0);
            let p_fold = 1.0 - p_push;

            // If BTN folds: loses small blind
            let ev_fold = -self.small_blind_bb;

            // If BTN pushes:
            let mut ev_push = 0.0f32;
            for j in 0..NUM_CANONICAL_HANDS {
                let p_j = self.prob_opp_given_hero[i][j];
                let q_call = q[j].clamp(0.0, 1.0);
                let q_fold = 1.0 - q_call;

                let payoff_bb_folds = self.big_blind_bb;
                let payoff_bb_calls = pot_showdown * self.equity[i][j] - self.stack_bb;

                ev_push += p_j * (q_fold * payoff_bb_folds + q_call * payoff_bb_calls);
            }

            u_btn += self.prior[i] * (p_fold * ev_fold + p_push * ev_push);
        }

        (u_btn, -u_btn)
    }

    /// Computes BB's exact Best Response against a fixed BTN push strategy p.
    /// Returns (q_br, u_bb_br).
    pub fn best_response_bb(&self, p: &[f32; 169]) -> ([f32; 169], f32) {
        let mut q_br = [0.0f32; 169];
        let pot_showdown = 2.0 * self.stack_bb;
        let fold_payoff = -self.big_blind_bb;

        let mut u_bb = 0.0f32;

        for j in 0..NUM_CANONICAL_HANDS {
            let mut weight_push = 0.0f32;
            let mut ev_call_sum = 0.0f32;

            for i in 0..NUM_CANONICAL_HANDS {
                let p_i_given_j = self.prob_opp_given_hero[j][i];
                let p_push = p[i].clamp(0.0, 1.0);
                let p_fold = 1.0 - p_push;

                // When BTN folds, BB wins small blind
                u_bb += self.prior[j] * p_i_given_j * p_fold * self.small_blind_bb;

                let w = p_i_given_j * p_push;
                weight_push += w;
                let sd_payoff = pot_showdown * (1.0 - self.equity[i][j]) - self.stack_bb;
                ev_call_sum += w * sd_payoff;
            }

            if weight_push > 1e-9 {
                let ev_call = ev_call_sum / weight_push;
                if ev_call > fold_payoff {
                    q_br[j] = 1.0;
                    u_bb += self.prior[j] * ev_call_sum;
                } else {
                    q_br[j] = 0.0;
                    u_bb += self.prior[j] * weight_push * fold_payoff;
                }
            } else {
                q_br[j] = 0.0;
            }
        }

        (q_br, u_bb)
    }

    /// Computes BTN's exact Best Response against a fixed BB call strategy q.
    /// Returns (p_br, u_btn_br).
    pub fn best_response_btn(&self, q: &[f32; 169]) -> ([f32; 169], f32) {
        let mut p_br = [0.0f32; 169];
        let pot_showdown = 2.0 * self.stack_bb;
        let fold_payoff = -self.small_blind_bb;

        let mut u_btn = 0.0f32;

        for i in 0..NUM_CANONICAL_HANDS {
            let mut ev_push = 0.0f32;
            for j in 0..NUM_CANONICAL_HANDS {
                let p_j = self.prob_opp_given_hero[i][j];
                let q_call = q[j].clamp(0.0, 1.0);
                let q_fold = 1.0 - q_call;

                let payoff_bb_folds = self.big_blind_bb;
                let payoff_bb_calls = pot_showdown * self.equity[i][j] - self.stack_bb;

                ev_push += p_j * (q_fold * payoff_bb_folds + q_call * payoff_bb_calls);
            }

            if ev_push > fold_payoff {
                p_br[i] = 1.0;
                u_btn += self.prior[i] * ev_push;
            } else {
                p_br[i] = 0.0;
                u_btn += self.prior[i] * fold_payoff;
            }
        }

        (p_br, u_btn)
    }

    /// Computes exact game exploitability and NashConv in bb/hand.
    pub fn exploitability(&self, p: &[f32; 169], q: &[f32; 169]) -> PushFoldExploitability {
        let (u_btn, u_bb) = self.evaluate_payoffs(p, q);
        let (q_br, u_bb_br) = self.best_response_bb(p);
        let (p_br, u_btn_br) = self.best_response_btn(q);

        let btn_exploitability = (u_bb_br - u_bb).max(0.0);
        let bb_exploitability = (u_btn_br - u_btn).max(0.0);
        let nash_conv = btn_exploitability + bb_exploitability;

        PushFoldExploitability {
            btn_exploitability,
            bb_exploitability,
            nash_conv,
            game_value: u_btn,
            q_best_response: q_br,
            p_best_response: p_br,
        }
    }

    /// Solves the exact Nash Equilibrium using CFR+ with linear averaging.
    pub fn solve_nash(&self, iterations: usize) -> (PushFoldStrategy, f32) {
        let mut r_btn_push = [0.0f32; 169];
        let mut r_btn_fold = [0.0f32; 169];
        let mut r_bb_call = [0.0f32; 169];
        let mut r_bb_fold = [0.0f32; 169];

        let mut avg_p = [0.0f32; 169];
        let mut avg_q = [0.0f32; 169];
        let mut weight_sum = 0.0f32;

        let mut p = [0.5f32; 169];
        let mut q = [0.5f32; 169];

        let pot_showdown = 2.0 * self.stack_bb;

        for t in 1..=iterations {
            let iter_weight = t as f32;

            for i in 0..NUM_CANONICAL_HANDS {
                let mut ev_push = 0.0f32;
                for j in 0..NUM_CANONICAL_HANDS {
                    let p_j = self.prob_opp_given_hero[i][j];
                    let payoff = (1.0 - q[j]) * self.big_blind_bb
                        + q[j] * (pot_showdown * self.equity[i][j] - self.stack_bb);
                    ev_push += p_j * payoff;
                }
                let ev_fold = -self.small_blind_bb;
                let v_node = p[i] * ev_push + (1.0 - p[i]) * ev_fold;

                r_btn_push[i] = (r_btn_push[i] + (ev_push - v_node)).max(0.0);
                r_btn_fold[i] = (r_btn_fold[i] + (ev_fold - v_node)).max(0.0);

                let sum = r_btn_push[i] + r_btn_fold[i];
                p[i] = if sum > 1e-9 { r_btn_push[i] / sum } else { 0.5 };
                avg_p[i] += iter_weight * p[i];
            }

            for j in 0..NUM_CANONICAL_HANDS {
                let mut ev_call_sum = 0.0f32;
                let mut weight_push = 0.0f32;
                for i in 0..NUM_CANONICAL_HANDS {
                    let w = self.prob_opp_given_hero[j][i] * p[i];
                    weight_push += w;
                    let sd_payoff = pot_showdown * (1.0 - self.equity[i][j]) - self.stack_bb;
                    ev_call_sum += w * sd_payoff;
                }
                let ev_fold = -self.big_blind_bb;

                if weight_push > 1e-9 {
                    let ev_call = ev_call_sum / weight_push;
                    let v_node = q[j] * ev_call + (1.0 - q[j]) * ev_fold;

                    r_bb_call[j] = (r_bb_call[j] + weight_push * (ev_call - v_node)).max(0.0);
                    r_bb_fold[j] = (r_bb_fold[j] + weight_push * (ev_fold - v_node)).max(0.0);

                    let sum = r_bb_call[j] + r_bb_fold[j];
                    q[j] = if sum > 1e-9 { r_bb_call[j] / sum } else { 0.5 };
                } else {
                    q[j] = 0.0;
                }
                avg_q[j] += iter_weight * q[j];
            }

            weight_sum += iter_weight;
        }

        let mut final_p = [0.0f32; 169];
        let mut final_q = [0.0f32; 169];
        for i in 0..169 {
            final_p[i] = avg_p[i] / weight_sum;
            final_q[i] = avg_q[i] / weight_sum;
        }

        let (u_btn, _) = self.evaluate_payoffs(&final_p, &final_q);
        (
            PushFoldStrategy {
                push_probs: final_p,
                call_probs: final_q,
            },
            u_btn,
        )
    }

    /// Computes overall weighted BTN Push and BB Call frequencies.
    pub fn overall_frequencies(&self, p: &[f32; 169], q: &[f32; 169]) -> (f32, f32) {
        let mut total_push = 0.0f32;
        let mut total_call = 0.0f32;
        for i in 0..169 {
            total_push += self.prior[i] * p[i];
            total_call += self.prior[i] * q[i];
        }
        (total_push * 100.0, total_call * 100.0)
    }

    /// Computes weighted L1 distance between strategy and Nash across the 169 grid.
    pub fn l1_distance(&self, p: &[f32; 169], q: &[f32; 169], nash: &PushFoldStrategy) -> (f32, f32) {
        let mut l1_p = 0.0f32;
        let mut l1_q = 0.0f32;
        for i in 0..169 {
            l1_p += self.prior[i] * (p[i] - nash.push_probs[i]).abs();
            l1_q += self.prior[i] * (q[i] - nash.call_probs[i]).abs();
        }
        (l1_p, l1_q)
    }
}

/// Strategy for 10 BB Push/Fold HUNL across 169 canonical hands.
#[derive(Clone, Copy, Debug)]
pub struct PushFoldStrategy {
    pub push_probs: [f32; 169],
    pub call_probs: [f32; 169],
}

/// Exploitability metrics for Push/Fold.
#[derive(Clone, Debug)]
pub struct PushFoldExploitability {
    pub btn_exploitability: f32,
    pub bb_exploitability: f32,
    pub nash_conv: f32,
    pub game_value: f32,
    pub q_best_response: [f32; 169],
    pub p_best_response: [f32; 169],
}

