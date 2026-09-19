use crate::card::Card;
use crate::tables::{M0, M1, M2, displacement_table, flush_table, no_flush_table};
use std::fmt;

// ============================================================================
// 1. HAND RANK & CATEGORY
// ============================================================================

/// Абсолютный ранг комбинации от 1 (Royal Flush) до 7462 (7-5-4-3-2 unsuited).
/// МЕНЬШЕЕ ЗНАЧЕНИЕ = СИЛЬНЕЕ РУКА!
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct HandRank(pub u16);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum HandCategory {
    StraightFlush = 1,
    FourOfAKind = 2,
    FullHouse = 3,
    Flush = 4,
    Straight = 5,
    ThreeOfAKind = 6,
    TwoPair = 7,
    OnePair = 8,
    HighCard = 9,
}

impl HandRank {
    pub const MAX: Self = Self(u16::MAX);
    pub const WORST: Self = Self(7462);
    pub const BEST: Self = Self(1);

    #[inline(always)]
    pub const fn new(rank: u16) -> Self {
        Self(rank)
    }

    #[inline(always)]
    pub fn category(self) -> HandCategory {
        match self.0 {
            1..=10 => HandCategory::StraightFlush,
            11..=166 => HandCategory::FourOfAKind,
            167..=322 => HandCategory::FullHouse,
            323..=1599 => HandCategory::Flush,
            1600..=1609 => HandCategory::Straight,
            1610..=2467 => HandCategory::ThreeOfAKind,
            2468..=3325 => HandCategory::TwoPair,
            3326..=6185 => HandCategory::OnePair,
            _ => HandCategory::HighCard,
        }
    }
}

impl fmt::Display for HandRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HandRank({}, {:?})", self.0, self.category())
    }
}

impl fmt::Debug for HandRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

// ============================================================================
// 2. ATOMIC 5-CARD EVALUATOR (CHD Perfect Hash, Zero Sorting, Zero Alloc)
// ============================================================================

/// Оценка заведомо не-флешевой 5-карточной руки по произведению простых чисел.
/// Полностью исключает необходимость сортировки (умножение коммутативно)!
#[inline(always)]
pub fn eval_5_non_flush(c1: Card, c2: Card, c3: Card, c4: Card, c5: Card) -> HandRank {
    let prod = c1.prime() * c2.prime() * c3.prime() * c4.prime() * c5.prime();
    let b = (prod.wrapping_mul(M0) >> 19) as usize;
    let disp = displacement_table()[b] as u32;
    let slot = (prod.wrapping_mul(M1).wrapping_add(disp.wrapping_mul(M2)) >> 19) as usize;
    unsafe { HandRank(*no_flush_table().get_unchecked(slot)) }
}

/// Полная оценка 5 карт за минимальное число тактов CPU
#[inline(always)]
pub fn eval_5(c1: Card, c2: Card, c3: Card, c4: Card, c5: Card) -> HandRank {
    let is_flush = (c1.suit() == c2.suit())
        & (c1.suit() == c3.suit())
        & (c1.suit() == c4.suit())
        & (c1.suit() == c5.suit());

    if is_flush {
        let mask = (c1.rank_bit() | c2.rank_bit() | c3.rank_bit() | c4.rank_bit() | c5.rank_bit())
            as usize;
        unsafe { HandRank(*flush_table().get_unchecked(mask)) }
    } else {
        eval_5_non_flush(c1, c2, c3, c4, c5)
    }
}

// ============================================================================
// 3. NLH EVALUATORS (FLOP, TURN, RIVER)
// ============================================================================

pub const C6_5: [[usize; 5]; 6] = [
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 5],
    [0, 1, 2, 4, 5],
    [0, 1, 3, 4, 5],
    [0, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
];

pub const C7_5: [[usize; 5]; 21] = [
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 5],
    [0, 1, 2, 3, 6],
    [0, 1, 2, 4, 5],
    [0, 1, 2, 4, 6],
    [0, 1, 2, 5, 6],
    [0, 1, 3, 4, 5],
    [0, 1, 3, 4, 6],
    [0, 1, 3, 5, 6],
    [0, 1, 4, 5, 6],
    [0, 2, 3, 4, 5],
    [0, 2, 3, 4, 6],
    [0, 2, 3, 5, 6],
    [0, 2, 4, 5, 6],
    [0, 3, 4, 5, 6],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 6],
    [1, 2, 3, 5, 6],
    [1, 2, 4, 5, 6],
    [1, 3, 4, 5, 6],
    [2, 3, 4, 5, 6],
];

/// NLH Flop: ровно 5 карт (2 карманки + 3 борд) -> 1 проверка!
#[inline(always)]
pub fn eval_nlh_flop(hole: [Card; 2], board: [Card; 3]) -> HandRank {
    eval_5(hole[0], hole[1], board[0], board[1], board[2])
}

/// NLH Turn: 6 карт (2 карманки + 4 борд) -> 6 проверок
#[inline(always)]
pub fn eval_nlh_turn(hole: [Card; 2], board: [Card; 4]) -> HandRank {
    let cards = [hole[0], hole[1], board[0], board[1], board[2], board[3]];
    let mut board_suits = [0u8; 4];
    for &b in &board {
        board_suits[b.suit() as usize] += 1;
    }
    let flush_possible =
        board_suits[0] >= 3 || board_suits[1] >= 3 || board_suits[2] >= 3 || board_suits[3] >= 3;

    let mut best = HandRank::MAX;
    if !flush_possible {
        for &idx in &C6_5 {
            let r = eval_5_non_flush(
                cards[idx[0]],
                cards[idx[1]],
                cards[idx[2]],
                cards[idx[3]],
                cards[idx[4]],
            );
            if r.0 < best.0 {
                best = r;
            }
        }
    } else {
        for &idx in &C6_5 {
            let r = eval_5(
                cards[idx[0]],
                cards[idx[1]],
                cards[idx[2]],
                cards[idx[3]],
                cards[idx[4]],
            );
            if r.0 < best.0 {
                best = r;
                if best.0 == 1 {
                    return best;
                }
            }
        }
    }
    best
}

/// NLH River: 7 карт (2 карманки + 5 борд) -> 21 проверка
#[inline(always)]
pub fn eval_nlh_river(hole: [Card; 2], board: [Card; 5]) -> HandRank {
    let cards = [
        hole[0], hole[1], board[0], board[1], board[2], board[3], board[4],
    ];
    let mut board_suits = [0u8; 4];
    for &b in &board {
        board_suits[b.suit() as usize] += 1;
    }
    let flush_possible =
        board_suits[0] >= 3 || board_suits[1] >= 3 || board_suits[2] >= 3 || board_suits[3] >= 3;

    let mut best = HandRank::MAX;
    if !flush_possible {
        for &idx in &C7_5 {
            let r = eval_5_non_flush(
                cards[idx[0]],
                cards[idx[1]],
                cards[idx[2]],
                cards[idx[3]],
                cards[idx[4]],
            );
            if r.0 < best.0 {
                best = r;
            }
        }
    } else {
        for &idx in &C7_5 {
            let r = eval_5(
                cards[idx[0]],
                cards[idx[1]],
                cards[idx[2]],
                cards[idx[3]],
                cards[idx[4]],
            );
            if r.0 < best.0 {
                best = r;
                if best.0 == 1 {
                    return best;
                }
            }
        }
    }
    best
}

// ============================================================================
// 4. COMBINATORIAL CONSTANTS FOR OMAHA
// ============================================================================

pub const PAIRS_4: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

pub const PAIRS_5: [[usize; 2]; 10] = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 2],
    [1, 3],
    [1, 4],
    [2, 3],
    [2, 4],
    [3, 4],
];

pub const PAIRS_6: [[usize; 2]; 15] = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [0, 5],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [2, 3],
    [2, 4],
    [2, 5],
    [3, 4],
    [3, 5],
    [4, 5],
];

pub const TRIPLETS_4: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

pub const TRIPLETS_5: [[usize; 3]; 10] = [
    [0, 1, 2],
    [0, 1, 3],
    [0, 1, 4],
    [0, 2, 3],
    [0, 2, 4],
    [0, 3, 4],
    [1, 2, 3],
    [1, 2, 4],
    [1, 3, 4],
    [2, 3, 4],
];

// ============================================================================
// 5. PLO4 EVALUATORS
// ============================================================================

/// PLO4 Flop: 6 пар руки x 1 тройка борда = 6 проверок
#[inline(always)]
pub fn eval_plo4_flop(hole: [Card; 4], board: [Card; 3]) -> HandRank {
    let mut best = HandRank::MAX;
    for &h in &PAIRS_4 {
        let r = eval_5(hole[h[0]], hole[h[1]], board[0], board[1], board[2]);
        if r.0 < best.0 {
            best = r;
        }
    }
    best
}

/// PLO4 Turn: 6 пар руки x 4 тройки борда = 24 проверки
#[inline(always)]
pub fn eval_plo4_turn(hole: [Card; 4], board: [Card; 4]) -> HandRank {
    let mut best = HandRank::MAX;
    for &h in &PAIRS_4 {
        let (c1, c2) = (hole[h[0]], hole[h[1]]);
        for &b in &TRIPLETS_4 {
            let r = eval_5(c1, c2, board[b[0]], board[b[1]], board[b[2]]);
            if r.0 < best.0 {
                best = r;
                if best.0 == 1 {
                    return best;
                }
            }
        }
    }
    best
}

/// PLO4 River: 6 пар руки x 10 троек борда = 60 проверок
#[inline(always)]
pub fn eval_plo4_river(hole: [Card; 4], board: [Card; 5]) -> HandRank {
    let mut board_suits = [0u8; 4];
    for &b in &board {
        board_suits[b.suit() as usize] += 1;
    }
    let flush_possible =
        board_suits[0] >= 3 || board_suits[1] >= 3 || board_suits[2] >= 3 || board_suits[3] >= 3;

    let mut best = HandRank::MAX;
    if !flush_possible {
        for &h in &PAIRS_4 {
            let (c1, c2) = (hole[h[0]], hole[h[1]]);
            for &b in &TRIPLETS_5 {
                let r = eval_5_non_flush(c1, c2, board[b[0]], board[b[1]], board[b[2]]);
                if r.0 < best.0 {
                    best = r;
                }
            }
        }
    } else {
        for &h in &PAIRS_4 {
            let (c1, c2) = (hole[h[0]], hole[h[1]]);
            for &b in &TRIPLETS_5 {
                let r = eval_5(c1, c2, board[b[0]], board[b[1]], board[b[2]]);
                if r.0 < best.0 {
                    best = r;
                    if best.0 == 1 {
                        return best;
                    }
                }
            }
        }
    }
    best
}

// ============================================================================
// 6. PLO5 EVALUATORS
// ============================================================================

/// PLO5 Flop: 10 пар руки x 1 тройка борда = 10 проверок
#[inline(always)]
pub fn eval_plo5_flop(hole: [Card; 5], board: [Card; 3]) -> HandRank {
    let mut best = HandRank::MAX;
    for &h in &PAIRS_5 {
        let r = eval_5(hole[h[0]], hole[h[1]], board[0], board[1], board[2]);
        if r.0 < best.0 {
            best = r;
        }
    }
    best
}

/// PLO5 Turn: 10 пар руки x 4 тройки борда = 40 проверок
#[inline(always)]
pub fn eval_plo5_turn(hole: [Card; 5], board: [Card; 4]) -> HandRank {
    let mut best = HandRank::MAX;
    for &h in &PAIRS_5 {
        let (c1, c2) = (hole[h[0]], hole[h[1]]);
        for &b in &TRIPLETS_4 {
            let r = eval_5(c1, c2, board[b[0]], board[b[1]], board[b[2]]);
            if r.0 < best.0 {
                best = r;
                if best.0 == 1 {
                    return best;
                }
            }
        }
    }
    best
}

/// PLO5 River: 10 пар руки x 10 троек борда = 100 проверок
#[inline(always)]
pub fn eval_plo5_river(hole: [Card; 5], board: [Card; 5]) -> HandRank {
    let mut board_suits = [0u8; 4];
    for &b in &board {
        board_suits[b.suit() as usize] += 1;
    }
    let flush_possible =
        board_suits[0] >= 3 || board_suits[1] >= 3 || board_suits[2] >= 3 || board_suits[3] >= 3;

    let mut best = HandRank::MAX;
    if !flush_possible {
        for &h in &PAIRS_5 {
            let (c1, c2) = (hole[h[0]], hole[h[1]]);
            for &b in &TRIPLETS_5 {
                let r = eval_5_non_flush(c1, c2, board[b[0]], board[b[1]], board[b[2]]);
                if r.0 < best.0 {
                    best = r;
                }
            }
        }
    } else {
        for &h in &PAIRS_5 {
            let (c1, c2) = (hole[h[0]], hole[h[1]]);
            for &b in &TRIPLETS_5 {
                let r = eval_5(c1, c2, board[b[0]], board[b[1]], board[b[2]]);
                if r.0 < best.0 {
                    best = r;
                    if best.0 == 1 {
                        return best;
                    }
                }
            }
        }
    }
    best
}

// ============================================================================
// 7. PLO6 EVALUATORS
// ============================================================================

/// PLO6 Flop: 15 пар руки x 1 тройка борда = 15 проверок
#[inline(always)]
pub fn eval_plo6_flop(hole: [Card; 6], board: [Card; 3]) -> HandRank {
    let mut best = HandRank::MAX;
    for &h in &PAIRS_6 {
        let r = eval_5(hole[h[0]], hole[h[1]], board[0], board[1], board[2]);
        if r.0 < best.0 {
            best = r;
        }
    }
    best
}

/// PLO6 Turn: 15 пар руки x 4 тройки борда = 60 проверок
#[inline(always)]
pub fn eval_plo6_turn(hole: [Card; 6], board: [Card; 4]) -> HandRank {
    let mut best = HandRank::MAX;
    for &h in &PAIRS_6 {
        let (c1, c2) = (hole[h[0]], hole[h[1]]);
        for &b in &TRIPLETS_4 {
            let r = eval_5(c1, c2, board[b[0]], board[b[1]], board[b[2]]);
            if r.0 < best.0 {
                best = r;
                if best.0 == 1 {
                    return best;
                }
            }
        }
    }
    best
}

/// PLO6 River: 15 пар руки x 10 троек борда = 150 проверок
#[inline(always)]
pub fn eval_plo6_river(hole: [Card; 6], board: [Card; 5]) -> HandRank {
    let mut board_suits = [0u8; 4];
    for &b in &board {
        board_suits[b.suit() as usize] += 1;
    }
    let flush_possible =
        board_suits[0] >= 3 || board_suits[1] >= 3 || board_suits[2] >= 3 || board_suits[3] >= 3;

    let mut best = HandRank::MAX;
    if !flush_possible {
        for &h in &PAIRS_6 {
            let (c1, c2) = (hole[h[0]], hole[h[1]]);
            for &b in &TRIPLETS_5 {
                let r = eval_5_non_flush(c1, c2, board[b[0]], board[b[1]], board[b[2]]);
                if r.0 < best.0 {
                    best = r;
                }
            }
        }
    } else {
        for &h in &PAIRS_6 {
            let (c1, c2) = (hole[h[0]], hole[h[1]]);
            for &b in &TRIPLETS_5 {
                let r = eval_5(c1, c2, board[b[0]], board[b[1]], board[b[2]]);
                if r.0 < best.0 {
                    best = r;
                    if best.0 == 1 {
                        return best;
                    }
                }
            }
        }
    }
    best
}

// ============================================================================
// 8. ZERO-ALLOCATION SHOWDOWN & WINNERS
// ============================================================================

/// Возвращает битовую маску победителей (бит i = 1, если игрок i выиграл).
/// Ровно 0 аллокаций в куче! Возвращается в одном регистре CPU.
#[inline(always)]
pub fn who_won_mask(player_ranks: &[HandRank]) -> u16 {
    if player_ranks.is_empty() {
        return 0;
    }
    let mut best = player_ranks[0];
    let mut mask = 1u16;

    for (i, &rank) in player_ranks.iter().enumerate().skip(1) {
        if rank < best {
            best = rank;
            mask = 1u16 << i;
        } else if rank == best {
            mask |= 1u16 << i;
        }
    }
    mask
}

/// Удобная обертка для UI/логов, возвращающая Vec индексов победителей
pub fn who_won(player_ranks: &[HandRank]) -> Vec<usize> {
    let mask = who_won_mask(player_ranks);
    let mut winners = Vec::new();
    let mut m = mask;
    while m != 0 {
        let idx = m.trailing_zeros() as usize;
        winners.push(idx);
        m &= m - 1;
    }
    winners
}
