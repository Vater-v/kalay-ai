use std::fmt;
use std::str::FromStr;

// ============================================================================
// 1. CARD (Ровно 1 байт в памяти)
// ============================================================================

/// Карта кодируется числом от 0 до 51:
/// rank = card >> 2  (0..12: 2, 3, 4, ..., T, J, Q, K, A)
/// suit = card & 3   (0..3:  Spades, Hearts, Diamonds, Clubs)
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Card(pub u8);

impl Card {
    // Простые числа для Cactus Kev / Perfect Hash эвалуатора:
    // Каждому рангу 2..A сопоставлено простое число.
    // Произведение 5 простых чисел уникально идентифицирует набор рангов!
    pub const PRIMES: [u32; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

    #[inline(always)]
    pub const fn new(rank: u8, suit: u8) -> Self {
        debug_assert!(rank < 13 && suit < 4);
        Self((rank << 2) | (suit & 3))
    }

    #[inline(always)]
    pub const fn rank(self) -> u8 {
        self.0 >> 2
    }

    #[inline(always)]
    pub const fn suit(self) -> u8 {
        self.0 & 3
    }

    /// Простое число ранга карты (для быстрого перемножения в эвалуаторе)
    #[inline(always)]
    pub const fn prime(self) -> u32 {
        Self::PRIMES[self.rank() as usize]
    }

    /// Бит ранга (1 << rank), где 2 = 1, A = 4096.
    /// Нужно для проверки флешей и стритов в эвалуаторе.
    #[inline(always)]
    pub const fn rank_bit(self) -> u16 {
        1 << self.rank()
    }

    /// Маска этой конкретной карты для битборда
    #[inline(always)]
    pub const fn mask(self) -> CardMask {
        CardMask(1u64 << self.0)
    }
}

/// Превращает 2 карманные карты в канонический индекс 0..168:
/// 0..12   - Карманные пары (22=0 .. AA=12)
/// 13..90  - Одномастные руки (32s=13 .. AKs=90)
/// 91..168 - Разномастные руки (32o=91 .. AKo=168)
#[inline(always)]
pub fn hand_169_index(c1: Card, c2: Card) -> usize {
    let r1 = c1.rank() as usize;
    let r2 = c2.rank() as usize;
    if r1 == r2 {
        r1
    } else {
        let (hi, lo) = if r1 > r2 { (r1, r2) } else { (r2, r1) };
        let pair_idx = (hi * (hi - 1)) / 2 + lo;
        if c1.suit() == c2.suit() {
            13 + pair_idx
        } else {
            91 + pair_idx
        }
    }
}

/// Возвращает репрезентативную пару карт для канонического индекса 0..168
pub fn hand_169_to_cards(idx: usize) -> [Card; 2] {
    if idx < 13 {
        let r = idx as u8;
        [Card::new(r, 0), Card::new(r, 1)] // Spades + Hearts
    } else {
        let is_suited = idx < 91;
        let p_idx = if is_suited { idx - 13 } else { idx - 91 };
        // Find hi and lo such that hi * (hi - 1) / 2 <= p_idx
        let mut hi = 1;
        while (hi + 1) * hi / 2 <= p_idx {
            hi += 1;
        }
        let lo = p_idx - (hi * (hi - 1)) / 2;
        let (s1, s2) = if is_suited { (0, 0) } else { (0, 1) };
        [Card::new(hi as u8, s1), Card::new(lo as u8, s2)]
    }
}

// ============================================================================
// 2. CARD MASK / BITBOARD (Множество карт в 64-битном регистре)
// ============================================================================

/// Битборд из 52 карт. Бит i = 1 означает, что Card(i) присутствует.
/// Идеально подходит для:
/// - Руки в NLH (2 бита), PLO4 (4 бита), PLO6 (6 бит)
/// - Борда (3, 4 или 5 бит)
/// - Мертвых карт и блокеров
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct CardMask(pub u64);

impl CardMask {
    pub const EMPTY: Self = Self(0);
    pub const FULL: Self = Self((1u64 << 52) - 1);

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Количество карт в наборе.
    /// Компилируется в аппаратную инструкцию процессора POPCNT (1 такт).
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    #[inline(always)]
    pub const fn contains(self, card: Card) -> bool {
        (self.0 & (1u64 << card.0)) != 0
    }

    #[inline(always)]
    pub fn insert(&mut self, card: Card) {
        self.0 |= 1u64 << card.0;
    }

    #[inline(always)]
    pub fn remove(&mut self, card: Card) {
        self.0 &= !(1u64 << card.0);
    }

    /// Проверка блокеров: пересекаются ли множества карт?
    /// Инструкция TEST в ассемблере (1 такт).
    #[inline(always)]
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    /// Достать и удалить младшую карту (для быстрого перебора без аллокаций)
    /// Использует аппаратную инструкцию TZCNT / BSF + BLSR.
    #[inline(always)]
    pub fn pop_card(&mut self) -> Option<Card> {
        if self.0 == 0 {
            None
        } else {
            let card_idx = self.0.trailing_zeros() as u8;
            self.0 &= self.0 - 1; // Сброс младшего бита за 1 такт
            Some(Card(card_idx))
        }
    }
}

// Побитовые операции для удобства: `mask1 | mask2`, `mask1 & !mask2`
impl std::ops::BitOr for CardMask {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitAnd for CardMask {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl std::ops::Not for CardMask {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0 & Self::FULL.0)
    }
}

// Итератор по картам: `for card in mask { ... }`
impl Iterator for CardMask {
    type Item = Card;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.pop_card()
    }
}

// ============================================================================
// 3. DECK (Колода без кучи)
// ============================================================================

#[derive(Clone)]
pub struct Deck {
    mask: CardMask,
}

impl Deck {
    pub fn new() -> Self {
        Self {
            mask: CardMask::FULL,
        }
    }
}

impl Default for Deck {
    fn default() -> Self {
        Self::new()
    }
}

impl Deck {
    /// Удалить мертвые карты (карты игроков, борд) за 1 операцию
    #[inline(always)]
    pub fn remove_dead(&mut self, dead: CardMask) {
        self.mask.0 &= !dead.0;
    }

    #[inline(always)]
    pub fn remaining(&self) -> CardMask {
        self.mask
    }

    #[inline(always)]
    pub fn count(&self) -> u32 {
        self.mask.count()
    }
}

// ============================================================================
// 4. PARSING & DISPLAY ("As", "Kd", "2c")
// ============================================================================

const RANK_CHARS: &[u8; 13] = b"23456789TJQKA";
const SUIT_CHARS: &[u8; 4] = b"shdc"; // Spades, Hearts, Diamonds, Clubs

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            RANK_CHARS[self.rank() as usize] as char,
            SUIT_CHARS[self.suit() as usize] as char
        )
    }
}

impl fmt::Debug for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParseCardError;

impl FromStr for Card {
    type Err = ParseCardError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return Err(ParseCardError);
        }

        let rank = match bytes[0] {
            b'2'..=b'9' => bytes[0] - b'2',
            b'T' | b't' => 8,
            b'J' | b'j' => 9,
            b'Q' | b'q' => 10,
            b'K' | b'k' => 11,
            b'A' | b'a' => 12,
            _ => return Err(ParseCardError),
        };

        let suit = match bytes[1] {
            b'S' | b's' => 0, // Spades
            b'H' | b'h' => 1, // Hearts
            b'D' | b'd' => 2, // Diamonds
            b'C' | b'c' => 3, // Clubs
            _ => return Err(ParseCardError),
        };

        Ok(Card::new(rank, suit))
    }
}
