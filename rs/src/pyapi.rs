//! pyo3 bindings: evaluator, equity table generator, exact push-fold solver.
use crate::card::{hand_169_index, Card};
use crate::equitygen;
use crate::eval::eval_nlh_river;
use crate::pushfold::PushFoldGame;
use pyo3::prelude::*;

#[pyfunction]
fn hand_169(c1: u8, c2: u8) -> usize {
    hand_169_index(Card(c1), Card(c2))
}

#[pyfunction]
fn hand_169_name_py(idx: usize) -> PyResult<String> {
    Ok(crate::pushfold::hand_name(idx).0.to_string())
}

#[pyfunction]
fn hand_169_combos_py(idx: usize) -> usize {
    crate::pushfold::hand_combos(idx)
}

/// Evaluate 7 cards (hole 2 + board 5). Lower rank value = stronger hand.
#[pyfunction]
fn eval_river7(cards: Vec<u8>) -> PyResult<u16> {
    if cards.len() != 7 {
        return Err(pyo3::exceptions::PyValueError::new_err("need 7 cards"));
    }
    let mut cs = [Card(0); 7];
    for (k, c) in cards.iter().enumerate() {
        if *c > 51 {
            return Err(pyo3::exceptions::PyValueError::new_err("card id > 51"));
        }
        cs[k] = Card(*c);
    }
    Ok(eval_nlh_river([cs[0], cs[1]], [cs[2], cs[3], cs[4], cs[5], cs[6]]).0)
}

/// Equity of class i vs class j from a saved table.
#[pyfunction]
fn equity_lookup(table_path: &str, i: usize, j: usize) -> PyResult<f64> {
    let table = equitygen::load(table_path)?;
    Ok(table[i][j] as f64)
}

/// Generate the 169x169 all-in equity table (Monte-Carlo, fixed seed).
#[pyfunction]
fn generate_equity_table(path: &str, sims: u64, seed: u64) -> PyResult<()> {
    let table = equitygen::generate(sims, seed);
    equitygen::save(&table, path)?;
    Ok(())
}

#[pyclass]
struct PushFoldSolver {
    game: PushFoldGame,
}

#[pymethods]
impl PushFoldSolver {
    #[new]
    fn new(stack_bb: f32, small_blind_bb: f32, big_blind_bb: f32, equity_path: &str) -> PyResult<Self> {
        let equity = equitygen::load(equity_path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("equity table: {e}")))?;
        Ok(Self {
            game: PushFoldGame::with_equity(stack_bb, small_blind_bb, big_blind_bb, equity),
        })
    }

    /// CFR+ solve. Returns (push_probs[169], call_probs[169], btn_value_bb).
    fn nash(&self, iterations: usize) -> (Vec<f32>, Vec<f32>, f32) {
        let (s, v) = self.game.solve_nash(iterations);
        (s.push_probs.to_vec(), s.call_probs.to_vec(), v)
    }

    /// Exact NashConv in bb/hand of a strategy pair.
    fn exploitability(&self, p: Vec<f32>, q: Vec<f32>) -> PyResult<f32> {
        let (p, q) = (to169(p)?, to169(q)?);
        Ok(self.game.exploitability(&p, &q).nash_conv)
    }

    fn payoffs(&self, p: Vec<f32>, q: Vec<f32>) -> PyResult<(f32, f32)> {
        let (p, q) = (to169(p)?, to169(q)?);
        Ok(self.game.evaluate_payoffs(&p, &q))
    }

    fn equity(&self, i: usize, j: usize) -> f32 {
        self.game.equity[i][j]
    }

    fn prior(&self, i: usize) -> f32 {
        self.game.prior[i]
    }
}

fn to169(v: Vec<f32>) -> PyResult<[f32; 169]> {
    if v.len() != 169 {
        return Err(pyo3::exceptions::PyValueError::new_err("need 169 probabilities"));
    }
    let mut a = [0.0f32; 169];
    a.copy_from_slice(&v);
    Ok(a)
}

#[pymodule]
fn kalay_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hand_169, m)?)?;
    m.add_function(wrap_pyfunction!(hand_169_name_py, m)?)?;
    m.add_function(wrap_pyfunction!(hand_169_combos_py, m)?)?;
    m.add_function(wrap_pyfunction!(eval_river7, m)?)?;
    m.add_function(wrap_pyfunction!(equity_lookup, m)?)?;
    m.add_function(wrap_pyfunction!(generate_equity_table, m)?)?;
    m.add_class::<PushFoldSolver>()?;
    Ok(())
}
