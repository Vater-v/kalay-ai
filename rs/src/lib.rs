//! kalay_rs: vendored native core for kalay-ai.
//!
//! Hand evaluator (perfect-hash 7-card, lookup.bin) and the exact push-fold
//! engine originate from the legacy poker-ai project; the equity table is
//! regenerated here with a fixed seed (the legacy table was Monte-Carlo of
//! unknown seed/precision) and validated from Python tier A against an
//! independent exact enumeration.
pub mod card;
pub mod equitygen;
pub mod eval;
pub mod pyapi;
pub mod pushfold;
pub mod tables;
