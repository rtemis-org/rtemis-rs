//! Platform-independent core of rtemis.
//!
//! The crate depends only on the standard library. Each language binding
//! (R, Python, WebAssembly) converts its own data into the slices these
//! functions take and reports errors in its own terms.
//!
//! # Modules
//!
//! * [`ml`] — machine learning: [`ml::metrics`], model performance metrics.
//!
//! # Example
//!
//! ```
//! use rtemis::ml::metrics::classification::RankedScores;
//!
//! let reference = [0, 0, 1, 1];
//! let scores = [0.1, 0.4, 0.35, 0.8];
//! let ranked = RankedScores::new(&reference, &scores, 1).unwrap();
//! assert_eq!(ranked.roc_auc(), 0.75);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod ml;

/// Compiles and runs the examples in `README.md` as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
