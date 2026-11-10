//! Rank-based performance metrics for classifiers: the area under the ROC
//! curve (AUC) and average precision (AP), for binary outcomes and one class
//! against the rest.
//!
//! Each metric depends on the scores only through their order. Scores are
//! mapped to order-preserving `u64` keys, split by class, and sorted once in a
//! [`RankedScores`]; each metric is then a single linear pass over the sorted
//! keys. Cases with equal scores form one threshold, so ties are handled the
//! same way regardless of input order.

use std::fmt;

/// Error returned when the input of a metric is inconsistent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetricError {
    /// `scores` has `found` elements where `expected` are required: one per
    /// reference label, or one per reference label and class for a one-vs-rest
    /// metric.
    LengthMismatch {
        /// Required number of scores.
        expected: usize,
        /// Number of scores supplied.
        found: usize,
    },
    /// `scores[index]` is NaN.
    NanScore {
        /// Zero-based position in `scores`.
        index: usize,
    },
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricError::LengthMismatch { expected, found } => {
                write!(f, "expected {expected} scores, found {found}")
            }
            MetricError::NanScore { index } => write!(f, "score at index {index} is NaN"),
        }
    }
}

impl std::error::Error for MetricError {}

/// Scores of a binary problem, split by class and sorted, from which every
/// rank-based metric is computed in linear time.
///
/// A case is positive when its reference label equals `positive` and negative
/// otherwise. The buffer holds one `u64` per case: positive keys first, then
/// negative keys, each part in ascending order. [`RankedScores::rank`] reuses
/// the buffer, so one allocation serves any number of problems of the same
/// size.
///
/// # Examples
///
/// ```
/// use rtemis::ml::metrics::classification::RankedScores;
///
/// let reference = [0, 0, 1, 1];
/// let scores = [0.1, 0.4, 0.35, 0.8];
/// let ranked = RankedScores::new(&reference, &scores, 1).unwrap();
/// assert_eq!(ranked.roc_auc(), 0.75);
/// assert!((ranked.average_precision() - 5.0 / 6.0).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Default)]
pub struct RankedScores {
    keys: Vec<u64>,
    n_pos: usize,
}

impl RankedScores {
    /// Ranks `scores` for the binary problem `reference == positive`.
    ///
    /// # Errors
    ///
    /// * [`MetricError::LengthMismatch`] if `scores` and `reference` differ in
    ///   length.
    /// * [`MetricError::NanScore`] if a score is NaN.
    pub fn new<T: Copy + PartialEq>(
        reference: &[T],
        scores: &[f64],
        positive: T,
    ) -> Result<Self, MetricError> {
        let mut ranked = Self::default();
        ranked.rank(reference, scores, positive)?;
        Ok(ranked)
    }

    /// Replaces the ranked problem with `reference == positive` and `scores`,
    /// reusing the existing allocation.
    ///
    /// # Errors
    ///
    /// As [`RankedScores::new`]. On error the previous contents are lost.
    pub fn rank<T: Copy + PartialEq>(
        &mut self,
        reference: &[T],
        scores: &[f64],
        positive: T,
    ) -> Result<(), MetricError> {
        if scores.len() != reference.len() {
            return Err(MetricError::LengthMismatch {
                expected: reference.len(),
                found: scores.len(),
            });
        }
        if let Some(index) = scores.iter().position(|s| s.is_nan()) {
            return Err(MetricError::NanScore { index });
        }

        let n_pos = reference.iter().filter(|&&label| label == positive).count();
        self.keys.resize(reference.len(), 0);
        self.n_pos = n_pos;

        let (pos, neg) = self.keys.split_at_mut(n_pos);
        let (mut p, mut q) = (0, 0);
        for (&label, &score) in reference.iter().zip(scores) {
            if label == positive {
                pos[p] = order_key(score);
                p += 1;
            } else {
                neg[q] = order_key(score);
                q += 1;
            }
        }
        pos.sort_unstable();
        neg.sort_unstable();
        Ok(())
    }

    /// Number of positive cases.
    pub fn n_positive(&self) -> usize {
        self.n_pos
    }

    /// Number of negative cases.
    pub fn n_negative(&self) -> usize {
        self.keys.len() - self.n_pos
    }

    /// Area under the ROC curve.
    ///
    /// The AUC is the probability that a randomly chosen positive case scores
    /// higher than a randomly chosen negative case, with ties counted as one
    /// half: the Mann–Whitney U statistic divided by `n_pos * n_neg`. It
    /// equals the trapezoidal area under the empirical ROC curve. The pair
    /// count accumulates as an exact integer.
    ///
    /// Returns `f64::NAN` when either class is empty.
    pub fn roc_auc(&self) -> f64 {
        let (n_pos, n_neg) = (self.n_positive(), self.n_negative());
        if n_pos == 0 || n_neg == 0 {
            return f64::NAN;
        }
        let mut twice_u: u128 = 0;
        let mut neg_above = 0usize;
        self.for_each_threshold(|p, q| {
            let neg_below = n_neg - neg_above - q;
            twice_u += p as u128 * (2 * neg_below + q) as u128;
            neg_above += q;
        });
        twice_u as f64 / (2.0 * n_pos as f64 * n_neg as f64)
    }

    /// Average precision: the area under the stepwise precision-recall
    /// curve.
    ///
    /// Each distinct score is one threshold, and the cases tied at it enter
    /// together. With recall `R_k` and precision `P_k` at the `k`-th highest
    /// threshold, AP is `sum_k (R_k - R_{k-1}) * P_k`. Thresholds that add no
    /// positive case add no area. The value is 1 when there are no negative
    /// cases.
    ///
    /// Returns `f64::NAN` when there are no positive cases.
    pub fn average_precision(&self) -> f64 {
        let n_pos = self.n_positive();
        if n_pos == 0 {
            return f64::NAN;
        }
        let (mut tp, mut fp) = (0usize, 0usize);
        let mut sum = 0.0;
        self.for_each_threshold(|p, q| {
            tp += p;
            fp += q;
            if p > 0 {
                sum += p as f64 * tp as f64 / (tp + fp) as f64;
            }
        });
        sum / n_pos as f64
    }

    /// Calls `f(p, q)` once per distinct score, from highest to lowest, with
    /// the number of positive (`p`) and negative (`q`) cases at that score.
    #[inline]
    fn for_each_threshold(&self, mut f: impl FnMut(usize, usize)) {
        let (pos, neg) = self.keys.split_at(self.n_pos);
        let (mut i, mut j) = (pos.len(), neg.len());
        while i > 0 || j > 0 {
            let key = match (i, j) {
                (0, _) => neg[j - 1],
                (_, 0) => pos[i - 1],
                _ => pos[i - 1].max(neg[j - 1]),
            };
            let (i_end, j_end) = (i, j);
            while i > 0 && pos[i - 1] == key {
                i -= 1;
            }
            while j > 0 && neg[j - 1] == key {
                j -= 1;
            }
            f(i_end - i, j_end - j);
        }
    }
}

/// Area under the ROC curve of the binary problem `reference == positive`.
///
/// See [`RankedScores::roc_auc`] for the definition. Time is `O(n log n)`;
/// extra memory is one `u64` per case.
///
/// # Errors
///
/// As [`RankedScores::new`].
///
/// # Examples
///
/// ```
/// use rtemis::ml::metrics::classification::roc_auc;
///
/// assert_eq!(roc_auc(&[0, 0, 1, 1], &[0.1, 0.4, 0.35, 0.8], 1), Ok(0.75));
/// // Reference labels of any comparable type; a single class leaves the AUC undefined.
/// assert!(roc_auc(&[true, true], &[0.2, 0.9], true).unwrap().is_nan());
/// ```
pub fn roc_auc<T: Copy + PartialEq>(
    reference: &[T],
    scores: &[f64],
    positive: T,
) -> Result<f64, MetricError> {
    RankedScores::new(reference, scores, positive).map(|r| r.roc_auc())
}

/// Average precision of the binary problem `reference == positive`.
///
/// See [`RankedScores::average_precision`] for the definition. Time is
/// `O(n log n)`; extra memory is one `u64` per case.
///
/// # Errors
///
/// As [`RankedScores::new`].
///
/// # Examples
///
/// ```
/// use rtemis::ml::metrics::classification::average_precision;
///
/// // Thresholds 0.8, 0.4, 0.35: precision 1, 1/2, 2/3 at recall 1/2, 1/2, 1.
/// let ap = average_precision(&[0, 0, 1, 1], &[0.1, 0.4, 0.35, 0.8], 1).unwrap();
/// assert!((ap - (0.5 * 1.0 + 0.5 * 2.0 / 3.0)).abs() < 1e-15);
/// ```
pub fn average_precision<T: Copy + PartialEq>(
    reference: &[T],
    scores: &[f64],
    positive: T,
) -> Result<f64, MetricError> {
    RankedScores::new(reference, scores, positive).map(|r| r.average_precision())
}

/// One-vs-rest AUC for each class.
///
/// `scores` is an `n`-by-`k` matrix in column-major order, where `n` is the
/// length of `reference` and `k` the number of classes: column `c` holds the
/// score of `classes[c]`. Element `c` of the result is the AUC of the binary problem
/// `reference == classes[c]` with scores from column `c`, or NaN when that class
/// is absent or is the only class present. A reference label outside `classes` is
/// negative in every problem.
///
/// One buffer of `n` keys serves every class. Time is `O(k n log n)`.
///
/// # Errors
///
/// * [`MetricError::LengthMismatch`] if `scores.len() != reference.len() *
///   classes.len()`.
/// * [`MetricError::NanScore`] if a score is NaN; `index` is its position in
///   `scores`.
///
/// # Examples
///
/// ```
/// use rtemis::ml::metrics::classification::roc_auc_ovr;
///
/// let reference = ["a", "b", "c", "a"];
/// #[rustfmt::skip]
/// let scores = [
///     0.8, 0.1, 0.1, 0.6, // a
///     0.1, 0.7, 0.2, 0.2, // b
///     0.1, 0.2, 0.7, 0.2, // c
/// ];
/// let auc = roc_auc_ovr(&reference, &scores, &["a", "b", "c"]).unwrap();
/// assert_eq!(auc, vec![1.0, 1.0, 1.0]);
/// ```
pub fn roc_auc_ovr<T: Copy + PartialEq>(
    reference: &[T],
    scores: &[f64],
    classes: &[T],
) -> Result<Vec<f64>, MetricError> {
    one_vs_rest(reference, scores, classes, RankedScores::roc_auc)
}

/// One-vs-rest average precision for each class.
///
/// Arguments, layout, and errors as [`roc_auc_ovr`]. Element `c` of the
/// result is NaN when `classes[c]` is absent.
///
/// # Examples
///
/// ```
/// use rtemis::ml::metrics::classification::average_precision_ovr;
///
/// let ap = average_precision_ovr(&[0, 1, 1], &[0.9, 0.2, 0.1, 0.1, 0.8, 0.9], &[0, 1]).unwrap();
/// assert_eq!(ap, vec![1.0, 1.0]);
/// ```
pub fn average_precision_ovr<T: Copy + PartialEq>(
    reference: &[T],
    scores: &[f64],
    classes: &[T],
) -> Result<Vec<f64>, MetricError> {
    one_vs_rest(reference, scores, classes, RankedScores::average_precision)
}

/// Applies `metric` to each one-vs-rest problem, reusing one [`RankedScores`].
fn one_vs_rest<T: Copy + PartialEq>(
    reference: &[T],
    scores: &[f64],
    classes: &[T],
    metric: fn(&RankedScores) -> f64,
) -> Result<Vec<f64>, MetricError> {
    let n = reference.len();
    if scores.len() != n * classes.len() {
        return Err(MetricError::LengthMismatch {
            expected: n * classes.len(),
            found: scores.len(),
        });
    }
    let mut ranked = RankedScores::default();
    let mut out = Vec::with_capacity(classes.len());
    for (c, &class) in classes.iter().enumerate() {
        let column = &scores[c * n..(c + 1) * n];
        ranked.rank(reference, column, class).map_err(|e| match e {
            MetricError::NanScore { index } => MetricError::NanScore {
                index: c * n + index,
            },
            other => other,
        })?;
        out.push(metric(&ranked));
    }
    Ok(out)
}

/// Maps a non-NaN `f64` to a `u64` whose unsigned order matches the numeric
/// order of the input.
///
/// Adding `0.0` maps `-0.0` to `0.0`, so the two receive the same key and
/// count as a tie. Negative values have every bit flipped, which reverses
/// their order; non-negative values have the sign bit set, which places them
/// above all negative values.
#[inline]
fn order_key(x: f64) -> u64 {
    let bits = (x + 0.0).to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | (1 << 63)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64 generator: reproducible test data with no dependency.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Integer in `lo..hi`. The modulo bias is below 2^-50 for these ranges.
        fn range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next_u64() % (hi - lo) as u64) as usize
        }
    }

    /// AUC by comparing every positive-negative pair.
    fn pairwise_auc(reference: &[i32], scores: &[f64]) -> f64 {
        let (mut concordant, mut pairs) = (0.0, 0.0);
        for (i, &li) in reference.iter().enumerate() {
            for (j, &lj) in reference.iter().enumerate() {
                if li == 1 && lj != 1 {
                    pairs += 1.0;
                    if scores[i] > scores[j] {
                        concordant += 1.0;
                    } else if scores[i] == scores[j] {
                        concordant += 0.5;
                    }
                }
            }
        }
        concordant / pairs
    }

    /// AP by evaluating precision and recall at every distinct score.
    fn threshold_ap(reference: &[i32], scores: &[f64]) -> f64 {
        let n_pos = reference.iter().filter(|&&l| l == 1).count() as f64;
        let mut thresholds = scores.to_vec();
        thresholds.sort_by(|a, b| b.total_cmp(a));
        thresholds.dedup();
        let (mut ap, mut prev_recall) = (0.0, 0.0);
        for t in thresholds {
            let selected = scores.iter().filter(|&&s| s >= t).count() as f64;
            let tp = reference
                .iter()
                .zip(scores)
                .filter(|&(&l, &s)| l == 1 && s >= t)
                .count() as f64;
            let recall = tp / n_pos;
            ap += (recall - prev_recall) * tp / selected;
            prev_recall = recall;
        }
        ap
    }

    fn random_problem(rng: &mut SplitMix64, n_classes: usize) -> (Vec<i32>, Vec<f64>) {
        let n = rng.range(2, 200);
        let reference: Vec<i32> = (0..n).map(|_| rng.range(0, n_classes) as i32).collect();
        // Few distinct values produce many ties.
        let levels = rng.range(1, 20);
        let scores = (0..n * n_classes)
            .map(|_| rng.range(0, levels) as f64 / levels as f64)
            .collect();
        (reference, scores)
    }

    fn assert_close(observed: f64, expected: f64) {
        if expected.is_nan() {
            assert!(observed.is_nan(), "{observed} vs NaN");
        } else {
            assert!(
                (observed - expected).abs() < 1e-12,
                "{observed} vs {expected}"
            );
        }
    }

    #[test]
    fn auc_known_values() {
        assert_eq!(roc_auc(&[0, 0, 1, 1], &[0.1, 0.2, 0.8, 0.9], 1), Ok(1.0));
        assert_eq!(roc_auc(&[1, 1, 0, 0], &[0.1, 0.2, 0.8, 0.9], 1), Ok(0.0));
        assert_eq!(roc_auc(&[0, 1, 0, 1], &[0.5; 4], 1), Ok(0.5));
        // Pairs: one tie and three wins.
        assert_eq!(roc_auc(&[1, 0, 1, 0], &[0.5, 0.5, 0.9, 0.1], 1), Ok(0.875));
    }

    #[test]
    fn ap_known_values() {
        assert_eq!(
            average_precision(&[0, 0, 1, 1], &[0.1, 0.2, 0.8, 0.9], 1),
            Ok(1.0)
        );
        assert_eq!(average_precision(&[1, 1], &[0.1, 0.2], 1), Ok(1.0));
        // One threshold holding every case: precision is the prevalence.
        assert_eq!(average_precision(&[0, 1, 0, 1], &[0.5; 4], 1), Ok(0.5));
        assert_eq!(
            average_precision(&[1, 0, 0, 0], &[0.1, 0.2, 0.3, 0.4], 1),
            Ok(0.25)
        );
    }

    #[test]
    fn undefined_metrics_are_nan() {
        assert!(roc_auc(&[1, 1], &[0.1, 0.2], 1).unwrap().is_nan());
        assert!(roc_auc::<i32>(&[], &[], 1).unwrap().is_nan());
        assert!(average_precision(&[0, 0], &[0.1, 0.2], 1).unwrap().is_nan());
    }

    #[test]
    fn signed_zero_ties_and_infinities_order() {
        assert_eq!(roc_auc(&[1, 0], &[0.0, -0.0], 1), Ok(0.5));
        let scores = [f64::NEG_INFINITY, -1e300, 1e300, f64::INFINITY];
        assert_eq!(roc_auc(&[0, 1, 0, 1], &scores, 1), Ok(0.75));
    }

    #[test]
    fn invalid_input_is_an_error() {
        let length = MetricError::LengthMismatch {
            expected: 2,
            found: 1,
        };
        assert_eq!(roc_auc(&[0, 1], &[0.1], 1), Err(length));
        assert_eq!(
            average_precision(&[0, 1], &[0.1, f64::NAN], 1),
            Err(MetricError::NanScore { index: 1 })
        );
        assert_eq!(
            roc_auc_ovr(&[0, 1], &[0.1, 0.2, 0.3], &[0, 1]),
            Err(MetricError::LengthMismatch {
                expected: 4,
                found: 3
            })
        );
        assert_eq!(
            roc_auc_ovr(&[0, 1], &[0.1, 0.2, 0.3, f64::NAN], &[0, 1]),
            Err(MetricError::NanScore { index: 3 })
        );
    }

    #[test]
    fn order_key_preserves_order() {
        let values = [
            f64::NEG_INFINITY,
            -1e300,
            -1.0,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            1.0,
            1e300,
            f64::INFINITY,
        ];
        for pair in values.windows(2) {
            let (a, b) = (order_key(pair[0]), order_key(pair[1]));
            if pair[0] == pair[1] {
                assert_eq!(a, b, "{} vs {}", pair[0], pair[1]);
            } else {
                assert!(a < b, "{} vs {}", pair[0], pair[1]);
            }
        }
    }

    #[test]
    fn binary_metrics_match_references() {
        let mut rng = SplitMix64(2026);
        for _ in 0..300 {
            let (reference, scores) = random_problem(&mut rng, 2);
            let scores = &scores[..reference.len()];
            let ranked = RankedScores::new(&reference, scores, 1).unwrap();
            assert_close(ranked.roc_auc(), pairwise_auc(&reference, scores));
            let expected_ap = if ranked.n_positive() == 0 {
                f64::NAN
            } else {
                threshold_ap(&reference, scores)
            };
            assert_close(ranked.average_precision(), expected_ap);
        }
    }

    #[test]
    fn one_vs_rest_matches_binary_per_class() {
        let mut rng = SplitMix64(2027);
        let classes = [0, 1, 2, 3];
        for _ in 0..100 {
            let (reference, scores) = random_problem(&mut rng, 4);
            let n = reference.len();
            let auc = roc_auc_ovr(&reference, &scores, &classes).unwrap();
            let ap = average_precision_ovr(&reference, &scores, &classes).unwrap();
            for (c, &class) in classes.iter().enumerate() {
                let column = &scores[c * n..(c + 1) * n];
                assert_close(auc[c], roc_auc(&reference, column, class).unwrap());
                assert_close(ap[c], average_precision(&reference, column, class).unwrap());
            }
        }
    }
}
