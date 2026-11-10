# rtemis

Platform-independent core of [rtemis](https://www.rtemis.org): machine learning
metrics with no dependencies beyond the Rust standard library.

## Classification metrics

`rtemis::ml::metrics::classification` computes the area under the ROC curve (AUC)
and average precision (AP) for binary outcomes, and their one-vs-rest values
for each class of a multiclass outcome.

- The AUC is the probability that a randomly chosen positive case scores
  higher than a randomly chosen negative case, with ties counted as one half.
- The AP is the area under the stepwise precision-recall curve, with each
  distinct score as one threshold.

Scores are sorted once per problem, after which each metric is one linear
pass. Extra memory is 8 bytes per case. Reference labels are a slice of any
type with equality, so bindings pass their own arrays without copying.

```rust
use rtemis::ml::metrics::classification::{average_precision, roc_auc, roc_auc_ovr};

let reference = [0, 0, 1, 1];
let scores = [0.1, 0.4, 0.35, 0.8];
assert_eq!(roc_auc(&reference, &scores, 1), Ok(0.75));
let ap = average_precision(&reference, &scores, 1).unwrap();
assert!((ap - 5.0 / 6.0).abs() < 1e-15);

// One-vs-rest: scores is an n-by-k column-major matrix, one column per class.
let reference = ["a", "b", "a"];
let scores = [0.9, 0.2, 0.7, 0.1, 0.8, 0.3];
assert_eq!(roc_auc_ovr(&reference, &scores, &["a", "b"]), Ok(vec![1.0, 1.0]));
```

## Minimum supported Rust version

Rust 1.63.

## License

BSD-3-Clause.
