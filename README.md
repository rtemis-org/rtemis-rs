[![crates.io](https://img.shields.io/crates/v/rtemis.svg)](https://crates.io/crates/rtemis) [![docs.rs](https://img.shields.io/docsrs/rtemis)](https://docs.rs/rtemis) [![Rust CI](https://github.com/rtemis-org/rtemis-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/rtemis-org/rtemis-rs/actions/workflows/ci.yml)

# rtemis-rs

[rtemis](https://www.rtemis.org) Rust workspace.

![](https://www.rtemis.org/assets/rtemis-rs-cover.avif)

| Crate | Module | Contents |
|---|---|---|
| [`rtemis`](crates/rtemis) | `rtemis::ml::metrics` | ROC AUC and average precision for binary and one-vs-rest classification |

The modules of `rtemis` mirror the rtemis R packages: `rtemis::ml` corresponds to rtemis.ml.

Language bindings live in their own repositories and depend on the published crates:

| Binding | Repository |
|---|---|
| R package `rtemis.ml` | [rtemis-org/rtemis.ml](https://github.com/rtemis-org/rtemis.ml) |

## Development

| Command | Action |
|---|---|
| `just check` | Formatting, clippy, tests on stable and on the MSRV, and documentation, as in CI |
| `just test` | Unit tests and doctests, including the README examples |
| `just test-msrv` | Tests with the minimum supported Rust version |
| `just package-list` | Files that a crates.io publication uploads |
| `just publish-dry-run` | Every crates.io check, without uploading |

The minimum supported Rust version (MSRV) is declared as `rust-version` in each crate's `Cargo.toml` (1.63 for `rtemis`); `just test-msrv` reads it from there. CI tests it as well.

## License

[BSD-3-Clause](LICENSE).
