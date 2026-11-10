# rtemis-rs

[rtemis](https://www.rtemis.org) Rust workspace.

| Crate | Module | Contents |
|---|---|---|
| [`rtemis`](crates/rtemis) | `rtemis::ml::metrics` | ROC AUC and average precision, binary and one-vs-rest |

The modules of `rtemis` mirror the rtemis R packages: `rtemis::ml` corresponds
to rtemis.ml.

Language bindings live in their own repositories and depend on the published
crates:

| Binding | Repository |
|---|---|
| R package `rtemis.ml` | [rtemis-org/rtemis.ml](https://github.com/rtemis-org/rtemis.ml) |

## Development

```sh
cargo test --workspace
cargo +1.63.0 test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

The minimum supported Rust version (MSRV) is declared as `rust-version` in each
crate's `Cargo.toml` (1.63 for `rtemis`) and tested in CI.

## License

[BSD-3-Clause](LICENSE).
