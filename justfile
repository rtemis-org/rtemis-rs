# justfile
# ::rtemis-rs::
# 2026- EDG rtemis.org

# Minimum supported Rust version, from `rust-version` in the crate manifest,
# as a rustup toolchain name (1.63 -> 1.63.0).
msrv := `awk -F'"' '/^rust-version/ { v = $2; if (split(v, p, ".") == 2) v = v ".0"; print v; exit }' crates/rtemis/Cargo.toml`

# List available recipes
default:
    @just --list

_msg msg:
    @printf '\033[38;2;108;163;160m[%s] %s\033[0m\n' "$(date '+%Y-%m-%d %H:%M:%S')" "{{ msg }}"

# Format Rust code
fmt:
    @just _msg "─── Formatting... ───"
    cargo fmt --all
    @just _msg "Done"

# Check formatting without modifying files
fmt-check:
    @just _msg "─── Checking formatting... ───"
    cargo fmt --all --check
    @just _msg "Done"

# Run clippy on all targets, with warnings as errors
lint:
    @just _msg "─── Running clippy... ───"
    cargo clippy --workspace --all-targets --locked -- -D warnings
    @just _msg "Done"

# Run unit tests and doctests, including the README examples
test:
    @just _msg "─── Running tests... ───"
    cargo test --workspace --locked
    @just _msg "Done"

# Run the tests with the minimum supported Rust version, `rust-version` in crates/rtemis/Cargo.toml
test-msrv:
    @just _msg "─── Running tests with Rust {{ msrv }}... ───"
    cargo +{{ msrv }} test --workspace --locked
    @just _msg "Done"

# Build the documentation, with warnings as errors
doc:
    @just _msg "─── Building documentation... ───"
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
    @just _msg "Done"

# Build the documentation and open it in a browser
doc-open:
    cargo doc --workspace --no-deps --locked --open

# Run every check that CI runs, except packaging
check: fmt-check lint test test-msrv doc

# List the files of the working tree that a crates.io publication of `rtemis` uploads
package-list:
    cargo package -p rtemis --list --allow-dirty

# Package and verify `rtemis` without uploading it (requires a committed tree)
package:
    @just _msg "─── Packaging rtemis... ───"
    cargo package -p rtemis --locked
    @just _msg "Done"

# Run every crates.io check of `rtemis` without uploading it (requires a committed tree)
publish-dry-run:
    @just _msg "─── Dry run of cargo publish... ───"
    cargo publish -p rtemis --dry-run --locked
    @just _msg "Done"

# Remove build artifacts
clean:
    @just _msg "─── Cleaning build artifacts... ───"
    cargo clean
    @just _msg "Done"
