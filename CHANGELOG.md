# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-11-13

### Fixed

- `decode_attribute` now dispatches on the protobuf `AttributeProto.type`
  discriminator instead of guessing the populated value field from emptiness or
  zero-ness. Legal attributes whose value is `INT 0`, `FLOAT 0.0`, an empty
  `STRING`, an `INTS`/`FLOATS` list containing zeros, or an empty list were
  previously rejected with `attribute "…" carries no value`. Attributes that
  declare `UNDEFINED` (or an unknown numeric type) are still rejected, with a
  clear error naming the attribute.
- Because zero-valued attributes now decode correctly, consumers no longer need
  workarounds that rewrite such attributes into non-zero sentinels before
  decoding (sciencekit will drop its sentinel shim once it bumps to this
  revision).

### Changed

- Bumped `serde-onnx` and `serde-onnx-macros` to `0.1.2`.

## [0.1.1] - 2026-11-13

### Added

- Declared `rust-version = "1.85"` in the workspace, the `serde-onnx` crate and
  the `serde-onnx-macros` crate, so cargo rejects unsupported toolchains early.
- Added an MSRV job to CI that checks the workspace with Rust 1.85, all
  features and all targets, preventing future regressions.

### Fixed

- Restored Rust 1.85 compatibility:
  - Replaced the unstable `usize::is_multiple_of` with an equivalent modulo
    check in the protobuf codec.
  - Dropped `const` from `TensorType::is_scalar` and
    `ValidationReport::is_ok`, whose bodies relied on `Vec::is_empty` as a
    const function; the public signatures are otherwise unchanged.
  - Rewrote `if let ... && ...` let-chains as nested `if let` blocks in the
    validator, exporter, protobuf codec and tree-ensemble ops, preserving
    behaviour.
- Removed the stray `examples/mod.rs`, root `mod.rs` and `tests/mod.rs` files
  that did not match the example and test targets declared in `Cargo.toml`,
  breaking `cargo test --all-targets` and the examples.

### Changed

- Bumped `serde-onnx` and `serde-onnx-macros` to `0.1.1`.

## [0.1.0]

- Initial release.

[0.1.2]: https://github.com/neurono-ml/serde-onnx/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/neurono-ml/serde-onnx/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/neurono-ml/serde-onnx/releases/tag/v0.1.0
