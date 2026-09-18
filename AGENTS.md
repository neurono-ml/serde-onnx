# AGENTS.md — serde-onnx

## Language rule (mandatory)

**All code, comments, documentation, identifiers, and commit messages MUST be
written in English.** This includes:

- Rust identifiers (types, functions, variables, modules, features, attributes)
- Code comments (`//`, `///`, `//!`)
- Documentation (README.md, `examples/README.md`, rustdoc, ADRs, specs)
- Example output and user-facing messages (`println!`, error strings, warnings)
- Commit messages and pull request descriptions

Never write code, comments, or docs in Portuguese or any other language. When
editing a file that contains non-English text, translate it to English as part
of the change.

## Working agreements

- Keep the typed IR (`src/ir`), typed ops (`src/ml`), protobuf codec
  (`src/proto`), export (`src/export`), and import (`src/import`) layers
  decoupled; do not leak protobuf types into the IR.
- `export_model` must validate before serializing; unknown ops decode to
  `NodePayload::Raw` instead of failing.
- Run `cargo fmt`, `cargo clippy`, and the relevant tests/examples before
  finishing a change.
