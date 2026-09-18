# `serde-onnx` examples

Didactic examples, ordered from simplest to most complete. Each file is
self-documented: the `//!` header explains the goal, and the code is
commented step by step in English.

| Example | Topic | No macros | With macros | Features |
|---|---|---|---|---|
| `sklearn_scaler.rs` | Minimal `Scaler` + protobuf round-trip | ✅ | — | `export` |
| `01_scaler_manual.rs` | Manual `ToOnnx`, fully commented | ✅ | — | `export` |
| `02_scaler_derive.rs` | `#[derive(OnnxExport)]` + `derive == manual` proof | — | ✅ | `export` |
| `03_pipeline_imputer_scaler.rs` | Multi-node pipeline (`Imputer` → `Scaler`) + initializer | ✅ | — | `export` |
| `04_linear_classifier_e2e.rs` | End to end: export → `.onnx` on disk → import via `DecodedModel` + `PayloadVisitor` | ✅ | — | `export`, `import` |
| `05_import_inspect.rs` | Typed vs `Raw`, warnings (subgraph, function, training) | ✅ | — | `export`, `import` |

## How to run

```sh
# Export (no macros, with macros, pipeline)
cargo run --example 01_scaler_manual --features export
cargo run --example 02_scaler_derive --features export
cargo run --example 03_pipeline_imputer_scaler --features export

# End to end (export + import)
cargo run --example 04_linear_classifier_e2e --features export,import
cargo run --example 05_import_inspect --features export,import

# Legacy (kept for compatibility)
cargo run --example sklearn_scaler --features export
```

## Suggested learning path

1. **Start with `01`**: understand `GraphBuilder` → `emit_op` → `export_model` → `encode_model`.
2. **Go to `02`** if your case is a simple struct (1 op): swap the manual
   `ToOnnx` for `#[derive(OnnxExport)]`.
3. **Study `03`** to chain several ops and embed initializers.
4. **Close the loop with `04`**: save an `.onnx` file and re-import it with `DecodedModel`.
5. **Use `05` as a reference** when consuming third-party models
   (unknown ops become `Raw`, advanced parts become warnings).
