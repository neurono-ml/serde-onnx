# serde-onnx

Typed Rust IR + protobuf codec + typed `ai.onnx.ml` ops + export/import framework for ONNX (Phase 1).

## Module map

- `ir` — typed IR: `ElemType`, `ValueType` (`Tensor`/`SparseTensor`/`Sequence`/`Map`/`Optional`, `Dim`), `Tensor`/`TensorData`, `Attribute`/`AttributeValue`, `Node`, `Graph`, `Model`, structural validation (SSA, defined values, topological order). Constants: `IR_VERSION = 10`, `ONNX_OPSET_VERSION = 21`, `ML_OPSET_VERSION = 5`.
- `ml` — typed ops + `NodePayload` recognition factory (`make_node_payload`, `Raw` fallback). Targets: `ML_EXPORT_OPSET_TARGET = 4`, `CORE_TYPED_OPSET_TARGET = 21`.
- `proto` — prost-generated bindings (`third_party/*.proto`) + IR↔proto encode/decode, dtype×size validation, external-data reference passthrough. Acceptance caps: IR ≤ 10, `ai.onnx` ≤ 25, `ai.onnx.ml` ≤ 5.
- `export` (feature `export`) — `GraphBuilder`, `ToOnnx`, `export_model` (validates before serializing), auto-naming, opset registration.
- `import` (feature `import`) — `.onnx` decode → semantic graph classified by `(domain, op_type)`; non-fatal out-of-scope marking; typed/`Raw` visitor.
- `macros` (`serde-onnx-macros`, re-exported via `serde_onnx::export`) — `#[derive(OnnxExport)]` with `#[onnx(op, domain, attr, ty)]`.

Typed `ai.onnx.ml` ops: `Scaler`, `Imputer`, `Normalizer`, `LabelEncoder`, `OneHotEncoder`, `DictVectorizer`, `FeatureVectorizer`, `LinearClassifier`, `LinearRegressor`, `SVMClassifier`, `SVMRegressor`, `TreeEnsembleClassifier`, `TreeEnsembleRegressor`, `ZipMap`. Typed `ai.onnx` subset: `Cast`, `Reshape`, `Concat`, `Gather`, `Identity`.

## Opset targets

| Purpose | Target |
|---|---|
| ML export (`ai.onnx.ml`) | 4 |
| Core typed (`ai.onnx` subset) | 21 |
| Codec acceptance: IR | ≤ 10 |
| Codec acceptance: `ai.onnx` | ≤ 25 |
| Codec acceptance: `ai.onnx.ml` | ≤ 5 |

## Not implemented (Phase 2)

- Subgraphs `If`/`Loop`/`Scan`: preserved as raw data only, no typed modeling or validation.
- `FunctionProto` / `training_info`: raw passthrough only.
- Full `ai.onnx` domain: only `Cast`/`Reshape`/`Concat`/`Gather`/`Identity` are typed; everything else decodes to `Raw`.
- f16/bf16 explicit f32↔bits; fp8/int4 stay raw bytes.
