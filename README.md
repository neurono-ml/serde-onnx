# Serde ONNX   [![Latest Version](https://img.shields.io/crates/v/serde-onnx.svg)](https://crates.io/crates/serde-onnx) [![Docs](https://docs.rs/serde-onnx/badge.svg)](https://docs.rs/serde-onnx)

**Serde ONNX is a framework for working with ONNX models as strongly typed Rust data structures, efficiently and generically.**

```toml
[dependencies]
serde-onnx = { version = "0.1", features = ["export", "import"] }
```

You may be looking for:

- [API documentation](https://docs.rs/serde-onnx)
- [ONNX specification](https://onnx.ai/onnx/intro/)
- [Runnable examples](examples/README.md)
- [Netron model visualizer](https://netron.app)

ONNX is an open standard for machine learning models. A model is a
computation graph of typed operator nodes — for example, a scaler that
normalizes its input before a classifier consumes it:

```text
X [N, 3] ──▶ Scaler ──▶ Xs ──▶ LinearClassifier ──▶ Y (labels), Z (scores)
```

There are three common ways that you might find yourself needing to work
with ONNX data in Rust.

- **As raw bytes.** An unprocessed `.onnx` protobuf blob that you receive
  from a training pipeline, read from a file, or prepare to ship to an
  inference runtime.
- **As a generic graph.** Maybe you want to inspect an arbitrary model —
  list its nodes, check its opsets — without knowing which operators it
  contains. Or you want to forward operators your code does not understand
  without losing a byte.
- **As strongly typed Rust data structures.** When you expect all or most
  of your nodes to be operators you know (`Scaler`, `LinearClassifier`,
  …) and want the compiler to check attribute names and types instead of
  stringly-typed dictionaries tripping you up.

Serde ONNX provides efficient, safe ways of converting models between each
of these representations.

## Deriving export with a macro

The fastest way to export a model: for the common case of a struct mapping
1-to-1 onto a single operator, the
[`OnnxExport`](https://docs.rs/serde-onnx/latest/serde_onnx/derive.OnnxExport.html)
derive macro (feature `export`) generates the `ToOnnx` implementation for
you. Fields annotated with `#[onnx(attr = "...")]` become node attributes;
`Option<T>` fields are skipped when `None`; fields without the annotation
are ignored.

```rust
use serde_onnx::OnnxExport;
use serde_onnx::export::export_model;

#[derive(OnnxExport)]
#[onnx(op = "Scaler", domain = "ai.onnx.ml")]
struct DerivedScaler {
    #[onnx(attr = "offset")]
    offset: Option<Vec<f32>>,
    #[onnx(attr = "scale")]
    scale: Option<Vec<f32>>,
    // Helper field: excluded from the export.
    not_exported: String,
}

fn derive_example() -> anyhow::Result<()> {
    let scaler = DerivedScaler {
        offset: Some(vec![1.0, 2.0]),
        scale: Some(vec![0.5, 0.25]),
        not_exported: "sklearn_version=1.3".to_string(),
    };
    // Validates the graph (SSA, defined values, topological order),
    // then serializes to .onnx protobuf bytes.
    let model = export_model(&scaler, "g")?;
    assert_eq!(model.graph.nodes[0].op_type, "Scaler");
    Ok(serde_onnx::proto::encode_model(&model));
}
```

Supported field types are `Vec<f32>`, `f32`, `Vec<i64>`, `i64`,
`Vec<String>`, and `String`, each optionally wrapped in `Option`. The
sections below cover what the macro cannot do: manual `ToOnnx`
implementations (multi-node pipelines, custom I/O types, derived
attributes) and reading models back.

## Inspecting any model as a generic graph

Any valid `.onnx` file can be decoded into a generic intermediate
representation: a [`Model`](https://docs.rs/serde-onnx/latest/serde_onnx/ir/struct.Model.html)
of [`Node`](https://docs.rs/serde-onnx/latest/serde_onnx/ir/struct.Node.html)s
with stringly-typed inputs, outputs, and attributes. The
[`proto::decode_model`](https://docs.rs/serde-onnx/latest/serde_onnx/proto/fn.decode_model.html)
function parses bytes; there is also
[`decode_file`](https://docs.rs/serde-onnx/latest/serde_onnx/import/struct.DecodedModel.html#method.decode_file)
for reading straight from disk.

```rust
fn inspect(path: &str) -> anyhow::Result<()> {
    // Some .onnx file. Maybe it comes from a training pipeline.
    let bytes = std::fs::read(path)?;

    // Parse the bytes into a generic graph.
    let model = serde_onnx::proto::decode_model(&bytes)?;

    // Access parts of the model just like any other Rust data structure.
    println!("graph: {}", model.graph.name);
    for node in &model.graph.nodes {
        println!("  {} :: {:?}: {} -> {}",
            node.op_type,
            node.domain,
            node.inputs.join(", "),
            node.outputs.join(", "));
    }
    for opset in &model.opset_import {
        println!("  opset {:?} version {}", opset.domain, opset.version);
    }
    Ok(())
}
```

The generic representation is sufficient for inspection and forwarding, but
it can be tedious to work with for anything more significant. Attribute
access is verbose to implement correctly — imagine checking that every
`Scaler` node carries a `scale` attribute of floats — and the compiler is
powerless to help you when you make a mistake, for example typoing
`"scale"` as `"sacle"` in one of the dozens of places it is used in your
code.

## Exporting strongly typed operators by hand

When the derive macro above is not enough — derived attributes, several
chained nodes, custom I/O types — implement
[`ToOnnx`](https://docs.rs/serde-onnx/latest/serde_onnx/export/trait.ToOnnx.html)
manually. Each supported operator is still a plain Rust struct implementing
the
[`OnnxOp`](https://docs.rs/serde-onnx/latest/serde_onnx/ml/trait.OnnxOp.html)
trait, so attribute names and types are checked at compile time.

```rust
use serde_onnx::export::{GraphBuilder, ToOnnx, ValueRef, export_model};
use serde_onnx::ir::{Dim, ElemType, ValueType};
use serde_onnx::ml::Scaler;

// A fitted scaler: just the parameters learned during training.
struct FittedStandardScaler {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl ToOnnx for FittedStandardScaler {
    fn to_graph(&self, builder: &mut GraphBuilder) -> Result<ValueRef, serde_onnx::export::ExportError> {
        // Describe the input/output type: float tensor [N, 3].
        let io = ValueType::tensor(ElemType::Float, Some(vec![Dim::Unknown, Dim::Fixed(3)]));
        let input = builder.input("X", io.clone())?;

        // The typed op knows how to produce the right attributes.
        let op = Scaler {
            offset: Some(self.mean.clone()),
            scale: Some(self.std.iter().map(|v| 1.0 / v).collect()),
        };
        let mut outputs = builder.emit_op(&op, vec![input], 1)?;
        let output = outputs.pop().expect("one output");
        builder.output(output.name().to_string(), io)?;
        Ok(output)
    }
}

fn export_example() -> anyhow::Result<Vec<u8>> {
    let fitted = FittedStandardScaler { mean: vec![1.0, 2.0, 3.0], std: vec![0.5, 1.0, 2.0] };

    // Runs `to_graph`, then validates the graph (SSA, defined values,
    // topological order) before a single byte is written.
    let model = export_model(&fitted, "standard_scaler")?;

    // Serialize to .onnx protobuf bytes. Print, write to a file,
    // or send to an inference runtime.
    Ok(serde_onnx::proto::encode_model(&model))
}
```

Once you have an `op` of type `Scaler`, your IDE and the Rust compiler can
help you use it correctly like they do for any other Rust code. Misspelled
attributes and wrong value types become compile errors instead of corrupt
models discovered at inference time.

Supported
[`ml` operators](https://docs.rs/serde-onnx/latest/serde_onnx/ml/index.html)
include `Scaler`, `Imputer`, `Normalizer`, `LabelEncoder`, `OneHotEncoder`,
`DictVectorizer`, `FeatureVectorizer`, `LinearClassifier`,
`LinearRegressor`, `SvmClassifier`, `SvmRegressor`,
`TreeEnsembleClassifier`, `TreeEnsembleRegressor`, `ZipMap`, plus the core
`ai.onnx` subset `Cast`, `Reshape`, `Concat`, `Gather`, `Identity`.

## Building pipelines with `GraphBuilder`

Real models chain several operators. [`GraphBuilder`](https://docs.rs/serde-onnx/latest/serde_onnx/export/struct.GraphBuilder.html)
is the low-level builder behind every export: declare inputs, emit typed
ops, embed constant initializers, declare outputs. One node's output
becomes the next node's input.

```rust
use serde_onnx::export::{GraphBuilder, ToOnnx, ValueRef};
use serde_onnx::ir::{Dim, ElemType, ValueType};
use serde_onnx::ml::{Imputer, Scaler};

struct CleaningPipeline {
    mean: Vec<f32>,
    scale: Vec<f32>,
}

impl ToOnnx for CleaningPipeline {
    fn to_graph(&self, builder: &mut GraphBuilder) -> Result<ValueRef, serde_onnx::export::ExportError> {
        let io = ValueType::tensor(ElemType::Float, Some(vec![Dim::Unknown, Dim::Fixed(3)]));
        let input = builder.input("X", io.clone())?;

        // Node 1: replace NaNs. Attributes are validated eagerly —
        // only one of each `*_float` / `*_int64` pair may be set.
        let imputer = Imputer {
            imputed_value_floats: None,
            imputed_value_int64s: None,
            replaced_value_float: Some(-1.0),
            replaced_value_int64: None,
        };
        let imputed = builder.emit_op(&imputer, vec![input], 1)?.pop().expect("one output");

        // Node 2: normalize, consuming node 1's output.
        let scaler = Scaler { offset: Some(self.mean.clone()), scale: Some(self.scale.clone()) };
        let scaled = builder.emit_op(&scaler, vec![imputed], 1)?.pop().expect("one output");

        // A constant embedded in the .onnx (weights, shapes, vocabularies, ...).
        builder.initializer("shape_hint", vec![Dim::Fixed(2)], vec![0i64, 3i64])?;

        builder.output(scaled.name().to_string(), io)?;
        Ok(scaled)
    }
}
```

```text
X [N, 3] ──▶ Imputer ──▶ node_0_Imputer_out ──▶ Scaler ──▶ Y [N, 3]
```

## Deriving export with a macro

For the common case of a struct mapping 1-to-1 onto a single operator, the
[`OnnxExport`](https://docs.rs/serde-onnx/latest/serde_onnx/derive.OnnxExport.html)
derive macro (feature `export`) generates the `ToOnnx` implementation for
you. Fields annotated with `#[onnx(attr = "...")]` become node attributes;
`Option<T>` fields are skipped when `None`; fields without the annotation
are ignored.

```rust
use serde_onnx::OnnxExport;
use serde_onnx::export::export_model;

#[derive(OnnxExport)]
#[onnx(op = "Scaler", domain = "ai.onnx.ml")]
struct DerivedScaler {
    #[onnx(attr = "offset")]
    offset: Option<Vec<f32>>,
    #[onnx(attr = "scale")]
    scale: Option<Vec<f32>>,
    // Helper field: excluded from the export.
    not_exported: String,
}

fn derive_example() -> anyhow::Result<()> {
    let scaler = DerivedScaler {
        offset: Some(vec![1.0, 2.0]),
        scale: Some(vec![0.5, 0.25]),
        not_exported: "sklearn_version=1.3".to_string(),
    };
    // Same validated export path as the manual implementation.
    let model = export_model(&scaler, "g")?;
    assert_eq!(model.graph.nodes[0].op_type, "Scaler");
    Ok(())
}
```

Supported field types are `Vec<f32>`, `f32`, `Vec<i64>`, `i64`,
`Vec<String>`, and `String`, each optionally wrapped in `Option`. Use the
manual `ToOnnx` implementation for multi-node pipelines, custom I/O types,
or derived attributes.

## Importing back as typed payloads

A model you (or anyone else) exported can be read back with
[`DecodedModel`](https://docs.rs/serde-onnx/latest/serde_onnx/import/struct.DecodedModel.html)
(feature `import`). Every node is classified as a typed
[`NodePayload`](https://docs.rs/serde-onnx/latest/serde_onnx/ml/enum.NodePayload.html)
or preserved byte-for-byte as `Raw` — unknown, custom-domain, or future
operators never fail the decode.

```rust
use serde_onnx::import::{DecodedModel, PayloadVisitor};
use serde_onnx::ir::Node;
use serde_onnx::ml::NodePayload;

struct Counter { typed: usize, raw: usize }

impl PayloadVisitor for Counter {
    fn visit_typed(&mut self, _index: usize, _payload: &NodePayload) { self.typed += 1; }
    fn visit_raw(&mut self, _index: usize, _node: &Node) { self.raw += 1; }
}

fn import_example(bytes: &[u8]) -> anyhow::Result<()> {
    let decoded = DecodedModel::decode_bytes(bytes)?;

    // Non-fatal notes: subgraph attributes, FunctionProto,
    // and training_info travel as bytes and are reported here.
    for warning in &decoded.warnings {
        println!("warning: {warning}");
    }

    let mut counter = Counter { typed: 0, raw: 0 };
    decoded.visit(&mut counter);
    println!("typed: {}, raw: {}", counter.typed, counter.raw);

    // Destructure what you know, forward what you don't.
    match &decoded.payloads[0] {
        NodePayload::Scaler(typed) => println!("scale: {:?}", typed.op.scale),
        NodePayload::Raw(node) => println!("forwarding unknown op {}", node.op_type),
        other => println!("got {}", other.node().op_type),
    }
    Ok(())
}
```

## Features

| Feature | Enables | Needs |
|---|---|---|
| `export` | `GraphBuilder`, `ToOnnx`, `export_model`, `#[derive(OnnxExport)]` | nothing |
| `import` | `DecodedModel`, `PayloadVisitor`, `decode_file` | nothing |

With no features you still get the typed IR (`ir`), the typed operators
(`ml`), and the protobuf codec (`proto`).

## Opset targets

| Purpose | Target |
|---|---|
| ML export (`ai.onnx.ml`) | 4 |
| Core typed (`ai.onnx` subset) | 21 |
| Codec acceptance: IR | ≤ 10 |
| Codec acceptance: `ai.onnx` | ≤ 25 |
| Codec acceptance: `ai.onnx.ml` | ≤ 5 |

## Validation

It is strict. `export_model` validates the graph (SSA form, defined
values, topological order) before serializing, so wiring mistakes surface
as aggregated `ExportError::Validation` errors at export time instead of
corrupt files discovered at inference time. On import, out-of-scope parts
(subgraphs, functions, training info) are preserved as raw bytes with
non-fatal warnings rather than failing the decode. Round-trip tests and
property tests live in `tests/`.

## Examples

Runnable, fully commented examples live in [`examples/`](examples/README.md):

| Example | Topic | Features |
|---|---|---|
| `01_scaler_manual` | Manual `ToOnnx` for a `Scaler` | `export` |
| `02_scaler_derive` | `#[derive(OnnxExport)]`, proved equal to manual | `export` |
| `03_pipeline_imputer_scaler` | Multi-node pipeline + initializer | `export` |
| `04_linear_classifier_e2e` | Export → `.onnx` file → typed re-import | `export`, `import` |
| `05_import_inspect` | Typed vs `Raw` nodes, warning kinds | `export`, `import` |

```sh
cargo run --example 01_scaler_manual --features export
cargo run --example 04_linear_classifier_e2e --features export,import
```

## Not implemented (Phase 2)

- Subgraphs `If`/`Loop`/`Scan`: preserved as raw data only, no typed
  modeling or validation.
- `FunctionProto` / `training_info`: raw passthrough only.
- Full `ai.onnx` domain: only `Cast`/`Reshape`/`Concat`/`Gather`/`Identity`
  are typed; everything else decodes to `Raw`.
- f16/bf16 explicit f32↔bits; fp8/int4 stay raw bytes.

## Getting help

ONNX questions are welcome wherever Rustaceans congregate — the
[ONNX community](https://onnx.ai/), Rust forums, or StackOverflow under
the `rust` and `onnx` tags. It's acceptable to file a support issue in this
repo, but chat channels tend to get more eyes. When reporting a problem,
please include the smallest `.onnx` file (or `GraphBuilder` snippet) that
reproduces it, plus the output of the relevant example.

## License

Licensed under either of [Apache License, Version
2.0](https://www.apache.org/licenses/LICENSE-2.0) or [MIT
license](https://opensource.org/licenses/MIT) at your option.
Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this crate by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
