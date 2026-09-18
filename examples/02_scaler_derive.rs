//! # Example 02 — Export WITH macros (`#[derive(OnnxExport)]`)
//!
//! ## What this example teaches
//!
//! 1. How to get the **same graph as example 01** without hand-writing `to_graph`.
//! 2. The anatomy of the `#[onnx(...)]` annotations:
//!    - on the struct: `op` (op name) + `domain` (ONNX domain);
//!    - on each field: `attr` (matching ONNX attribute name).
//! 3. Macro rules: fields without `#[onnx(attr)]` are ignored, `Option<T>`
//!    becomes an optional attribute (skipped when `None`), and the supported
//!    types are `Vec<f32>`, `f32`, `Vec<i64>`, `i64`, `Vec<String>`, `String`.
//! 4. How to prove the derive generates bytecode identical to the manual `ToOnnx`.
//!
//! ## When to use this approach
//!
//! Use the derive for simple structs with **1 input → 1 op → 1 output** and
//! `tensor(float, [N, ?])` I/O. For multi-node pipelines, initializers, or
//! different I/O types, implement `ToOnnx` manually (examples 01 and 03).
//!
//! ## How to run
//!
//! ```sh
//! cargo run --example 02_scaler_derive --features export
//! ```

use serde_onnx::OnnxExport; // re-exported via `serde_onnx::export` (feature `export`)
use serde_onnx::export::{ToOnnx, ValueRef, export_model};
use serde_onnx::ir::{Attribute, Dim, ElemType, ValueType};

// ---------------------------------------------------------------------------
// STEP 1 — Annotate the struct with `#[derive(OnnxExport)]`.
// ---------------------------------------------------------------------------
// `op = "Scaler"` + `domain = "ai.onnx.ml"` say which ONNX node to generate.
// Each field marked with `attr = "..."` becomes one node attribute.
// The `not_exported` field has NO annotation: it is skipped by the export
// (handy for metadata such as the sklearn version, hyperparams, etc.).
/// Declarative equivalent of example 01's `FittedStandardScaler`.
#[derive(OnnxExport, Debug)]
#[onnx(op = "Scaler", domain = "ai.onnx.ml")]
struct DerivedScaler {
    /// Becomes the ONNX `offset` attribute (skipped when `None`).
    #[onnx(attr = "offset")]
    offset: Option<Vec<f32>>,
    /// Becomes the ONNX `scale` attribute (skipped when `None`).
    #[onnx(attr = "scale")]
    scale: Option<Vec<f32>>,
    /// Helper field: excluded from the export.
    #[allow(dead_code)]
    not_exported: String,
}

// ---------------------------------------------------------------------------
// STEP 2 (comparison) — The same op, written by hand.
// ---------------------------------------------------------------------------
// We keep a manual implementation next to the derive to show that the
// macro-generated code is equivalent.
struct ManualScaler {
    offset: Option<Vec<f32>>,
    scale: Option<Vec<f32>>,
}

fn matrix() -> ValueType {
    ValueType::tensor(ElemType::Float, Some(vec![Dim::Unknown, Dim::Unknown]))
}

impl ToOnnx for ManualScaler {
    fn to_graph(
        &self,
        builder: &mut serde_onnx::export::GraphBuilder,
    ) -> Result<ValueRef, serde_onnx::export::ExportError> {
        let io = matrix();
        let input = builder.input("X", io.clone())?;
        let mut attributes: Vec<Attribute> = Vec::new();
        // Each `Option` that is `Some` becomes one `floats` attribute.
        if let Some(values) = &self.offset {
            attributes.push(Attribute::floats("offset", values.clone()));
        }
        if let Some(values) = &self.scale {
            attributes.push(Attribute::floats("scale", values.clone()));
        }
        let mut outputs = builder.push_node(
            "Scaler",
            "ai.onnx.ml",
            vec![input.name().to_string()],
            attributes,
            1,
        )?;
        let output = outputs.pop().expect("single output");
        builder.output(output.name().to_string(), io)?;
        Ok(output)
    }
}

fn main() {
    // -----------------------------------------------------------------------
    // STEP 3 — Export both ways and compare.
    // -----------------------------------------------------------------------
    let derived = DerivedScaler {
        offset: Some(vec![1.0, 2.0]),
        scale: Some(vec![0.5, 0.25]),
        not_exported: "e.g. sklearn_version=1.3".to_string(),
    };
    let manual = ManualScaler {
        offset: Some(vec![1.0, 2.0]),
        scale: Some(vec![0.5, 0.25]),
    };

    let from_derive = export_model(&derived, "g").expect("derive export");
    let from_manual = export_model(&manual, "g").expect("manual export");

    // The generated graph is identical — the macro only automates the
    // boilerplate above.
    assert_eq!(from_derive.graph, from_manual.graph);
    assert_eq!(from_derive.opset_import, from_manual.opset_import);
    println!(
        "derive == manual: OK ({} node)",
        from_derive.graph.nodes.len()
    );

    let node = &from_derive.graph.nodes[0];
    println!("op: {} :: {:?}", node.op_type, node.domain);
    println!("auto-named output: {}", node.outputs[0]);

    // The payload is recognized as a typed `Scaler` on import (example 05).
    let payload = serde_onnx::ml::make_node_payload(node);
    assert!(matches!(payload, serde_onnx::ml::NodePayload::Scaler(_)));
    println!("payload recognized as typed Scaler: OK");

    // -----------------------------------------------------------------------
    // STEP 4 — The `None` case: no attribute is emitted.
    // -----------------------------------------------------------------------
    let empty = DerivedScaler {
        offset: None,
        scale: None,
        not_exported: String::new(),
    };
    let empty_model = export_model(&empty, "g").expect("empty export");
    assert!(empty_model.graph.nodes[0].attributes.is_empty());
    println!("None fields produce an attribute-free node: OK");

    // QUICK RECIPE for your own op:
    //   1. `#[derive(OnnxExport)]` + `#[onnx(op = "Imputer", domain = "ai.onnx.ml")]`
    //   2. Mark each parameter with `#[onnx(attr = "<onnx_name>")]`
    //   3. Use `Option<...>` for optional attributes.
    //   4. `export_model(&my_struct, "my_graph")` — done.
}
