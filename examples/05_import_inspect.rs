//! # Example 05 — Inspect an `.onnx` file: typed vs `Raw` + warnings
//!
//! ## What this example teaches
//!
//! 1. How an arbitrary `.onnx` file is classified node by node as
//!    **typed** (`Scaler`, `LinearClassifier`, …) or **Raw**
//!    (unknown op, custom domain, or future opset).
//! 2. How to read non-fatal `warnings`: subgraph attributes (`Loop`/`If`),
//!    `FunctionProto`, and `training_info` are preserved as bytes without
//!    breaking the decode.
//! 3. The helper iterators [`DecodedModel::typed`](serde_onnx::import::DecodedModel::typed)
//!    and [`raw_nodes`](serde_onnx::import::DecodedModel::raw_nodes).
//!
//! ## How to run
//!
//! ```sh
//! cargo run --example 05_import_inspect --features export,import
//! ```

use serde_onnx::export::{GraphBuilder, export_model};
use serde_onnx::import::{DecodedModel, ImportWarningKind};
use serde_onnx::ir::{Attribute, Dim, ElemType, GraphRef, Model, Node, OpsetId, ValueType};
use serde_onnx::ml::Scaler;
use serde_onnx::proto::encode_model;

fn matrix() -> ValueType {
    ValueType::tensor(ElemType::Float, Some(vec![Dim::Unknown, Dim::Unknown]))
}

fn main() {
    // -----------------------------------------------------------------------
    // SCENARIO A — Model with 1 typed node + 1 unknown (Raw) node.
    // -----------------------------------------------------------------------
    // We hand-build a graph whose second node uses a fictional op from a
    // custom domain. Import must recognize the Scaler and preserve the
    // future op as `Raw`, without error.
    let mut model = export_model(&ScalerModel, "mixed_graph").expect("scaler export");
    model.graph.nodes.push(Node::new(
        "FancyFutureOp", // op this library version does not type yet
        "custom.domain", // unknown domain
        vec!["node_0_Scaler_out".to_string()],
        vec!["z".to_string()],
        Vec::new(),
    ));
    model
        .graph
        .outputs
        .push(serde_onnx::ir::ValueInfo::new("z", matrix()));

    let decoded = DecodedModel::decode_bytes(&encode_model(&model)).expect("decode");
    println!("scenario A: {} payload(s)", decoded.payloads.len());
    println!("  typed: {}", decoded.typed().count());
    println!("  raw:   {}", decoded.raw_nodes().count());
    for (i, node) in decoded.raw_nodes() {
        println!("  raw [{i}]: {} :: {:?}", node.op_type, node.domain);
    }
    assert_eq!(decoded.typed().count(), 1);
    assert_eq!(decoded.raw_nodes().count(), 1);

    // -----------------------------------------------------------------------
    // SCENARIO B — A subgraph attribute produces a non-fatal warning.
    // -----------------------------------------------------------------------
    // `Loop`/`If`/`Scan` have no typed modeling yet (phase 2 of the project):
    // the subgraph travels as bytes (`GraphRef { proto }`) and import warns.
    let mut graph = serde_onnx::ir::Graph::new("loop_graph");
    graph
        .inputs
        .push(serde_onnx::ir::ValueInfo::new("x", matrix()));
    graph.nodes.push(Node::new(
        "Loop",
        "",
        vec!["x".to_string()],
        vec!["y".to_string()],
        vec![Attribute::graph(
            "body",
            GraphRef {
                name: "body".to_string(),
                proto: vec![1, 2, 3], // opaque subgraph bytes
            },
        )],
    ));
    graph
        .outputs
        .push(serde_onnx::ir::ValueInfo::new("y", matrix()));
    let loop_model = Model::new(
        graph,
        vec![OpsetId {
            domain: String::new(),
            version: 21,
        }],
    );
    let decoded_loop = DecodedModel::decode_bytes(&encode_model(&loop_model)).expect("decode loop");
    println!("scenario B: warnings = {}", decoded_loop.warnings.len());
    for w in &decoded_loop.warnings {
        println!("  warning: {w}");
    }
    assert!(decoded_loop.payloads[0].is_raw());
    assert!(
        decoded_loop
            .warnings
            .iter()
            .any(|w| w.kind == ImportWarningKind::SubgraphAttribute)
    );

    // -----------------------------------------------------------------------
    // SCENARIO C — `FunctionProto` / `training_info` also produce warnings.
    // -----------------------------------------------------------------------
    let mut fn_model = export_model(&ScalerModel, "fn_graph").expect("scaler export");
    fn_model.functions_raw.push(serde_onnx::ir::FunctionRaw {
        domain: "custom.domain".to_string(),
        name: "CustomFn".to_string(),
        overload: String::new(),
        proto: vec![9, 9, 9],
    });
    fn_model
        .training_info_raw
        .push(serde_onnx::ir::TrainingInfoRaw { proto: vec![7, 7] });
    let decoded_fn = DecodedModel::decode_bytes(&encode_model(&fn_model)).expect("decode fn");
    println!("scenario C: warnings = {}", decoded_fn.warnings.len());
    for w in &decoded_fn.warnings {
        println!("  warning: {w}");
    }
    assert_eq!(decoded_fn.warnings.len(), 2);

    println!("import inspection: OK");
    println!();
    println!("Import MENTAL MODEL:");
    println!("  - typed   → the lib understands the op; use `NodePayload::*` safely;");
    println!("  - Raw     → op preserved byte-for-byte; re-export without loss;");
    println!("  - warning → advanced part (subgraph/function/training) traveled as bytes.");
}

// Minimal model, just to produce a valid Scaler for scenarios A and C.
#[derive(Default)]
struct ScalerModel;

impl serde_onnx::export::ToOnnx for ScalerModel {
    fn to_graph(
        &self,
        builder: &mut GraphBuilder,
    ) -> Result<serde_onnx::export::ValueRef, serde_onnx::export::ExportError> {
        let x = builder.input("X", matrix())?;
        let op = Scaler {
            offset: Some(vec![0.0]),
            scale: Some(vec![1.0]),
        };
        let out = builder.emit_op(&op, vec![x], 1)?.pop().expect("1 output");
        builder.output(out.name(), matrix())?;
        Ok(serde_onnx::export::ValueRef::from(out.name().to_string()))
    }
}
