//! # Example 04 — End to end: export a `LinearClassifier`, save `.onnx`, re-import
//!
//! ## What this example teaches (the full cycle)
//!
//! 1. **Export**: manual `ToOnnx` for an op with 2 outputs (`LinearClassifier`
//!    emits `Y` = labels and `Z` = scores).
//! 2. **Save**: write the protobuf bytes to a real `.onnx` file.
//! 3. **Import**: read the file back with
//!    [`DecodedModel::decode_file`](serde_onnx::import::DecodedModel) and get the
//!    typed [`NodePayload`](serde_onnx::ml::NodePayload)s.
//! 4. **Visit**: use the [`PayloadVisitor`](serde_onnx::import::PayloadVisitor)
//!    trait to separate typed nodes from `Raw` (unknown/future) ones.
//!
//! ## How to run
//!
//! ```sh
//! cargo run --example 04_linear_classifier_e2e --features export,import
//! # the `linear_classifier.onnx` file is written to the temp dir and can be
//! # opened in Netron (https://netron.app).
//! ```

use serde_onnx::export::{GraphBuilder, ToOnnx, ValueRef, export_model};
use serde_onnx::import::{DecodedModel, PayloadVisitor};
use serde_onnx::ir::{Dim, ElemType, Node, ValueType};
use serde_onnx::ml::{ClassLabels, LinearClassifier, NodePayload};
use serde_onnx::proto::encode_model;

/// Tiny binary linear classifier: `Y = argmax(X·W + b)`.
struct TinyClassifier {
    op: LinearClassifier,
}

impl TinyClassifier {
    fn new() -> Self {
        // 2 input features, 2 output classes.
        // `coefficients` uses the flattened [class, feature] layout.
        TinyClassifier {
            op: LinearClassifier {
                classlabels: ClassLabels {
                    ints: Some(vec![0, 1]),
                    strings: None,
                },
                coefficients: vec![
                    0.5, -0.3, // class 0
                    -0.5, 0.8, // class 1
                ],
                intercepts: Some(vec![0.1, -0.1]),
                multi_class: None,
                post_transform: None,
            },
        }
    }
}

impl ToOnnx for TinyClassifier {
    fn to_graph(
        &self,
        builder: &mut GraphBuilder,
    ) -> Result<ValueRef, serde_onnx::export::ExportError> {
        // Input: float matrix [N, 2].
        let input_ty = ValueType::tensor(ElemType::Float, Some(vec![Dim::Unknown, Dim::Fixed(2)]));
        let x = builder.input("X", input_ty)?;

        // LinearClassifier has 2 OUTPUTS: ask `emit_op` for 2.
        let mut outs = builder.emit_op(&self.op, vec![x], 2)?;
        let z = outs.pop().expect("second output (scores)");
        let y = outs.pop().expect("first output (labels)");

        // Each output must be declared: Y = int64 [N], Z = float [N, 2].
        builder.output(
            y.name().to_string(),
            ValueType::tensor(ElemType::Int64, Some(vec![Dim::Unknown])),
        )?;
        builder.output(
            z.name().to_string(),
            ValueType::tensor(ElemType::Float, Some(vec![Dim::Unknown, Dim::Fixed(2)])),
        )?;
        Ok(y)
    }
}

/// Teaching visitor that counts and describes each imported node.
#[derive(Default)]
struct Describer {
    typed: usize,
    raw: usize,
}

impl PayloadVisitor for Describer {
    fn visit_typed(&mut self, index: usize, payload: &NodePayload) {
        self.typed += 1;
        let node = payload.node();
        println!(
            "  [{index}] TYPED {} :: {:?} (recognized)",
            node.op_type, node.domain
        );
    }
    fn visit_raw(&mut self, index: usize, node: &Node) {
        self.raw += 1;
        println!(
            "  [{index}] RAW   {} :: {:?} (preserved untyped)",
            node.op_type, node.domain
        );
    }
}

fn main() {
    // STAGE 1 — Export the in-memory model.
    let model = export_model(&TinyClassifier::new(), "tiny_classifier").expect("export");
    println!("export: {} node(s)", model.graph.nodes.len());

    // STAGE 2 — Serialize and save the `.onnx` file to disk.
    let bytes = encode_model(&model);
    let path = std::env::temp_dir().join("serde_onnx_linear_classifier.onnx");
    std::fs::write(&path, &bytes).expect("write .onnx");
    println!("saved to {} ({} bytes)", path.display(), bytes.len());

    // STAGE 3 — Re-import from disk, as a model consumer would.
    let decoded = DecodedModel::decode_file(&path).expect("decode_file");
    println!("import: graph {:?}, opsets:", decoded.graph_name());
    for opset in decoded.opset_import() {
        println!("  {:?} v{}", opset.domain, opset.version);
    }
    println!(
        "inputs: {:?}",
        decoded.inputs().iter().map(|i| &i.name).collect::<Vec<_>>()
    );
    println!(
        "outputs: {:?}",
        decoded
            .outputs()
            .iter()
            .map(|o| &o.name)
            .collect::<Vec<_>>()
    );

    // STAGE 4 — Visit the payloads: the classifier must come back typed.
    let mut describer = Describer::default();
    decoded.visit(&mut describer);
    assert_eq!(describer.typed, 1, "expected 1 typed node");
    assert_eq!(describer.raw, 0, "expected no Raw nodes");
    assert!(decoded.warnings.is_empty(), "expected no warnings");

    match &decoded.payloads[0] {
        NodePayload::LinearClassifier(typed) => {
            println!("coefficients: {:?}", typed.op.coefficients);
            println!("labels: {:?}", typed.op.classlabels.ints);
        }
        other => panic!("expected LinearClassifier, got {}", other.node().op_type),
    }
    println!("export → file → import cycle: OK");
}
