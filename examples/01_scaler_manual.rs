//! # Example 01 — Export WITHOUT macros (manual `ToOnnx` implementation)
//!
//! ## What this example teaches
//!
//! 1. How to model a "fitted" model (here, a scikit-learn-style `StandardScaler`).
//! 2. How to implement the [`ToOnnx`](serde_onnx::export::ToOnnx) trait by hand.
//! 3. How to build the graph with [`GraphBuilder`](serde_onnx::export::GraphBuilder):
//!    declare an `input`, emit a typed op with `emit_op`, declare an `output`.
//! 4. How to validate + serialize with [`export_model`](serde_onnx::export::export_model)
//!    and do the protobuf round-trip (`encode_model` / `decode_model`).
//!
//! ## When to use this approach
//!
//! Use the manual implementation when you need custom logic in `to_graph` —
//! e.g. deriving attributes (`scale = 1/std`), emitting several chained nodes,
//! or creating initializers. If your case is a simple struct with
//! 1 input → 1 op → 1 output, prefer the `#[derive(OnnxExport)]` macro
//! (see example `02_scaler_derive`).
//!
//! ## How to run
//!
//! ```sh
//! cargo run --example 01_scaler_manual --features export
//! ```

use serde_onnx::export::{GraphBuilder, ToOnnx, ValueRef, export_model};
use serde_onnx::ir::{Dim, ElemType, ValueType};
use serde_onnx::ml::Scaler;

// ---------------------------------------------------------------------------
// STEP 1 — Model your trained model as a plain Rust struct.
// ---------------------------------------------------------------------------
// Nothing here is ONNX-specific: just the parameters learned during training
// (per-column mean and standard deviation, like sklearn's `StandardScaler`).
/// A fake "StandardScaler", already fitted on 3 columns.
pub struct FittedStandardScaler {
    /// Mean of each column (ONNX `offset` attribute).
    pub mean: Vec<f32>,
    /// Standard deviation of each column (used to derive `scale = 1/std`).
    pub std: Vec<f32>,
}

impl FittedStandardScaler {
    /// The ONNX `Scaler` op applies `Y = (X - offset) * scale`, so we convert
    /// the standard deviation into the matching scale factor.
    fn scale(&self) -> Vec<f32> {
        self.std.iter().map(|v| 1.0 / v).collect()
    }
}

// ---------------------------------------------------------------------------
// STEP 2 — Implement `ToOnnx` by hand.
// ---------------------------------------------------------------------------
// The contract is simple: given a `GraphBuilder`, declare inputs, emit nodes,
// declare outputs, and return the `ValueRef` of the main output.
impl ToOnnx for FittedStandardScaler {
    fn to_graph(
        &self,
        builder: &mut GraphBuilder,
    ) -> Result<ValueRef, serde_onnx::export::ExportError> {
        // 2a. Describe the input/output type: float tensor `[N, 3]`
        //     with a dynamic batch (`Dim::Unknown` = unknown dimension).
        let io = ValueType::tensor(ElemType::Float, Some(vec![Dim::Unknown, Dim::Fixed(3)]));

        // 2b. Declare the graph input under the name "X".
        let input = builder.input("X", io.clone())?;

        // 2c. Build the typed op. `Scaler` already knows how to produce the
        //     right `Attribute`s (`offset`, `scale`) via `OnnxOp::to_node`.
        let op = Scaler {
            offset: Some(self.mean.clone()),
            scale: Some(self.scale()),
        };

        // 2d. Emit the op into the graph. `emit_op` creates the `Node`,
        //     registers the `ai.onnx.ml` opset automatically, and generates
        //     deterministic output names (`node_0_Scaler_out`).
        let mut outputs = builder.emit_op(&op, vec![input], 1)?;
        let output = outputs.pop().expect("Scaler has exactly 1 output");

        // 2e. Declare the graph output, reusing the generated name.
        builder.output(output.name().to_string(), io)?;

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// STEP 3 — Export, serialize, and verify the round-trip.
// ---------------------------------------------------------------------------
fn main() {
    // 3a. Build the "trained" model.
    let fitted = FittedStandardScaler {
        mean: vec![1.0, 2.0, 3.0],
        std: vec![0.5, 1.0, 2.0],
    };

    // 3b. `export_model` runs `to_graph` and then VALIDATES the graph
    //     (SSA, defined values, topological order). If anything is
    //     inconsistent you get `ExportError::Validation` here —
    //     before a single byte is written.
    let model = export_model(&fitted, "standard_scaler").expect("export must succeed");

    // 3c. Inspect the result: 1 node in the `ai.onnx.ml` domain.
    println!("graph: {}", model.graph.name);
    println!("nodes: {}", model.graph.nodes.len());
    for node in &model.graph.nodes {
        println!(
            "  node {} (domain {:?}): [{}] -> [{}]",
            node.op_type,
            node.domain,
            node.inputs.join(", "),
            node.outputs.join(", "),
        );
        for attr in &node.attributes {
            println!("    attribute: {}", attr.name);
        }
    }
    for opset in &model.opset_import {
        println!("  opset {:?} version {}", opset.domain, opset.version);
    }

    // 3d. Serialize to the protobuf `.onnx` format.
    let bytes = serde_onnx::proto::encode_model(&model);
    println!("encoded bytes: {}", bytes.len());

    // 3e. Prove the round-trip: decoding must return an identical graph.
    let decoded = serde_onnx::proto::decode_model(&bytes).expect("round-trip");
    assert_eq!(decoded.graph.nodes, model.graph.nodes);
    println!("protobuf round-trip: OK");

    // TIP: save to disk with `std::fs::write("scaler.onnx", bytes)` and open
    // the file in Netron (https://netron.app) to visualize the graph.
}
