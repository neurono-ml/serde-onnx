//! # Example 03 — Multi-node pipeline WITHOUT macros (`Imputer` → `Scaler` + initializer)
//!
//! ## What this example teaches
//!
//! 1. How to chain **several ops** in one graph (`Imputer` → `Scaler`).
//! 2. How to mix **typed** ops (`emit_op` with `OnnxOp`) — with attribute
//!    validation at build time — instead of the raw `push_node`.
//! 3. How to add an **initializer** (a constant embedded in the `.onnx`).
//! 4. How one node's output becomes the next node's input via `ValueRef`.
//!
//! ## Pipeline diagram
//!
//! ```text
//!   X [N, 3] ──▶ Imputer ──▶ node_0_Imputer_out ──▶ Scaler ──▶ Y [N, 3]
//!
//!   shape_hint [2] (int64 initializer, embedded but not wired to any node)
//! ```
//!
//! ## How to run
//!
//! ```sh
//! cargo run --example 03_pipeline_imputer_scaler --features export
//! ```

use serde_onnx::export::{GraphBuilder, ToOnnx, ValueRef, export_model};
use serde_onnx::ir::{Dim, ElemType, ValueType};
use serde_onnx::ml::{Imputer, Scaler};

/// "Clean NaNs and normalize" pipeline: the equivalent of chaining
/// sklearn's `SimpleImputer` + `StandardScaler`.
pub struct CleaningPipeline {
    /// Value replacing NaNs (`replaced_value_float` attribute).
    pub nan_replacement: f32,
    /// Per-column means (`offset` attribute of Scaler).
    pub mean: Vec<f32>,
    /// Per-column scale factors (`scale` attribute of Scaler).
    pub scale: Vec<f32>,
}

fn matrix_3cols() -> ValueType {
    ValueType::tensor(ElemType::Float, Some(vec![Dim::Unknown, Dim::Fixed(3)]))
}

impl ToOnnx for CleaningPipeline {
    fn to_graph(
        &self,
        builder: &mut GraphBuilder,
    ) -> Result<ValueRef, serde_onnx::export::ExportError> {
        let io = matrix_3cols();

        // STEP 1 — Graph input.
        let input = builder.input("X", io.clone())?;

        // STEP 2 — Node 1: Imputer. Replaces NaN with `nan_replacement`.
        // `emit_op` validates attributes eagerly (e.g. only one of the
        // `imputed_*` pair may be set) and registers the `ai.onnx.ml` opset.
        let imputer = Imputer {
            imputed_value_floats: None,
            imputed_value_int64s: None,
            replaced_value_float: Some(self.nan_replacement),
            replaced_value_int64: None,
        };
        let imputed = builder
            .emit_op(&imputer, vec![input], 1)?
            .pop()
            .expect("Imputer has 1 output");

        // STEP 3 — Node 2: Scaler, consuming the Imputer output.
        // Note how we pass `imputed` (a `ValueRef`) as the input —
        // this is how nodes are chained: the previous output name becomes
        // the next input name.
        let scaler = Scaler {
            offset: Some(self.mean.clone()),
            scale: Some(self.scale.clone()),
        };
        let scaled = builder
            .emit_op(&scaler, vec![imputed], 1)?
            .pop()
            .expect("Scaler has 1 output");

        // STEP 4 — Example initializer: a constant int64 tensor embedded in
        // the model. Initializers are handy for weights, shapes, frozen
        // vocabularies, etc. (`Scalar` infers the dtype from the Rust type.)
        let _shape_const: ValueRef =
            builder.initializer("shape_hint", vec![Dim::Fixed(2)], vec![0i64, 3i64])?;
        // (`shape_hint` is not wired to any node — it just travels embedded
        // in `graph.initializers` inside the `.onnx` file.)

        // STEP 5 — Graph output = last node's output.
        builder.output(scaled.name().to_string(), io)?;
        Ok(scaled)
    }
}

fn main() {
    let pipeline = CleaningPipeline {
        // NOTE: avoid `0.0` here — protobuf (proto3) drops singular floats
        // holding the default zero value on the wire, and the round-trip
        // would read the attribute back as missing.
        nan_replacement: -1.0,
        mean: vec![1.0, 2.0, 3.0],
        scale: vec![2.0, 1.0, 0.5],
    };

    // `export_model` validates the whole graph (SSA + topological order),
    // so wiring mistakes surface here as aggregated errors.
    let model = export_model(&pipeline, "cleaning_pipeline").expect("export");

    println!("graph: {}", model.graph.name);
    println!("nodes: {}", model.graph.nodes.len());
    for (i, node) in model.graph.nodes.iter().enumerate() {
        println!(
            "  [{i}] {} :: {:?}: [{}] -> [{}]",
            node.op_type,
            node.domain,
            node.inputs.join(", "),
            node.outputs.join(", "),
        );
    }
    println!("initializers: {}", model.graph.initializers.len());
    for init in &model.graph.initializers {
        println!("  {}: {:?} {:?}", init.name, init.elem, init.shape);
    }

    // Protobuf round-trip to prove everything serializes.
    let bytes = serde_onnx::proto::encode_model(&model);
    let decoded = serde_onnx::proto::decode_model(&bytes).expect("round-trip");
    assert_eq!(decoded.graph.nodes, model.graph.nodes);
    println!("pipeline with {} bytes, round-trip: OK", bytes.len());

    // SUGGESTED next steps:
    // - Append a `Normalizer` after the Scaler (3 chained nodes).
    // - Could this pipeline use `#[derive(OnnxExport)]`? No: the derive
    //   covers 1 op per struct — pipelines need a manual `ToOnnx`.
}
