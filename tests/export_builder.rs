#![cfg(feature = "export")]

use serde_onnx::export::ExportError;
use serde_onnx::export::GraphBuilder;
use serde_onnx::export::ToOnnx;
use serde_onnx::export::ValueRef;
use serde_onnx::export::export_model;
use serde_onnx::ir::Attribute;
use serde_onnx::ir::Dim;
use serde_onnx::ir::ElemType;
use serde_onnx::ir::ValueType;
use serde_onnx::ml::CORE_TYPED_OPSET_TARGET;
use serde_onnx::ml::ML_EXPORT_OPSET_TARGET;
use serde_onnx::ml::OnnxOp;

fn matrix() -> ValueType {
    ValueType::tensor(ElemType::Float, Some(vec![Dim::Unknown, Dim::Unknown]))
}

#[test]
fn initializer_infers_dtype_from_scalar() {
    let mut builder = GraphBuilder::new("g");
    builder
        .initializer("w", vec![Dim::Fixed(2)], vec![1.0f32, 2.0])
        .expect("initializer");
    builder
        .initializer("ids", vec![Dim::Fixed(3)], vec![1i64, 2, 3])
        .expect("initializer");
    let (graph, _) = builder.finish();
    assert_eq!(graph.initializers[0].elem, ElemType::Float);
    assert_eq!(graph.initializers[1].elem, ElemType::Int64);
}

#[test]
fn duplicate_names_are_rejected() {
    let mut builder = GraphBuilder::new("g");
    builder.input("X", matrix()).expect("input");
    let err = builder.input("X", matrix()).expect_err("duplicate input");
    assert!(matches!(err, ExportError::DuplicateName(_)));
    let err = builder
        .initializer("X", vec![Dim::Fixed(1)], vec![1.0f32])
        .expect_err("initializer collides with input");
    assert!(matches!(err, ExportError::DuplicateName(_)));
}

#[test]
fn auto_named_outputs_collide_with_existing_names() {
    let mut builder = GraphBuilder::new("g");
    builder.input("X", matrix()).expect("input");
    builder
        .input("node_0_Identity_out", matrix())
        .expect("input");
    let err = builder
        .push_node("Identity", "", vec!["X".to_string()], Vec::new(), 1)
        .expect_err("auto name collides");
    assert!(matches!(err, ExportError::DuplicateName(_)));
}

#[test]
fn opset_imports_register_per_domain() {
    let mut builder = GraphBuilder::new("g");
    let input = builder.input("X", matrix()).expect("input");
    builder
        .push_node(
            "Scaler",
            "ai.onnx.ml",
            vec![input.name().to_string()],
            vec![Attribute::floats("scale", vec![1.0])],
            1,
        )
        .expect("ml node");
    builder
        .push_node(
            "Identity",
            "",
            vec!["node_0_Scaler_out".to_string()],
            Vec::new(),
            1,
        )
        .expect("core node");
    let opsets = builder.opset_import();
    let ml = opsets
        .iter()
        .find(|o| o.domain == "ai.onnx.ml")
        .expect("ml opset");
    assert_eq!(ml.version, ML_EXPORT_OPSET_TARGET);
    let core = opsets
        .iter()
        .find(|o| o.domain.is_empty())
        .expect("core opset");
    assert_eq!(core.version, CORE_TYPED_OPSET_TARGET);
}

#[test]
fn emit_op_uses_deterministic_auto_names() {
    let mut builder = GraphBuilder::new("g");
    let input = builder.input("X", matrix()).expect("input");
    let op = serde_onnx::ml::Scaler {
        offset: None,
        scale: Some(vec![2.0]),
    };
    let outputs = builder.emit_op(&op, vec![input], 1).expect("emit");
    assert_eq!(outputs[0].name(), "node_0_Scaler_out");
}

#[test]
fn sink_emit_registers_opset_and_initializers() {
    use serde_onnx::ml::OpSink;
    let mut builder = GraphBuilder::new("g");
    let op = serde_onnx::ml::Scaler {
        offset: None,
        scale: Some(vec![2.0]),
    };
    let emitted = op
        .to_node(vec!["X".to_string()], vec!["Y".to_string()])
        .expect("node");
    builder.input("X", matrix()).expect("input");
    let (inputs, outputs) = builder.emit(emitted).expect("sink emit");
    assert_eq!(inputs, vec!["X".to_string()]);
    assert_eq!(outputs, vec!["Y".to_string()]);
    let opsets = builder.opset_import();
    assert!(opsets.iter().any(|o| o.domain == "ai.onnx.ml"));
}

struct Dangling;

impl ToOnnx for Dangling {
    fn to_graph(&self, builder: &mut GraphBuilder) -> Result<ValueRef, ExportError> {
        let outputs =
            builder.push_node("Identity", "", vec!["missing".to_string()], Vec::new(), 1)?;
        let output = outputs.into_iter().next().expect("single output");
        builder.output(output.name().to_string(), matrix())?;
        Ok(output)
    }
}

#[test]
fn export_model_rejects_invalid_graph_with_aggregated_errors() {
    let err = export_model(&Dangling, "g").expect_err("undefined value");
    match err {
        ExportError::Validation(errors) => assert!(!errors.is_empty()),
        other => panic!("expected validation errors, got {other:?}"),
    }
}
