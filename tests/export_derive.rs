#![cfg(feature = "export")]

use serde_onnx::OnnxExport;
use serde_onnx::export::ToOnnx;
use serde_onnx::export::ValueRef;
use serde_onnx::export::export_model;
use serde_onnx::ir::Attribute;
use serde_onnx::ir::Dim;
use serde_onnx::ir::ElemType;
use serde_onnx::ir::ValueType;

fn io_type() -> ValueType {
    ValueType::tensor(ElemType::Float, Some(vec![Dim::Unknown, Dim::Unknown]))
}

#[derive(OnnxExport)]
#[onnx(op = "Scaler", domain = "ai.onnx.ml")]
#[allow(dead_code)]
struct DerivedScaler {
    #[onnx(attr = "offset")]
    offset: Option<Vec<f32>>,
    #[onnx(attr = "scale")]
    scale: Option<Vec<f32>>,
    unexported: String,
}

struct ManualScaler {
    offset: Option<Vec<f32>>,
    scale: Option<Vec<f32>>,
}

impl ToOnnx for ManualScaler {
    fn to_graph(
        &self,
        builder: &mut serde_onnx::export::GraphBuilder,
    ) -> Result<ValueRef, serde_onnx::export::ExportError> {
        let io = io_type();
        let input = builder.input("X", io.clone())?;
        let mut attributes: Vec<Attribute> = Vec::new();
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

#[test]
fn derived_scaler_matches_manual_impl() {
    let derived = DerivedScaler {
        offset: Some(vec![1.0, 2.0]),
        scale: Some(vec![0.5, 0.25]),
        unexported: "ignored".to_string(),
    };
    let manual = ManualScaler {
        offset: Some(vec![1.0, 2.0]),
        scale: Some(vec![0.5, 0.25]),
    };
    let derived_model = export_model(&derived, "g").expect("derived export");
    let manual_model = export_model(&manual, "g").expect("manual export");
    assert_eq!(derived_model.graph, manual_model.graph);
    assert_eq!(derived_model.opset_import, manual_model.opset_import);
    assert_eq!(derived_model.graph.nodes.len(), 1);
    let node = &derived_model.graph.nodes[0];
    assert_eq!(node.op_type, "Scaler");
    assert_eq!(node.domain, "ai.onnx.ml");
    assert_eq!(node.outputs, vec!["node_0_Scaler_out".to_string()]);
    let payload = serde_onnx::ml::make_node_payload(node);
    assert!(matches!(payload, serde_onnx::ml::NodePayload::Scaler(_)));
}

#[test]
fn derived_scaler_with_empty_options_matches_manual() {
    let derived = DerivedScaler {
        offset: None,
        scale: None,
        unexported: String::new(),
    };
    let manual = ManualScaler {
        offset: None,
        scale: None,
    };
    let derived_model = export_model(&derived, "g").expect("derived export");
    let manual_model = export_model(&manual, "g").expect("manual export");
    assert_eq!(derived_model.graph, manual_model.graph);
    assert!(derived_model.graph.nodes[0].attributes.is_empty());
}
