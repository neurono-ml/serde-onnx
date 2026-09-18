use serde_onnx::export::GraphBuilder;
use serde_onnx::export::ToOnnx;
use serde_onnx::export::ValueRef;
use serde_onnx::export::export_model;
use serde_onnx::ir::Dim;
use serde_onnx::ir::ElemType;
use serde_onnx::ir::ValueType;
use serde_onnx::ml::Scaler;

pub struct SKStandardScalerModel {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl SKStandardScalerModel {
    pub fn scale(&self) -> Vec<f32> {
        self.std.iter().map(|v| 1.0 / v).collect()
    }
}

impl ToOnnx for SKStandardScalerModel {
    fn to_graph(
        &self,
        builder: &mut GraphBuilder,
    ) -> Result<ValueRef, serde_onnx::export::ExportError> {
        let io = ValueType::tensor(ElemType::Float, Some(vec![Dim::Unknown, Dim::Unknown]));
        let input = builder.input("X", io.clone())?;
        let op = Scaler {
            offset: Some(self.mean.clone()),
            scale: Some(self.scale()),
        };
        let mut outputs = builder.emit_op(&op, vec![input], 1)?;
        let output = outputs.pop().expect("single output");
        builder.output(output.name().to_string(), io)?;
        Ok(output)
    }
}

fn main() {
    let fitted = SKStandardScalerModel {
        mean: vec![1.0, 2.0, 3.0],
        std: vec![0.5, 1.0, 2.0],
    };
    let model = export_model(&fitted, "standard_scaler").expect("export succeeds");
    println!("graph: {}", model.graph.name);
    println!("nodes: {}", model.graph.nodes.len());
    for node in &model.graph.nodes {
        println!(
            "node {} in domain {:?}: {} -> {}",
            node.op_type,
            node.domain,
            node.inputs.join(", "),
            node.outputs.join(", ")
        );
        for attr in &node.attributes {
            println!("  attr {}", attr.name);
        }
    }
    for opset in &model.opset_import {
        println!("opset {:?} version {}", opset.domain, opset.version);
    }
    let bytes = serde_onnx::proto::encode_model(&model);
    println!("encoded bytes: {}", bytes.len());
    let decoded = serde_onnx::proto::decode_model(&bytes).expect("round trip");
    assert_eq!(decoded.graph.nodes, model.graph.nodes);
    println!("round trip through protobuf: ok");
}
