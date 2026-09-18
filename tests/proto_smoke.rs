use prost::Message;
use serde_onnx::proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto, Version, attribute_proto::AttributeType,
    tensor_proto::DataType, tensor_shape_proto, type_proto,
};

#[test]
fn core_protos_exist() {
    let mut m = ModelProto {
        ir_version: Version::IrVersion2024325 as i64,
        graph: Some(GraphProto::default()),
        ..Default::default()
    };
    let graph = GraphProto {
        node: vec![NodeProto {
            op_type: "Scaler".into(),
            domain: "ai.onnx.ml".into(),
            input: vec!["x".into()],
            output: vec!["y".into()],
            attribute: vec![AttributeProto {
                name: "scale".into(),
                r#type: AttributeType::Floats as i32,
                floats: vec![1.0],
                ..Default::default()
            }],
            ..Default::default()
        }],
        initializer: vec![TensorProto {
            data_type: DataType::Uint2 as i32,
            raw_data: vec![0b11100100],
            dims: vec![4],
            ..Default::default()
        }],
        output: vec![ValueInfoProto {
            name: "y".into(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: DataType::Float as i32,
                    shape: Some(TensorShapeProto {
                        dim: vec![tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(1)),
                            denotation: String::new(),
                        }],
                    }),
                })),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    m.graph = Some(graph);
    assert_eq!(DataType::Uint2 as i32, 25);
    assert_eq!(DataType::Int2 as i32, 26);
    assert_eq!(m.graph.as_ref().unwrap().node.len(), 1);
}

#[test]
fn roundtrip_minimal() {
    let m = ModelProto {
        ir_version: Version::IrVersion2024325 as i64,
        graph: Some(GraphProto {
            name: "main_graph".into(),
            ..Default::default()
        }),
        producer_name: "serde-onnx".into(),
        opset_import: vec![OperatorSetIdProto {
            domain: "ai.onnx.ml".into(),
            version: 5,
        }],
        ..Default::default()
    };
    let bytes = m.encode_to_vec();
    let back = ModelProto::decode(bytes.as_slice()).unwrap();
    let graph = back.graph.unwrap();
    assert_eq!(graph.name, "main_graph");
    assert_eq!(back.producer_name, "serde-onnx");
    assert_eq!(back.opset_import[0].domain, "ai.onnx.ml");
    assert_eq!(back.opset_import[0].version, 5);
}

#[test]
fn onnx_ml_package_reexported() {
    let _ = serde_onnx::proto::tensor_proto::DataType::Uint2;
    let _ = TypeProto::default();
}
