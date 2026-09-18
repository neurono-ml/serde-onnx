use serde_onnx::ir::{
    Attribute, Dim, ElemType, ExternalData, Graph, ML_DOMAIN, Model, Node, OpsetId, Tensor,
    TensorData, TensorStorage, ValueInfo, ValueType,
};
use serde_onnx::proto::{
    ProtoCodecError, decode_model, decode_model_proto, encode_model, read_external,
};

fn scaler_model() -> Model {
    let mut graph = Graph::new("main_graph");
    graph.inputs.push(ValueInfo::new(
        "x",
        ValueType::tensor(
            ElemType::Float,
            Some(vec![Dim::Param("N".into()), Dim::Fixed(3)]),
        ),
    ));
    graph.initializers.push(Tensor::new(
        "w",
        ElemType::Float,
        vec![Dim::Fixed(3)],
        TensorData::F32(vec![1.5, 2.0, -0.5]),
    ));
    graph.nodes.push(Node::new(
        "Scaler",
        ML_DOMAIN,
        vec!["x".into(), "w".into()],
        vec!["y".into()],
        vec![
            Attribute::floats("scale", vec![2.0, 2.0, 2.0]),
            Attribute::int("axis", 1),
        ],
    ));
    graph.outputs.push(ValueInfo::new(
        "y",
        ValueType::tensor(
            ElemType::Float,
            Some(vec![Dim::Param("N".into()), Dim::Fixed(3)]),
        ),
    ));
    Model::new(
        graph,
        vec![
            OpsetId {
                domain: String::new(),
                version: 21,
            },
            OpsetId {
                domain: ML_DOMAIN.into(),
                version: 5,
            },
        ],
    )
}

#[test]
fn round_trip_scaler_model_is_semantically_identical() {
    let model = scaler_model();
    let bytes = encode_model(&model);
    let back = decode_model(&bytes).expect("decode must succeed");
    assert_eq!(back, model);

    let graph = &back.graph;
    assert_eq!(graph.name, "main_graph");
    assert_eq!(graph.nodes.len(), 1);
    let node = &graph.nodes[0];
    assert_eq!(node.op_type, "Scaler");
    assert_eq!(node.domain, ML_DOMAIN);
    assert_eq!(node.inputs, vec!["x".to_string(), "w".to_string()]);
    assert_eq!(node.outputs, vec!["y".to_string()]);
    assert_eq!(
        node.attr("axis").map(|a| &a.value),
        Some(&serde_onnx::ir::AttributeValue::Int(1))
    );

    let init = graph.initializer("w").unwrap();
    assert_eq!(init.data().unwrap(), &TensorData::F32(vec![1.5, 2.0, -0.5]));
    assert_eq!(init.shape, vec![Dim::Fixed(3)]);

    let input = &graph.inputs[0];
    match input.value_type.as_ref().unwrap() {
        ValueType::Tensor(t) => {
            assert_eq!(t.elem, ElemType::Float);
            assert_eq!(
                t.shape.as_ref().unwrap(),
                &vec![Dim::Param("N".into()), Dim::Fixed(3)]
            );
        }
        other => panic!("expected tensor type, got {other:?}"),
    }
}

#[test]
fn round_trip_preserves_strings_and_int64_attr() {
    let mut graph = Graph::new("strings_graph");
    graph.initializers.push(Tensor::new(
        "labels",
        ElemType::String,
        vec![Dim::Fixed(2)],
        TensorData::String(vec!["alpha".into(), "beta".into()]),
    ));
    graph.nodes.push(Node::new(
        "Identity",
        "",
        vec!["labels".into()],
        vec!["out".into()],
        vec![Attribute::ints("axes", vec![0, 1, 2])],
    ));
    graph.outputs.push(ValueInfo::new(
        "out",
        ValueType::tensor(ElemType::String, Some(vec![Dim::Fixed(2)])),
    ));
    let model = Model::new(
        graph,
        vec![OpsetId {
            domain: String::new(),
            version: 21,
        }],
    );
    let back = decode_model(&encode_model(&model)).expect("decode");
    assert_eq!(back, model);
    assert_eq!(
        back.graph.initializer("labels").unwrap().data().unwrap(),
        &TensorData::String(vec!["alpha".into(), "beta".into()])
    );
}

#[test]
fn round_trip_f16_tensor_keeps_bit_precision() {
    let bits: Vec<u16> = [1.5f32, -2.25, 0.0, f32::INFINITY]
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    let mut graph = Graph::new("f16_graph");
    graph.initializers.push(Tensor::new(
        "h",
        ElemType::Float16,
        vec![Dim::Fixed(4)],
        TensorData::F16(bits.clone()),
    ));
    graph.nodes.push(Node::new(
        "Identity",
        "",
        vec!["h".into()],
        vec!["y".into()],
        vec![],
    ));
    graph.outputs.push(ValueInfo::new(
        "h",
        ValueType::tensor(ElemType::Float16, Some(vec![Dim::Fixed(4)])),
    ));
    let model = Model::new(
        graph,
        vec![OpsetId {
            domain: String::new(),
            version: 21,
        }],
    );
    let back = decode_model(&encode_model(&model)).expect("decode");
    assert_eq!(back, model);
    assert_eq!(
        back.graph.initializer("h").unwrap().f16_bits(),
        Some(bits.as_slice())
    );
}

#[test]
fn external_data_reference_preserved_and_readable() {
    let dir = std::env::temp_dir().join(format!("serde_onnx_codec_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create base dir");
    std::fs::write(dir.join("weights.bin"), payload_bytes(&[1.0, 2.5])).unwrap();

    let mut graph = Graph::new("ext_graph");
    graph.initializers.push(Tensor::external(
        "w",
        ElemType::Double,
        vec![Dim::Fixed(2)],
        ExternalData {
            location: "weights.bin".into(),
            offset: Some(0),
            length: Some(16),
            checksum: None,
        },
    ));
    graph.nodes.push(Node::new(
        "Identity",
        "",
        vec!["w".into()],
        vec!["y".into()],
        vec![],
    ));
    graph.outputs.push(ValueInfo::new(
        "w",
        ValueType::tensor(ElemType::Double, Some(vec![Dim::Fixed(2)])),
    ));
    let model = Model::new(
        graph,
        vec![OpsetId {
            domain: String::new(),
            version: 21,
        }],
    );
    let bytes = encode_model(&model);

    let back = decode_model(&bytes).expect("decode");
    assert_eq!(back, model);
    assert!(matches!(
        back.graph.initializers[0].data,
        TensorStorage::External(_)
    ));

    let loaded = read_external(&back.graph.initializers[0], &dir).expect("read external");
    assert_eq!(loaded.data().unwrap(), &TensorData::F64(vec![1.0, 2.5]));

    std::fs::remove_dir_all(&dir).ok();
}

fn payload_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn external_data_with_offset_reads_slice() {
    let dir = std::env::temp_dir().join(format!("serde_onnx_codec_off_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create base dir");
    let mut blob = vec![0xFFu8; 4096];
    blob.extend(payload_bytes(&[7.0f64, 8.0f64]));
    std::fs::write(dir.join("blob.bin"), &blob).unwrap();

    let tensor = Tensor::external(
        "w",
        ElemType::Double,
        vec![Dim::Fixed(2)],
        ExternalData {
            location: "blob.bin".into(),
            offset: Some(4096),
            length: Some(16),
            checksum: None,
        },
    );
    let loaded = read_external(&tensor, &dir).expect("read external");
    assert_eq!(loaded.data().unwrap(), &TensorData::F64(vec![7.0, 8.0]));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn external_data_rejects_path_escape_and_absolute() {
    let dir = std::env::temp_dir().join(format!("serde_onnx_codec_esc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create base dir");
    let tensor = Tensor::external(
        "w",
        ElemType::Double,
        vec![Dim::Fixed(1)],
        ExternalData {
            location: "../outside.bin".into(),
            offset: None,
            length: None,
            checksum: None,
        },
    );
    let err = read_external(&tensor, &dir).unwrap_err();
    assert!(matches!(err, ProtoCodecError::InvalidExternalData { .. }));

    let absolute = Tensor::external(
        "w",
        ElemType::Double,
        vec![Dim::Fixed(2)],
        ExternalData {
            location: "/etc/passwd".into(),
            offset: None,
            length: None,
            checksum: None,
        },
    );
    let err = read_external(&absolute, &dir).unwrap_err();
    assert!(matches!(err, ProtoCodecError::InvalidExternalData { .. }));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn corrupted_bytes_error_without_panic() {
    let cases: Vec<Vec<u8>> = vec![
        vec![],
        vec![0xFF, 0xFF, 0xFF, 0xFF, 0x01],
        vec![0x0A, 0x80],
        {
            let mut b = encode_model(&scaler_model());
            b[1] = 0xFE;
            b
        },
    ];
    for bytes in cases {
        let err = decode_model(&bytes);
        assert!(err.is_err(), "expected error for bytes {bytes:?}");
    }
}

#[test]
fn tensor_size_divergence_is_reported() {
    use serde_onnx::proto::{GraphProto, ModelProto, TensorProto, tensor_proto::DataType};

    let proto = ModelProto {
        ir_version: 10,
        graph: Some(GraphProto {
            name: "g".into(),
            initializer: vec![TensorProto {
                name: "t".into(),
                data_type: DataType::Float as i32,
                dims: vec![10],
                raw_data: vec![0u8; 37],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    match decode_model_proto(&proto) {
        Err(ProtoCodecError::TensorSizeMismatch {
            name,
            expected,
            got,
        }) => {
            assert_eq!(name, "t");
            assert_eq!((expected, got), (10, 37));
        }
        other => panic!("expected TensorSizeMismatch, got {other:?}"),
    }

    let proto = ModelProto {
        ir_version: 10,
        graph: Some(GraphProto {
            name: "g".into(),
            initializer: vec![TensorProto {
                name: "t".into(),
                data_type: DataType::Double as i32,
                dims: vec![2, 2],
                double_data: vec![1.0],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    match decode_model_proto(&proto) {
        Err(ProtoCodecError::TensorSizeMismatch { expected, got, .. }) => {
            assert_eq!((expected, got), (4, 1));
        }
        other => panic!("expected TensorSizeMismatch, got {other:?}"),
    }
}

#[test]
fn unsupported_ir_and_opset_versions_error_with_versions_named() {
    let mut model = scaler_model();
    model.ir_version = 11;
    let bytes = encode_model(&model);
    match decode_model(&bytes) {
        Err(ProtoCodecError::UnsupportedIrVersion {
            declared,
            supported,
        }) => {
            assert_eq!((declared, supported), (11, 10));
        }
        other => panic!("expected UnsupportedIrVersion, got {other:?}"),
    }

    let mut model = scaler_model();
    model.opset_import[0].version = 26;
    let bytes = encode_model(&model);
    match decode_model(&bytes) {
        Err(ProtoCodecError::UnsupportedOpsetVersion {
            domain,
            declared,
            supported,
        }) => {
            assert_eq!(domain, "ai.onnx");
            assert_eq!((declared, supported), (26, 25));
        }
        other => panic!("expected UnsupportedOpsetVersion, got {other:?}"),
    }

    let mut model = scaler_model();
    model.opset_import[1].version = 6;
    let bytes = encode_model(&model);
    match decode_model(&bytes) {
        Err(ProtoCodecError::UnsupportedOpsetVersion {
            domain,
            declared,
            supported,
        }) => {
            assert_eq!(domain, "ai.onnx.ml");
            assert_eq!((declared, supported), (6, 5));
        }
        other => panic!("expected UnsupportedOpsetVersion, got {other:?}"),
    }
}

#[test]
fn unknown_opset_domain_is_allowed() {
    let mut graph = Graph::new("custom");
    graph.outputs.push(ValueInfo::new(
        "y",
        ValueType::tensor(ElemType::Float, Some(vec![Dim::Fixed(1)])),
    ));
    let model = Model::new(
        graph,
        vec![OpsetId {
            domain: "com.example.custom".into(),
            version: 999,
        }],
    );
    let bytes = encode_model(&model);
    let back = decode_model(&bytes).expect("unknown domains stay raw and allowed");
    assert_eq!(back, model);
}
