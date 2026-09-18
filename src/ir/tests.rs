use super::*;

fn fake_graph_proto(seed: u8) -> Vec<u8> {
    vec![seed, seed.wrapping_add(1), seed.wrapping_add(2), 0xAB, 0xCD]
}

#[test]
fn minimal_model_construction() {
    let mut graph = Graph::new("main");
    graph.inputs.push(ValueInfo::new(
        "x",
        ValueType::tensor(
            ElemType::Float,
            Some(vec![Dim::Param("N".into()), Dim::Fixed(3)]),
        ),
    ));
    let init = Tensor::new(
        "scale",
        ElemType::Float,
        vec![Dim::Fixed(3)],
        TensorData::F32(vec![1.0, 2.0, 3.0]),
    );
    graph.initializers.push(init);
    graph.nodes.push(Node::new(
        "Scaler",
        ML_DOMAIN,
        vec!["x".into()],
        vec!["y".into()],
        vec![Attribute::floats("scale", vec![1.0, 1.0, 1.0])],
    ));
    graph.outputs.push(ValueInfo::new(
        "y",
        ValueType::tensor(
            ElemType::Float,
            Some(vec![Dim::Param("N".into()), Dim::Fixed(3)]),
        ),
    ));

    let model = Model::new(
        graph,
        vec![
            OpsetId {
                domain: String::new(),
                version: 21,
            },
            OpsetId {
                domain: ML_DOMAIN.to_string(),
                version: 5,
            },
        ],
    );

    assert_eq!(model.graph.name, "main");
    assert_eq!(model.graph.nodes.len(), 1);
    assert_eq!(model.graph.initializer("scale").unwrap().name, "scale");
    assert_eq!(model.opset(ML_DOMAIN), Some(5));
    let out = &model.graph.outputs[0];
    match out.value_type.as_ref().unwrap() {
        ValueType::Tensor(t) => {
            assert_eq!(t.elem, ElemType::Float);
            assert_eq!(t.shape.as_ref().unwrap()[1], Dim::Fixed(3));
        }
        other => panic!("expected tensor type, got {other:?}"),
    }
}

#[test]
fn model_preserves_unsupported_raw_data() {
    let graph_proto = fake_graph_proto(0x10);
    let function_proto = fake_graph_proto(0x20);
    let training_proto = fake_graph_proto(0x30);
    let sparse_proto = fake_graph_proto(0x40);

    let subgraph_attr = Attribute::graph(
        "body",
        GraphRef {
            name: "loop_body".into(),
            proto: graph_proto.clone(),
        },
    );

    let mut graph = Graph::new("main");
    graph.nodes.push(Node::new(
        "Loop",
        DEFAULT_DOMAIN,
        vec!["trip".into()],
        vec!["out".into()],
        vec![subgraph_attr],
    ));
    graph.sparse_initializers_raw.push(SparseTensorRaw {
        name: "sparse_w".into(),
        proto: sparse_proto.clone(),
    });

    let mut model = Model::new(
        graph,
        vec![OpsetId {
            domain: String::new(),
            version: 21,
        }],
    );
    model.functions_raw.push(FunctionRaw {
        domain: "com.example".into(),
        name: "MyFn".into(),
        overload: String::new(),
        proto: function_proto.clone(),
    });
    model.training_info_raw.push(TrainingInfoRaw {
        proto: training_proto.clone(),
    });

    let cloned = model.clone();
    assert_eq!(cloned, model);

    let attr = cloned.graph.nodes[0].attr("body").unwrap();
    match &attr.value {
        AttributeValue::Graph(g) => assert_eq!(g.proto, graph_proto),
        other => panic!("expected graph ref, got {other:?}"),
    }
    assert_eq!(cloned.functions_raw[0].proto, function_proto);
    assert_eq!(cloned.training_info_raw[0].proto, training_proto);
    assert_eq!(cloned.graph.sparse_initializers_raw[0].proto, sparse_proto);
}

#[test]
fn node_optional_and_variadic_positions() {
    let mut node = Node::new(
        "Concat",
        DEFAULT_DOMAIN,
        vec!["a".into(), String::new()],
        vec!["c".into()],
        vec![Attribute::int("axis", 0)],
    );
    node.name = Some("concat_0".into());
    assert_eq!(node.inputs[1], "");
    assert_eq!(node.name.as_deref(), Some("concat_0"));
    node.inputs.push("b".into());
    assert_eq!(node.inputs.len(), 3);
}

#[test]
fn ssa_duplicate_output_rejected() {
    let mut graph = Graph::new("dup");
    graph.inputs.push(ValueInfo::new(
        "x",
        ValueType::tensor(ElemType::Float, Some(vec![Dim::Fixed(2)])),
    ));
    graph.nodes.push(Node::new(
        "Identity",
        DEFAULT_DOMAIN,
        vec!["x".into()],
        vec!["y".into()],
        vec![],
    ));
    graph.nodes.push(Node::new(
        "Identity",
        DEFAULT_DOMAIN,
        vec!["x".into()],
        vec!["y".into()],
        vec![],
    ));
    graph.outputs.push(ValueInfo::new(
        "y",
        ValueType::tensor(ElemType::Float, Some(vec![Dim::Fixed(2)])),
    ));
    let report = validate_graph(&graph);
    assert!(report.errors.iter().any(|e| matches!(
        e,
        ValidationError::DuplicateOutput { name, .. } if name == "y"
    )));
}

#[test]
fn undefined_value_reference_detected() {
    let mut graph = Graph::new("undef");
    graph.inputs.push(ValueInfo::new(
        "x",
        ValueType::tensor(ElemType::Float, Some(vec![Dim::Fixed(2)])),
    ));
    graph.nodes.push(Node::new(
        "Identity",
        DEFAULT_DOMAIN,
        vec!["ghost".into()],
        vec!["y".into()],
        vec![],
    ));
    graph.outputs.push(ValueInfo::new(
        "y",
        ValueType::tensor(ElemType::Float, Some(vec![Dim::Fixed(2)])),
    ));
    let report = validate_graph(&graph);
    assert!(report.errors.iter().any(|e| matches!(
        e,
        ValidationError::UndefinedValue { name, .. } if name == "ghost"
    )));
}

#[test]
fn cycle_detected_via_references() {
    let mut graph = Graph::new("cyclic");
    graph.nodes.push(Node::new(
        "Identity",
        DEFAULT_DOMAIN,
        vec!["b".into()],
        vec!["a".into()],
        vec![],
    ));
    graph.nodes.push(Node::new(
        "Identity",
        DEFAULT_DOMAIN,
        vec!["a".into()],
        vec!["b".into()],
        vec![],
    ));
    graph.outputs.push(ValueInfo::new(
        "a",
        ValueType::tensor(ElemType::Float, Some(vec![Dim::Fixed(1)])),
    ));
    let report = validate_graph(&graph);
    assert!(
        report
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::Cycle { .. }))
    );
}

#[test]
fn well_formed_graph_passes_validation() {
    let mut graph = Graph::new("ok");
    graph.inputs.push(ValueInfo::new(
        "x",
        ValueType::tensor(ElemType::Float, Some(vec![Dim::Fixed(2)])),
    ));
    graph.initializers.push(Tensor::new(
        "w",
        ElemType::Float,
        vec![Dim::Fixed(2)],
        TensorData::F32(vec![1.0, 2.0]),
    ));
    graph.nodes.push(Node::new(
        "Add",
        DEFAULT_DOMAIN,
        vec!["x".into(), "w".into()],
        vec!["y".into()],
        vec![],
    ));
    graph.outputs.push(ValueInfo::new(
        "y",
        ValueType::tensor(ElemType::Float, Some(vec![Dim::Fixed(2)])),
    ));
    let report = validate_graph(&graph);
    assert!(report.is_ok(), "unexpected errors: {report}");
}

#[test]
fn missing_shape_on_main_io_reported() {
    let mut graph = Graph::new("no_shape");
    graph.inputs.push(ValueInfo::new(
        "x",
        ValueType::tensor(ElemType::Float, None),
    ));
    graph.outputs.push(ValueInfo {
        name: "y".into(),
        value_type: None,
        doc_string: String::new(),
    });
    let report = validate_graph(&graph);
    assert_eq!(
        report
            .errors
            .iter()
            .filter(|e| matches!(e, ValidationError::MissingShape { .. }))
            .count(),
        2
    );
}

#[test]
fn conflicting_attribute_values_rejected() {
    let attr = attribute_from_fields(
        "both".into(),
        AttributeFields {
            float: Some(1.0),
            int: Some(7),
            string_bytes: None,
            tensor: None,
            graph: None,
            floats: None,
            ints: None,
            strings: None,
            tensors: None,
            graphs: None,
        },
    );
    assert!(matches!(
        attr,
        Err(AttributeError::ConflictingValues { .. })
    ));

    let single = attribute_from_fields(
        "one".into(),
        AttributeFields {
            float: None,
            int: Some(7),
            string_bytes: None,
            tensor: None,
            graph: None,
            floats: None,
            ints: None,
            strings: None,
            tensors: None,
            graphs: None,
        },
    );
    assert!(matches!(single, Ok(a) if a.value == AttributeValue::Int(7)));
}

#[test]
fn optional_empty_string_input_is_absent_position() {
    let mut graph = Graph::new("optional");
    graph.inputs.push(ValueInfo::new(
        "x",
        ValueType::tensor(ElemType::Float, Some(vec![Dim::Fixed(4)])),
    ));
    let mut node = Node::new(
        "Sum",
        DEFAULT_DOMAIN,
        vec!["x".into(), String::new()],
        vec!["y".into()],
        vec![],
    );
    node.outputs.push(String::new());
    graph.nodes.push(node);
    graph.outputs.push(ValueInfo::new(
        "y",
        ValueType::tensor(ElemType::Float, Some(vec![Dim::Fixed(4)])),
    ));
    let report = validate_graph(&graph);
    assert!(
        report.is_ok(),
        "empty-string optional slots must not error: {report}"
    );
}
