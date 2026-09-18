use super::*;
use crate::ir::{Attribute, AttributeValue, ElemType, Node, Tensor, TensorData};

fn io1() -> (Vec<String>, Vec<String>) {
    (vec!["x".to_string()], vec!["y".to_string()])
}

fn io2() -> (Vec<String>, Vec<String>) {
    (
        vec!["x".to_string()],
        vec!["y".to_string(), "z".to_string()],
    )
}

fn roundtrip<O: OnnxOp>(op: &O, inputs: Vec<String>, outputs: Vec<String>) -> O {
    let emitted = op.to_node(inputs, outputs).expect("to_node failed");
    assert!(emitted.initializers.is_empty());
    O::from_node(&emitted.node).expect("from_node failed")
}

fn f32_tensor(name: &str, vals: Vec<f32>) -> Tensor {
    Tensor::new(
        name,
        ElemType::Float,
        vec![crate::ir::Dim::Fixed(vals.len() as i64)],
        TensorData::F32(vals),
    )
}

#[test]
fn scaler_roundtrip_full() {
    let op = Scaler {
        offset: Some(vec![1.0, 2.0]),
        scale: Some(vec![0.5, 0.25]),
    };
    let (i, o) = io1();
    assert_eq!(roundtrip(&op, i, o), op);
}

#[test]
fn scaler_defaults_absent_stay_absent() {
    let op = Scaler {
        offset: None,
        scale: None,
    };
    let (i, o) = io1();
    let emitted = op.to_node(i, o).expect("to_node failed");
    assert!(emitted.node.attributes.is_empty());
    assert_eq!(
        Scaler::from_node(&emitted.node).expect("from_node failed"),
        op
    );
}

#[test]
fn imputer_roundtrip_float() {
    let op = Imputer {
        imputed_value_floats: Some(vec![-1.0]),
        imputed_value_int64s: None,
        replaced_value_float: Some(0.0),
        replaced_value_int64: None,
    };
    let (i, o) = io1();
    assert_eq!(roundtrip(&op, i, o), op);
}

#[test]
fn imputer_roundtrip_int_absent_defaults() {
    let op = Imputer {
        imputed_value_floats: None,
        imputed_value_int64s: Some(vec![7, 8]),
        replaced_value_float: None,
        replaced_value_int64: None,
    };
    let (i, o) = io1();
    let back = roundtrip(&op, i, o);
    assert_eq!(back, op);
    assert!(back.replaced_value_float.is_none());
    assert!(back.replaced_value_int64.is_none());
}

#[test]
fn imputer_rejects_both_imputed_kinds() {
    let op = Imputer {
        imputed_value_floats: Some(vec![1.0]),
        imputed_value_int64s: Some(vec![1]),
        replaced_value_float: None,
        replaced_value_int64: None,
    };
    let (i, o) = io1();
    assert!(op.to_node(i, o).is_err());
}

#[test]
fn normalizer_roundtrip() {
    for norm in [None, Some("L1".to_string()), Some("MAX".to_string())] {
        let op = Normalizer { norm };
        let (i, o) = io1();
        assert_eq!(roundtrip(&op, i.clone(), o.clone()), op);
    }
}

#[test]
fn label_encoder_strings_roundtrip() {
    let op = LabelEncoder {
        keys_strings: Some(vec!["a".to_string(), "b".to_string()]),
        values_int64s: Some(vec![0, 1]),
        default_int64: Some(42),
        ..LabelEncoder::default()
    };
    let (i, o) = io1();
    assert_eq!(roundtrip(&op, i, o), op);
}

#[test]
fn label_encoder_tensor_attrs_roundtrip() {
    let op = LabelEncoder {
        keys_tensor: Some(f32_tensor("keys", vec![1.0, 2.0])),
        values_tensor: Some(f32_tensor("values", vec![10.0, 20.0])),
        default_tensor: Some(f32_tensor("default", vec![-1.0])),
        ..LabelEncoder::default()
    };
    let (i, o) = io1();
    assert_eq!(roundtrip(&op, i, o), op);
}

#[test]
fn label_encoder_rejects_missing_values() {
    let op = LabelEncoder {
        keys_strings: Some(vec!["a".to_string()]),
        ..LabelEncoder::default()
    };
    let (i, o) = io1();
    assert!(op.to_node(i, o).is_err());
}

#[test]
fn one_hot_encoder_int_roundtrip() {
    let op = OneHotEncoder {
        cats_int64s: Some(vec![1, 2, 3]),
        cats_strings: None,
        zeros: Some(1),
    };
    let (i, o) = io1();
    assert_eq!(roundtrip(&op, i, o), op);
}

#[test]
fn one_hot_encoder_strings_absent_zeros() {
    let op = OneHotEncoder {
        cats_int64s: None,
        cats_strings: Some(vec!["x".to_string()]),
        zeros: None,
    };
    let (i, o) = io1();
    let emitted = op.to_node(i, o).expect("to_node failed");
    assert!(emitted.node.attr("zeros").is_none());
    assert_eq!(
        OneHotEncoder::from_node(&emitted.node).expect("from_node failed"),
        op
    );
}

#[test]
fn dict_vectorizer_roundtrip() {
    let op = DictVectorizer {
        int64_vocabulary: None,
        string_vocabulary: Some(vec!["a".to_string(), "b".to_string()]),
    };
    let (i, o) = io1();
    assert_eq!(roundtrip(&op, i, o), op);
}

#[test]
fn feature_vectorizer_roundtrip() {
    let op = FeatureVectorizer {
        inputdimensions: Some(vec![2, 3]),
    };
    let emitted = op
        .to_node(
            vec!["a".to_string(), "b".to_string()],
            vec!["y".to_string()],
        )
        .expect("to_node failed");
    assert_eq!(
        FeatureVectorizer::from_node(&emitted.node).expect("from_node failed"),
        op
    );
}

#[test]
fn linear_classifier_roundtrip() {
    let mut op = LinearClassifier::with_int_labels(vec![0, 1], vec![0.5, -0.5, 0.25, 0.75])
        .expect("build failed");
    op.intercepts = Some(vec![0.1, -0.1]);
    op.multi_class = Some(0);
    op.post_transform = Some("LOGISTIC".to_string());
    let (i, o) = io2();
    assert_eq!(roundtrip(&op, i, o), op);
}

#[test]
fn linear_classifier_absent_optionals() {
    let op = LinearClassifier::with_string_labels(vec!["a".to_string()], vec![1.0])
        .expect("build failed");
    let (i, o) = io2();
    let emitted = op.to_node(i, o).expect("to_node failed");
    assert!(emitted.node.attr("intercepts").is_none());
    assert!(emitted.node.attr("post_transform").is_none());
    assert!(emitted.node.attr("multi_class").is_none());
    assert_eq!(
        LinearClassifier::from_node(&emitted.node).expect("from_node failed"),
        op
    );
}

#[test]
fn linear_classifier_rejects_empty_coefficients() {
    let op = LinearClassifier {
        classlabels: ClassLabels {
            ints: Some(vec![0]),
            strings: None,
        },
        coefficients: Vec::new(),
        intercepts: None,
        multi_class: None,
        post_transform: None,
    };
    let (i, o) = io2();
    assert!(op.to_node(i, o).is_err());
}

#[test]
fn linear_regressor_roundtrip() {
    let op = LinearRegressor {
        coefficients: vec![1.0, 2.0],
        intercepts: None,
        post_transform: None,
        targets: Some(1),
    };
    let (i, o) = io1();
    assert_eq!(roundtrip(&op, i, o), op);
}

#[test]
fn svm_classifier_roundtrip() {
    let op = SvmClassifier {
        classlabels: ClassLabels {
            ints: Some(vec![0, 1]),
            strings: None,
        },
        common: SvmCommon {
            coefficients: Some(vec![1.0, -1.0]),
            kernel_params: Some(vec![0.5, 0.0, 3.0]),
            kernel_type: Some("RBF".to_string()),
            post_transform: None,
            prob_a: Some(vec![0.1]),
            prob_b: Some(vec![0.2]),
            rho: Some(vec![0.3]),
            support_vectors: Some(vec![1.0, 2.0]),
        },
        vectors_per_class: Some(vec![1, 1]),
    };
    let (i, o) = io2();
    assert_eq!(roundtrip(&op, i, o), op);
}

#[test]
fn svm_regressor_roundtrip() {
    let op = SvmRegressor {
        common: SvmCommon {
            kernel_type: Some("LINEAR".to_string()),
            rho: Some(vec![0.5]),
            ..SvmCommon::default()
        },
        n_supports: Some(2),
        one_class: None,
    };
    let (i, o) = io1();
    let back = roundtrip(&op, i, o);
    assert_eq!(back, op);
    assert!(back.one_class.is_none());
    assert!(back.common.post_transform.is_none());
}

fn sample_tree_nodes() -> TreeNodes {
    TreeNodes {
        falsenodeids: vec![1, 2, 2],
        featureids: vec![0, 1, 0],
        hitrates: Some(vec![1.0, 0.5, 0.5]),
        hitrates_as_tensor: None,
        missing_value_tracks_true: None,
        modes: vec![
            "BRANCH_LEQ".to_string(),
            "LEAF".to_string(),
            "LEAF".to_string(),
        ],
        nodeids: vec![0, 1, 2],
        treeids: vec![0, 0, 0],
        truenodeids: vec![1, 1, 2],
        values: Some(vec![0.5, 0.0, 0.0]),
        values_as_tensor: None,
    }
}

#[test]
fn tree_ensemble_classifier_full_roundtrip() {
    let op = TreeEnsembleClassifier {
        base_values: Some(vec![0.1, 0.2]),
        base_values_as_tensor: None,
        class_ids: vec![0, 1],
        class_nodeids: vec![1, 2],
        class_treeids: vec![0, 0],
        class_weights: Some(vec![1.5, 2.5]),
        class_weights_as_tensor: None,
        classlabels_int64s: Some(vec![0, 1]),
        classlabels_strings: None,
        nodes: sample_tree_nodes(),
        post_transform: Some("SOFTMAX".to_string()),
    };
    let (i, o) = io2();
    assert_eq!(roundtrip(&op, i, o), op);
}

#[test]
fn tree_ensemble_classifier_tensor_arrays_roundtrip() {
    let mut nodes = sample_tree_nodes();
    nodes.hitrates = None;
    nodes.hitrates_as_tensor = Some(f32_tensor("hitrates", vec![1.0, 0.5, 0.5]));
    nodes.values = None;
    nodes.values_as_tensor = Some(f32_tensor("splits", vec![0.5, 0.0, 0.0]));
    let op = TreeEnsembleClassifier {
        base_values: None,
        base_values_as_tensor: Some(f32_tensor("base", vec![0.0, 0.0])),
        class_ids: vec![0],
        class_nodeids: vec![1],
        class_treeids: vec![0],
        class_weights: None,
        class_weights_as_tensor: Some(f32_tensor("cw", vec![3.0])),
        classlabels_int64s: None,
        classlabels_strings: Some(vec!["neg".to_string(), "pos".to_string()]),
        nodes,
        post_transform: None,
    };
    let (i, o) = io2();
    let back = roundtrip(&op, i, o);
    assert_eq!(back, op);
    assert!(back.post_transform.is_none());
    assert!(back.base_values.is_none());
}

#[test]
fn tree_ensemble_regressor_full_roundtrip() {
    let op = TreeEnsembleRegressor {
        aggregate_function: Some("SUM".to_string()),
        base_values: None,
        base_values_as_tensor: None,
        n_targets: Some(1),
        nodes: sample_tree_nodes(),
        post_transform: None,
        target_ids: vec![0, 0],
        target_nodeids: vec![1, 2],
        target_treeids: vec![0, 0],
        target_weights: Some(vec![1.0, -1.0]),
        target_weights_as_tensor: None,
    };
    let (i, o) = io1();
    let back = roundtrip(&op, i, o);
    assert_eq!(back, op);
    assert!(back.post_transform.is_none());
    assert!(back.base_values.is_none());
}

#[test]
fn zipmap_roundtrip() {
    for op in [
        ZipMap {
            classlabels_int64s: Some(vec![0, 1]),
            classlabels_strings: None,
        },
        ZipMap {
            classlabels_int64s: None,
            classlabels_strings: Some(vec!["a".to_string()]),
        },
    ] {
        let (i, o) = io1();
        assert_eq!(roundtrip(&op, i.clone(), o.clone()), op);
    }
}

#[test]
fn cast_roundtrip() {
    let op = Cast {
        to: ElemType::Float,
        saturate: Some(1),
    };
    let (i, o) = io1();
    assert_eq!(roundtrip(&op, i, o), op);
    let bare = Cast {
        to: ElemType::Int64,
        saturate: None,
    };
    let (i, o) = io1();
    let emitted = bare.to_node(i, o).expect("to_node failed");
    assert!(emitted.node.attr("saturate").is_none());
    assert_eq!(
        Cast::from_node(&emitted.node).expect("from_node failed"),
        bare
    );
}

#[test]
fn reshape_roundtrip() {
    let op = Reshape { allowzero: Some(1) };
    let emitted = op
        .to_node(
            vec!["d".to_string(), "s".to_string()],
            vec!["r".to_string()],
        )
        .expect("to_node failed");
    assert_eq!(
        Reshape::from_node(&emitted.node).expect("from_node failed"),
        op
    );
    let bare = Reshape { allowzero: None };
    let (i, o) = io1();
    let emitted = bare.to_node(i, o).expect("to_node failed");
    assert!(emitted.node.attributes.is_empty());
    assert_eq!(
        Reshape::from_node(&emitted.node).expect("from_node failed"),
        bare
    );
}

#[test]
fn concat_roundtrip() {
    let op = Concat { axis: -1 };
    let emitted = op
        .to_node(
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string()],
        )
        .expect("to_node failed");
    assert_eq!(
        Concat::from_node(&emitted.node).expect("from_node failed"),
        op
    );
}

#[test]
fn gather_roundtrip() {
    let op = Gather { axis: None };
    let emitted = op
        .to_node(
            vec!["d".to_string(), "i".to_string()],
            vec!["o".to_string()],
        )
        .expect("to_node failed");
    assert!(emitted.node.attributes.is_empty());
    assert_eq!(
        Gather::from_node(&emitted.node).expect("from_node failed"),
        op
    );
}

#[test]
fn identity_roundtrip() {
    let op = Identity;
    let (i, o) = io1();
    assert_eq!(roundtrip(&op, i, o), op);
}

#[test]
fn recognition_typed_and_raw() {
    let scaler = Scaler {
        offset: Some(vec![0.0]),
        scale: Some(vec![1.0]),
    };
    let (i, o) = io1();
    let node = scaler.to_node(i, o).expect("to_node failed").node;
    match make_node_payload(&node) {
        NodePayload::Scaler(typed) => assert_eq!(typed.op, scaler),
        other => panic!("expected Scaler payload, got {:?}", other.node().op_type),
    }
    let unknown = Node::new(
        "FancyFutureOp",
        "ai.onnx.ml",
        vec!["x".to_string()],
        vec!["y".to_string()],
        Vec::new(),
    );
    assert!(make_node_payload(&unknown).is_raw());
}

#[test]
fn recognition_unknown_attr_falls_back_to_raw() {
    let scaler = Scaler {
        offset: Some(vec![0.0]),
        scale: None,
    };
    let (i, o) = io1();
    let mut node = scaler.to_node(i, o).expect("to_node failed").node;
    node.attributes.push(Attribute::int("future_flag", 1));
    let payload = make_node_payload(&node);
    assert!(payload.is_raw());
    assert_eq!(payload.node(), &node);
}

#[test]
fn recognition_wrong_attr_type_falls_back_to_raw() {
    let node = Node::new(
        "Scaler",
        crate::ir::ML_DOMAIN,
        vec!["x".to_string()],
        vec!["y".to_string()],
        vec![Attribute::int("scale", 3)],
    );
    assert!(make_node_payload(&node).is_raw());
}

#[test]
fn recognition_duplicate_attr_falls_back_to_raw() {
    let node = Node::new(
        "Concat",
        "",
        vec!["a".to_string()],
        vec!["c".to_string()],
        vec![Attribute::int("axis", 0), Attribute::int("axis", 1)],
    );
    assert!(make_node_payload(&node).is_raw());
}

#[test]
fn recognition_missing_required_falls_back_to_raw() {
    let node = Node::new(
        "Concat",
        "",
        vec!["a".to_string()],
        vec!["c".to_string()],
        Vec::new(),
    );
    assert!(make_node_payload(&node).is_raw());
    let cast = Node::new(
        "Cast",
        "",
        vec!["x".to_string()],
        vec!["y".to_string()],
        Vec::new(),
    );
    assert!(make_node_payload(&cast).is_raw());
}

#[test]
fn recognition_preserves_node_metadata() {
    let scaler = Scaler {
        offset: Some(vec![1.0]),
        scale: None,
    };
    let (i, o) = io1();
    let mut node = scaler.to_node(i, o).expect("to_node failed").node;
    node.name = Some("scale0".to_string());
    node.doc_string = "doc".to_string();
    node.metadata = vec![("k".to_string(), "v".to_string())];
    node.device_configurations = vec![crate::ir::NodeDeviceConfiguration {
        configuration_id: "cfg".to_string(),
        pipeline_stage: 2,
        sharding_specs_raw: Vec::new(),
    }];
    match make_node_payload(&node) {
        NodePayload::Scaler(typed) => {
            assert_eq!(typed.node, node);
            assert_eq!(typed.op, scaler);
        }
        _ => panic!("expected typed Scaler"),
    }
    let unknown = Node::new(
        "Other",
        "custom.domain",
        vec!["x".to_string()],
        vec!["y".to_string()],
        vec![Attribute::float("w", 1.0)],
    );
    let mut unknown = unknown;
    unknown.metadata = vec![("a".to_string(), "b".to_string())];
    match make_node_payload(&unknown) {
        NodePayload::Raw(kept) => assert_eq!(kept, unknown),
        _ => panic!("expected Raw"),
    }
}

#[test]
fn wrong_attr_value_type_rejected() {
    let node = Node::new(
        "Gather",
        "",
        vec!["d".to_string(), "i".to_string()],
        vec!["o".to_string()],
        vec![Attribute::float("axis", 1.0)],
    );
    assert!(matches!(
        Gather::from_node(&node),
        Err(OpError::WrongAttributeType { .. })
    ));
}

#[test]
fn attribute_value_kinds_rejected() {
    let node = Node::new(
        "Scaler",
        crate::ir::ML_DOMAIN,
        vec!["x".to_string()],
        vec!["y".to_string()],
        vec![Attribute::new(
            "scale",
            AttributeValue::String("bad".to_string()),
        )],
    );
    assert!(matches!(
        Scaler::from_node(&node),
        Err(OpError::WrongAttributeType { .. })
    ));
}

#[test]
fn opset_target_constants_match_codec_limits() {
    assert_eq!(ML_EXPORT_OPSET_TARGET, 4);
    assert_eq!(CORE_TYPED_OPSET_TARGET, 21);
    assert_eq!(CODEC_MAX_IR_VERSION, 10);
    assert_eq!(CODEC_MAX_ONNX_OPSET, 25);
    assert_eq!(CODEC_MAX_ML_OPSET, 5);
}
