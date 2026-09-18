#![cfg(all(feature = "export", feature = "import"))]

use serde_onnx::export::{ExportError, GraphBuilder, ToOnnx, ValueRef, export_model};
use serde_onnx::import::{DecodedModel, ImportWarningKind, PayloadVisitor};
use serde_onnx::ir::{Attribute, Dim, ElemType, GraphRef, Model, Node, OpsetId, ValueType};
use serde_onnx::ml::{NodePayload, Scaler, TreeEnsembleClassifier, TreeNodes};
use serde_onnx::proto::encode_model;

fn matrix() -> ValueType {
    ValueType::tensor(ElemType::Float, Some(vec![Dim::Unknown, Dim::Unknown]))
}

struct ScalerModel {
    scaler: Scaler,
}

impl ToOnnx for ScalerModel {
    fn to_graph(&self, builder: &mut GraphBuilder) -> Result<ValueRef, ExportError> {
        let x = builder.input("X", matrix())?;
        let outs = builder.emit_op(&self.scaler, vec![x], 1)?;
        let out = outs.into_iter().next().expect("single output");
        builder.output(out.name(), matrix())?;
        Ok(ValueRef::from(out.name().to_string()))
    }
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

struct EnsembleModel {
    ensemble: TreeEnsembleClassifier,
}

impl ToOnnx for EnsembleModel {
    fn to_graph(&self, builder: &mut GraphBuilder) -> Result<ValueRef, ExportError> {
        let x = builder.input("X", matrix())?;
        let outs = builder.emit_op(&self.ensemble, vec![x], 2)?;
        let mut outs = outs.into_iter();
        let y = outs.next().expect("first output");
        let z = outs.next().expect("second output");
        builder.output(y.name(), matrix())?;
        builder.output(z.name(), matrix())?;
        Ok(ValueRef::from(y.name().to_string()))
    }
}

#[test]
fn scaler_export_encode_decode_recognizes_typed() {
    let scaler = Scaler {
        offset: Some(vec![1.0, 2.0]),
        scale: Some(vec![0.5, 0.25]),
    };
    let model = export_model(
        &ScalerModel {
            scaler: scaler.clone(),
        },
        "scaler_graph",
    )
    .expect("export must succeed");
    let decoded = DecodedModel::decode_bytes(&encode_model(&model)).expect("decode must succeed");
    assert_eq!(decoded.payloads.len(), 1);
    match &decoded.payloads[0] {
        NodePayload::Scaler(typed) => assert_eq!(typed.op, scaler),
        other => panic!("expected typed Scaler, got {:?}", other.node().op_type),
    }
    assert_eq!(decoded.inputs().len(), 1);
    assert_eq!(decoded.inputs()[0].name, "X");
    assert_eq!(decoded.outputs().len(), 1);
    assert!(decoded.initializers().is_empty());
    assert!(decoded.warnings.is_empty());
}

#[test]
fn tree_ensemble_classifier_export_encode_decode_recognizes_typed() {
    let ensemble = TreeEnsembleClassifier {
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
    let model = export_model(
        &EnsembleModel {
            ensemble: ensemble.clone(),
        },
        "ensemble_graph",
    )
    .expect("export must succeed");
    let decoded = DecodedModel::decode_bytes(&encode_model(&model)).expect("decode must succeed");
    assert_eq!(decoded.payloads.len(), 1);
    match &decoded.payloads[0] {
        NodePayload::TreeEnsembleClassifier(typed) => assert_eq!(typed.op, ensemble),
        other => panic!(
            "expected typed TreeEnsembleClassifier, got {:?}",
            other.node().op_type
        ),
    }
    assert!(decoded.warnings.is_empty());
}

#[test]
fn unknown_op_decodes_with_raw_preserved() {
    let mut graph = serde_onnx::ir::Graph::new("raw_graph");
    graph
        .inputs
        .push(serde_onnx::ir::ValueInfo::new("x", matrix()));
    let node = Node::new(
        "FancyFutureOp",
        "ai.onnx.ml",
        vec!["x".to_string()],
        vec!["y".to_string()],
        vec![Attribute::float("weight", 1.5)],
    );
    graph.nodes.push(node.clone());
    graph
        .outputs
        .push(serde_onnx::ir::ValueInfo::new("y", matrix()));
    let model = Model::new(
        graph,
        vec![OpsetId {
            domain: "ai.onnx.ml".to_string(),
            version: 4,
        }],
    );
    let decoded = DecodedModel::decode_bytes(&encode_model(&model)).expect("decode must succeed");
    assert_eq!(decoded.payloads.len(), 1);
    match &decoded.payloads[0] {
        NodePayload::Raw(kept) => assert_eq!(kept, &node),
        other => panic!("expected Raw, got {:?}", other.node().op_type),
    }
    assert!(decoded.warnings.is_empty());
}

#[test]
fn subgraph_attribute_marks_non_fatal_warning() {
    let mut graph = serde_onnx::ir::Graph::new("loop_graph");
    graph
        .inputs
        .push(serde_onnx::ir::ValueInfo::new("x", matrix()));
    graph.nodes.push(Node::new(
        "Loop",
        "",
        vec!["x".to_string()],
        vec!["y".to_string()],
        vec![Attribute::graph(
            "body",
            GraphRef {
                name: "body".to_string(),
                proto: vec![1, 2, 3],
            },
        )],
    ));
    graph
        .outputs
        .push(serde_onnx::ir::ValueInfo::new("y", matrix()));
    let model = Model::new(
        graph,
        vec![OpsetId {
            domain: String::new(),
            version: 21,
        }],
    );
    let decoded = DecodedModel::decode_bytes(&encode_model(&model)).expect("decode must succeed");
    assert_eq!(decoded.payloads.len(), 1);
    assert!(decoded.payloads[0].is_raw());
    assert_eq!(decoded.warnings.len(), 1);
    assert_eq!(
        decoded.warnings[0].kind,
        ImportWarningKind::SubgraphAttribute
    );
    assert_eq!(decoded.warnings[0].node_index, Some(0));
}

#[test]
fn function_and_training_blobs_mark_non_fatal_warnings() {
    let scaler = Scaler {
        offset: Some(vec![0.0]),
        scale: Some(vec![1.0]),
    };
    let mut model =
        export_model(&ScalerModel { scaler }, "warn_graph").expect("export must succeed");
    model.functions_raw.push(serde_onnx::ir::FunctionRaw {
        domain: "custom.domain".to_string(),
        name: "CustomFn".to_string(),
        overload: String::new(),
        proto: vec![9, 9, 9],
    });
    model
        .training_info_raw
        .push(serde_onnx::ir::TrainingInfoRaw { proto: vec![7, 7] });
    let decoded = DecodedModel::decode_bytes(&encode_model(&model)).expect("decode must succeed");
    assert!(matches!(decoded.payloads[0], NodePayload::Scaler(_)));
    assert_eq!(decoded.warnings.len(), 2);
    assert!(
        decoded
            .warnings
            .iter()
            .any(|w| w.kind == ImportWarningKind::Function)
    );
    assert!(
        decoded
            .warnings
            .iter()
            .any(|w| w.kind == ImportWarningKind::TrainingInfo)
    );
}

#[test]
fn decode_file_reads_onnx_bytes_from_disk() {
    let scaler = Scaler {
        offset: None,
        scale: Some(vec![2.0]),
    };
    let model = export_model(
        &ScalerModel {
            scaler: scaler.clone(),
        },
        "file_graph",
    )
    .expect("export must succeed");
    let dir = std::env::temp_dir().join(format!("serde_onnx_import_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("model.onnx");
    std::fs::write(&path, encode_model(&model)).expect("write model");
    let decoded = DecodedModel::decode_file(&path).expect("decode must succeed");
    match &decoded.payloads[0] {
        NodePayload::Scaler(typed) => assert_eq!(typed.op, scaler),
        other => panic!("expected typed Scaler, got {:?}", other.node().op_type),
    }
    std::fs::remove_dir_all(&dir).ok();
}

struct Counter {
    typed: usize,
    raw: usize,
}

impl PayloadVisitor for Counter {
    fn visit_typed(&mut self, _index: usize, _payload: &NodePayload) {
        self.typed += 1;
    }

    fn visit_raw(&mut self, _index: usize, _node: &Node) {
        self.raw += 1;
    }
}

#[test]
fn visitor_distinguishes_typed_and_raw() {
    let scaler = Scaler {
        offset: Some(vec![1.0]),
        scale: None,
    };
    let mut model =
        export_model(&ScalerModel { scaler }, "visit_graph").expect("export must succeed");
    model.graph.nodes.push(Node::new(
        "FancyFutureOp",
        "custom.domain",
        vec!["node_0_Scaler_out".to_string()],
        vec!["z".to_string()],
        Vec::new(),
    ));
    model
        .graph
        .outputs
        .push(serde_onnx::ir::ValueInfo::new("z", matrix()));
    let decoded = DecodedModel::decode_bytes(&encode_model(&model)).expect("decode must succeed");
    let mut counter = Counter { typed: 0, raw: 0 };
    decoded.visit(&mut counter);
    assert_eq!((counter.typed, counter.raw), (1, 1));
    assert_eq!(decoded.typed().count(), 1);
    assert_eq!(decoded.raw_nodes().count(), 1);
}
