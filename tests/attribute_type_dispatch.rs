use serde_onnx::ir::{Attribute, AttributeValue};
use serde_onnx::proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, ProtoCodecError, Version,
    attribute_proto::AttributeType, decode_model_proto,
};

fn decode_attributes(attributes: Vec<AttributeProto>) -> Result<Vec<Attribute>, ProtoCodecError> {
    let proto = ModelProto {
        ir_version: Version::IrVersion2024325 as i64,
        graph: Some(GraphProto {
            name: "dispatch_graph".into(),
            node: vec![NodeProto {
                op_type: "Test".into(),
                domain: String::new(),
                attribute: attributes,
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let model = decode_model_proto(&proto)?;
    Ok(model.graph.nodes[0].attributes.clone())
}

fn decode_one(attribute: AttributeProto) -> Result<Attribute, ProtoCodecError> {
    let mut attributes = decode_attributes(vec![attribute])?;
    assert_eq!(attributes.len(), 1);
    Ok(attributes.remove(0))
}

#[test]
fn int_attribute_with_zero_value_is_decoded() {
    let attribute = decode_one(AttributeProto {
        name: "axis".into(),
        r#type: AttributeType::Int as i32,
        i: 0,
        ..Default::default()
    })
    .expect("INT attribute with value 0 is legal and must decode");

    assert_eq!(attribute.name, "axis");
    assert_eq!(attribute.value, AttributeValue::Int(0));
}

#[test]
fn float_attribute_with_zero_value_is_decoded() {
    let attribute = decode_one(AttributeProto {
        name: "epsilon".into(),
        r#type: AttributeType::Float as i32,
        f: 0.0,
        ..Default::default()
    })
    .expect("FLOAT attribute with value 0.0 is legal and must decode");

    assert_eq!(attribute.value, AttributeValue::Float(0.0));
}

#[test]
fn string_attribute_with_empty_value_is_decoded() {
    let attribute = decode_one(AttributeProto {
        name: "pad".into(),
        r#type: AttributeType::String as i32,
        s: Vec::new(),
        ..Default::default()
    })
    .expect("STRING attribute with an empty value is legal and must decode");

    assert_eq!(attribute.value, AttributeValue::String(String::new()));
}

#[test]
fn ints_attribute_containing_zero_is_decoded() {
    let attribute = decode_one(AttributeProto {
        name: "pads".into(),
        r#type: AttributeType::Ints as i32,
        ints: vec![0, 1, 0, 2],
        ..Default::default()
    })
    .expect("INTS attribute containing zeros is legal and must decode");

    assert_eq!(attribute.value, AttributeValue::Ints(vec![0, 1, 0, 2]));
}

#[test]
fn floats_attribute_containing_zero_is_decoded() {
    let attribute = decode_one(AttributeProto {
        name: "coefficients".into(),
        r#type: AttributeType::Floats as i32,
        floats: vec![0.0, 1.5, 0.0],
        ..Default::default()
    })
    .expect("FLOATS attribute containing zeros is legal and must decode");

    assert_eq!(attribute.value, AttributeValue::Floats(vec![0.0, 1.5, 0.0]));
}

#[test]
fn empty_ints_attribute_is_decoded_as_empty_list() {
    let attribute = decode_one(AttributeProto {
        name: "empty_ints".into(),
        r#type: AttributeType::Ints as i32,
        ..Default::default()
    })
    .expect("INTS attribute with an empty list is legal and must decode");

    assert_eq!(attribute.value, AttributeValue::Ints(Vec::new()));
}

#[test]
fn attribute_without_declared_type_is_rejected() {
    let error = decode_one(AttributeProto {
        name: "value_less".into(),
        r#type: AttributeType::Undefined as i32,
        ..Default::default()
    })
    .expect_err("an attribute without a declared type must still be rejected");

    let message = error.to_string();
    assert!(
        message.contains("value_less") && message.contains("carries no value"),
        "unexpected error message: {message}"
    );
}

#[test]
fn declared_type_wins_over_populated_foreign_field() {
    let attribute = decode_one(AttributeProto {
        name: "axis".into(),
        r#type: AttributeType::Int as i32,
        i: 0,
        floats: vec![9.0, 9.0],
        ..Default::default()
    })
    .expect("the declared INT type discriminates the value field");

    assert_eq!(attribute.value, AttributeValue::Int(0));
}

#[test]
fn round_trip_preserves_zero_valued_attributes() {
    use serde_onnx::ir::{Graph, Model, Node, OpsetId};
    use serde_onnx::proto::{decode_model, encode_model};

    let mut graph = Graph::new("zero_graph");
    graph.nodes.push(Node::new(
        "Reduce",
        "",
        vec!["x".into()],
        vec!["y".into()],
        vec![
            Attribute::int("axis", 0),
            Attribute::int("keepdims", 0),
            Attribute::float("epsilon", 0.0),
            Attribute::string("padding", ""),
            Attribute::ints("pads", vec![0, 0]),
            Attribute::floats("weights", vec![0.0, 0.0]),
        ],
    ));
    let model = Model::new(
        graph,
        vec![OpsetId {
            domain: String::new(),
            version: 21,
        }],
    );

    let decoded = decode_model(&encode_model(&model)).expect("zero-valued attributes round-trip");
    assert_eq!(decoded, model);
}

#[test]
fn malformed_declared_type_is_rejected() {
    let error = decode_one(AttributeProto {
        name: "bogus".into(),
        r#type: 9999,
        i: 7,
        ..Default::default()
    })
    .expect_err("an attribute declaring an unknown type must be rejected");

    let message = error.to_string();
    assert!(
        message.contains("bogus") && message.contains("9999"),
        "unexpected error message: {message}"
    );
}
