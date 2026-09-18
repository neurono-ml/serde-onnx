use std::path::Path;

use prost::Message;

use crate::ir::{
    Attribute, AttributeFields, AttributeValue, Dim, ElemType, ExternalData, FunctionRaw, Graph,
    GraphRef, ML_DOMAIN, Model, Node, OpsetId, SparseTensorRaw, Tensor, TensorData, TensorError,
    TensorStorage, TrainingInfoRaw, ValueInfo, ValueType,
};
use crate::proto::{
    AttributeProto, DeviceConfigurationProto, FunctionProto, GraphProto, ModelProto,
    NodeDeviceConfigurationProto, NodeProto, OperatorSetIdProto, ShardingSpecProto,
    SparseTensorProto, StringStringEntryProto, TensorProto, TensorShapeProto, TrainingInfoProto,
    TypeProto, ValueInfoProto, attribute_proto, tensor_proto, tensor_shape_proto, type_proto,
};

fn kv_entries(entries: &[(String, String)]) -> Vec<StringStringEntryProto> {
    entries
        .iter()
        .map(|(k, v)| StringStringEntryProto {
            key: k.clone(),
            value: v.clone(),
        })
        .collect()
}

fn decode_kv_entries(entries: &[StringStringEntryProto]) -> Vec<(String, String)> {
    entries
        .iter()
        .map(|e| (e.key.clone(), e.value.clone()))
        .collect()
}

fn encode_opset_id(o: &OpsetId) -> OperatorSetIdProto {
    OperatorSetIdProto {
        domain: o.domain.clone(),
        version: o.version,
    }
}

pub const SUPPORTED_IR_VERSION: i64 = crate::ir::IR_VERSION;

pub const SUPPORTED_ONNX_OPSET: i64 = 25;

pub const SUPPORTED_ML_OPSET: i64 = crate::ir::ML_OPSET_VERSION;

pub fn supported_opset(domain: &str) -> Option<i64> {
    match domain {
        "" | "ai.onnx" => Some(SUPPORTED_ONNX_OPSET),
        ML_DOMAIN => Some(SUPPORTED_ML_OPSET),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProtoCodecError {
    Protobuf(String),
    UnsupportedIrVersion {
        declared: i64,
        supported: i64,
    },
    UnsupportedOpsetVersion {
        domain: String,
        declared: i64,
        supported: i64,
    },
    TensorSizeMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    TensorDataMismatch {
        name: String,
        detail: String,
    },
    InvalidTypeProto {
        name: String,
        detail: String,
    },
    InvalidExternalData {
        name: String,
        detail: String,
    },
    Attribute(crate::ir::AttributeError),
    Io(String),
}

impl core::fmt::Display for ProtoCodecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Protobuf(m) => write!(f, "invalid protobuf data: {m}"),
            Self::UnsupportedIrVersion {
                declared,
                supported,
            } => write!(
                f,
                "unsupported IR version {declared}: this library supports up to IR version {supported}"
            ),
            Self::UnsupportedOpsetVersion {
                domain,
                declared,
                supported,
            } => write!(
                f,
                "unsupported opset version {declared} for domain {domain:?}: this library supports up to {supported}"
            ),
            Self::TensorSizeMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "tensor {name:?}: shape implies {expected} elements but data carries {got}"
            ),
            Self::TensorDataMismatch { name, detail } => {
                write!(f, "tensor {name:?}: inconsistent data: {detail}")
            }
            Self::InvalidTypeProto { name, detail } => {
                write!(f, "value info {name:?}: invalid type proto: {detail}")
            }
            Self::InvalidExternalData { name, detail } => {
                write!(f, "tensor {name:?}: invalid external data: {detail}")
            }
            Self::Attribute(e) => write!(f, "invalid attribute: {e}"),
            Self::Io(m) => write!(f, "io error: {m}"),
        }
    }
}

impl std::error::Error for ProtoCodecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Attribute(e) => Some(e),
            _ => None,
        }
    }
}

impl From<prost::DecodeError> for ProtoCodecError {
    fn from(e: prost::DecodeError) -> Self {
        Self::Protobuf(e.to_string())
    }
}

impl From<std::io::Error> for ProtoCodecError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

impl From<TensorError> for ProtoCodecError {
    fn from(e: TensorError) -> Self {
        match e {
            TensorError::InvalidExternalData(m) => Self::InvalidExternalData {
                name: String::new(),
                detail: m,
            },
            other => Self::TensorDataMismatch {
                name: String::new(),
                detail: other.to_string(),
            },
        }
    }
}

pub type Result<T> = core::result::Result<T, ProtoCodecError>;

pub fn encode_model(model: &Model) -> Vec<u8> {
    ModelProto::from(model).encode_to_vec()
}

pub fn decode_model(bytes: &[u8]) -> Result<Model> {
    let proto = ModelProto::decode(bytes)?;
    decode_model_proto(&proto)
}

pub fn decode_model_proto(proto: &ModelProto) -> Result<Model> {
    Model::try_from(proto)
}

impl From<&Model> for ModelProto {
    fn from(model: &Model) -> Self {
        encode_model_proto(model)
    }
}

impl TryFrom<&ModelProto> for Model {
    type Error = ProtoCodecError;

    fn try_from(proto: &ModelProto) -> Result<Self> {
        if proto.ir_version > SUPPORTED_IR_VERSION {
            return Err(ProtoCodecError::UnsupportedIrVersion {
                declared: proto.ir_version,
                supported: SUPPORTED_IR_VERSION,
            });
        }
        for opset in &proto.opset_import {
            if let Some(supported) = supported_opset(&opset.domain)
                && opset.version > supported
            {
                return Err(ProtoCodecError::UnsupportedOpsetVersion {
                    domain: if opset.domain.is_empty() {
                        "ai.onnx".to_string()
                    } else {
                        opset.domain.clone()
                    },
                    declared: opset.version,
                    supported,
                });
            }
        }
        let graph = proto
            .graph
            .as_ref()
            .ok_or_else(|| ProtoCodecError::Protobuf("ModelProto without graph".to_string()))?;
        Ok(Model {
            ir_version: proto.ir_version,
            opset_import: proto.opset_import.iter().map(decode_opset_id).collect(),
            producer_name: proto.producer_name.clone(),
            producer_version: proto.producer_version.clone(),
            domain: proto.domain.clone(),
            model_version: proto.model_version,
            doc_string: proto.doc_string.clone(),
            metadata: decode_kv_entries(&proto.metadata_props),
            graph: decode_graph(graph)?,
            functions_raw: proto
                .functions
                .iter()
                .map(|f| FunctionRaw {
                    domain: f.domain.clone(),
                    name: f.name.clone(),
                    overload: f.overload.clone(),
                    proto: f.encode_to_vec(),
                })
                .collect(),
            training_info_raw: proto
                .training_info
                .iter()
                .map(|t| TrainingInfoRaw {
                    proto: t.encode_to_vec(),
                })
                .collect(),
            configurations: proto
                .configuration
                .iter()
                .map(|c| crate::ir::DeviceConfiguration {
                    name: c.name.clone(),
                    num_devices: c.num_devices,
                    devices: c.device.clone(),
                })
                .collect(),
        })
    }
}

fn decode_opset_id(o: &OperatorSetIdProto) -> OpsetId {
    OpsetId {
        domain: o.domain.clone(),
        version: o.version,
    }
}

fn encode_node(node: &Node) -> NodeProto {
    NodeProto {
        input: node.inputs.clone(),
        output: node.outputs.clone(),
        name: node.name.clone().unwrap_or_default(),
        op_type: node.op_type.clone(),
        domain: node.domain.clone(),
        overload: node.overload.clone(),
        attribute: node.attributes.iter().map(encode_attribute).collect(),
        doc_string: node.doc_string.clone(),
        metadata_props: kv_entries(&node.metadata),
        device_configurations: node
            .device_configurations
            .iter()
            .map(encode_node_device_configuration)
            .collect(),
    }
}

fn encode_node_device_configuration(
    config: &crate::ir::NodeDeviceConfiguration,
) -> NodeDeviceConfigurationProto {
    NodeDeviceConfigurationProto {
        configuration_id: config.configuration_id.clone(),
        sharding_spec: config
            .sharding_specs_raw
            .iter()
            .map(|bytes| ShardingSpecProto::decode(bytes.as_slice()).unwrap_or_default())
            .collect(),
        pipeline_stage: config.pipeline_stage,
    }
}

fn decode_node_device_configuration(
    proto: &NodeDeviceConfigurationProto,
) -> crate::ir::NodeDeviceConfiguration {
    crate::ir::NodeDeviceConfiguration {
        configuration_id: proto.configuration_id.clone(),
        pipeline_stage: proto.pipeline_stage,
        sharding_specs_raw: proto
            .sharding_spec
            .iter()
            .map(|s| s.encode_to_vec())
            .collect(),
    }
}

fn decode_node(proto: &NodeProto) -> Result<Node> {
    let attributes = proto
        .attribute
        .iter()
        .map(decode_attribute)
        .collect::<core::result::Result<Vec<_>, _>>()?;
    Ok(Node {
        name: if proto.name.is_empty() {
            None
        } else {
            Some(proto.name.clone())
        },
        op_type: proto.op_type.clone(),
        domain: proto.domain.clone(),
        inputs: proto.input.clone(),
        outputs: proto.output.clone(),
        attributes,
        overload: proto.overload.clone(),
        doc_string: proto.doc_string.clone(),
        metadata: decode_kv_entries(&proto.metadata_props),
        device_configurations: proto
            .device_configurations
            .iter()
            .map(decode_node_device_configuration)
            .collect(),
    })
}

fn encode_attribute(attr: &Attribute) -> AttributeProto {
    let mut proto = AttributeProto {
        name: attr.name.clone(),
        doc_string: attr.doc_string.clone(),
        ..Default::default()
    };
    match &attr.value {
        AttributeValue::Float(v) => {
            proto.r#type = attribute_proto::AttributeType::Float as i32;
            proto.f = *v;
        }
        AttributeValue::Int(v) => {
            proto.r#type = attribute_proto::AttributeType::Int as i32;
            proto.i = *v;
        }
        AttributeValue::String(v) => {
            proto.r#type = attribute_proto::AttributeType::String as i32;
            proto.s = v.as_bytes().to_vec();
        }
        AttributeValue::Tensor(t) => {
            proto.r#type = attribute_proto::AttributeType::Tensor as i32;
            proto.t = Some(encode_tensor(t));
        }
        AttributeValue::Graph(g) => {
            proto.r#type = attribute_proto::AttributeType::Graph as i32;
            proto.g = GraphProto::decode(g.proto.as_slice()).ok().or_else(|| {
                Some(GraphProto {
                    name: g.name.clone(),
                    ..Default::default()
                })
            });
        }
        AttributeValue::SparseTensor(s) => {
            proto.r#type = attribute_proto::AttributeType::SparseTensor as i32;
            proto.sparse_tensor = SparseTensorProto::decode(s.proto.as_slice()).ok();
        }
        AttributeValue::TypeProto(vt) => {
            proto.r#type = attribute_proto::AttributeType::TypeProto as i32;
            proto.tp = Some(encode_value_type(vt));
        }
        AttributeValue::Floats(v) => {
            proto.r#type = attribute_proto::AttributeType::Floats as i32;
            proto.floats = v.clone();
        }
        AttributeValue::Ints(v) => {
            proto.r#type = attribute_proto::AttributeType::Ints as i32;
            proto.ints = v.clone();
        }
        AttributeValue::Strings(v) => {
            proto.r#type = attribute_proto::AttributeType::Strings as i32;
            proto.strings = v.iter().map(|s| s.as_bytes().to_vec()).collect();
        }
        AttributeValue::Tensors(v) => {
            proto.r#type = attribute_proto::AttributeType::Tensors as i32;
            proto.tensors = v.iter().map(encode_tensor).collect();
        }
        AttributeValue::Graphs(gs) => {
            proto.r#type = attribute_proto::AttributeType::Graphs as i32;
            proto.graphs = gs
                .iter()
                .map(|g| GraphProto::decode(g.proto.as_slice()).unwrap_or_default())
                .collect();
        }
        AttributeValue::SparseTensors(ss) => {
            proto.r#type = attribute_proto::AttributeType::SparseTensors as i32;
            proto.sparse_tensors = ss
                .iter()
                .map(|s| SparseTensorProto::decode(s.proto.as_slice()).unwrap_or_default())
                .collect();
        }
        AttributeValue::TypeProtos(ts) => {
            proto.r#type = attribute_proto::AttributeType::TypeProtos as i32;
            proto.type_protos = ts.iter().map(encode_value_type).collect();
        }
    }
    proto
}

fn decode_value_info(proto: &ValueInfoProto) -> Result<ValueInfo> {
    let value_type = match &proto.r#type {
        Some(tp) => {
            Some(
                decode_value_type(tp).map_err(|detail| ProtoCodecError::InvalidTypeProto {
                    name: proto.name.clone(),
                    detail,
                })?,
            )
        }
        None => None,
    };
    Ok(ValueInfo {
        name: proto.name.clone(),
        value_type,
        doc_string: proto.doc_string.clone(),
    })
}

fn decode_attribute(proto: &AttributeProto) -> Result<Attribute> {
    let mut fields = AttributeFields {
        float: None,
        int: None,
        string_bytes: None,
        tensor: None,
        graph: None,
        floats: None,
        ints: None,
        strings: None,
        tensors: None,
        graphs: None,
    };
    if !proto.floats.is_empty() {
        fields.floats = Some(proto.floats.clone());
    }
    if !proto.ints.is_empty() {
        fields.ints = Some(proto.ints.clone());
    }
    if !proto.strings.is_empty() {
        fields.strings = Some(
            proto
                .strings
                .iter()
                .map(|b| String::from_utf8_lossy(b).into_owned())
                .collect(),
        );
    }
    if let Some(t) = &proto.t {
        fields.tensor = Some(decode_tensor(t)?);
    }
    if let Some(g) = &proto.g {
        fields.graph = Some(GraphRef {
            name: g.name.clone(),
            proto: g.encode_to_vec(),
        });
    }
    if !proto.tensors.is_empty() {
        fields.tensors = Some(
            proto
                .tensors
                .iter()
                .map(decode_tensor)
                .collect::<core::result::Result<Vec<_>, _>>()?,
        );
    }
    if !proto.graphs.is_empty() {
        fields.graphs = Some(
            proto
                .graphs
                .iter()
                .map(|g| GraphRef {
                    name: g.name.clone(),
                    proto: g.encode_to_vec(),
                })
                .collect(),
        );
    }
    if !proto.s.is_empty() {
        fields.string_bytes = Some(proto.s.clone());
    }
    if proto.f != 0.0 {
        fields.float = Some(proto.f);
    }
    if proto.i != 0 {
        fields.int = Some(proto.i);
    }
    if fields.float.is_some() && fields.int.is_some() {
        return Err(ProtoCodecError::Attribute(
            crate::ir::AttributeError::ConflictingValues {
                name: proto.name.clone(),
            },
        ));
    }
    if fields.float.is_none()
        && fields.int.is_none()
        && fields.string_bytes.is_none()
        && fields.tensor.is_none()
        && fields.graph.is_none()
        && fields.floats.is_none()
        && fields.ints.is_none()
        && fields.strings.is_none()
        && fields.tensors.is_none()
        && fields.graphs.is_none()
    {
        return Err(ProtoCodecError::Protobuf(format!(
            "attribute {:?} carries no value",
            proto.name
        )));
    }
    crate::ir::attribute_from_fields(proto.name.clone(), fields).map_err(ProtoCodecError::Attribute)
}

fn encode_value_info(vi: &ValueInfo) -> ValueInfoProto {
    ValueInfoProto {
        name: vi.name.clone(),
        r#type: vi.value_type.as_ref().map(encode_value_type),
        doc_string: vi.doc_string.clone(),
        metadata_props: Vec::new(),
    }
}

fn dim_fixed(d: &Dim) -> Option<i64> {
    match d {
        Dim::Fixed(v) => Some(*v),
        _ => None,
    }
}

fn encode_tensor(tensor: &Tensor) -> TensorProto {
    let mut proto = TensorProto {
        dims: tensor.shape.iter().filter_map(dim_fixed).collect(),
        data_type: tensor.elem.disc(),
        name: tensor.name.clone(),
        doc_string: tensor.doc_string.clone(),
        ..Default::default()
    };
    match &tensor.data {
        TensorStorage::Inline(data) => encode_tensor_data(data, &mut proto),
        TensorStorage::External(external) => {
            proto.data_location = tensor_proto::DataLocation::External as i32;
            proto.external_data = external
                .to_entries()
                .into_iter()
                .map(|(k, v)| StringStringEntryProto { key: k, value: v })
                .collect();
        }
    }
    proto
}

fn encode_tensor_data(data: &TensorData, proto: &mut TensorProto) {
    match data {
        TensorData::F32(v) => proto.float_data = v.clone(),
        TensorData::F64(v) => proto.double_data = v.clone(),
        TensorData::I64(v) => proto.int64_data = v.clone(),
        TensorData::U32(v) => proto.uint64_data = v.iter().map(|&x| x as u64).collect(),
        TensorData::U64(v) => proto.uint64_data = v.clone(),
        TensorData::Complex64(v) => proto.float_data = v.clone(),
        TensorData::Complex128(v) => proto.double_data = v.clone(),
        TensorData::String(v) => {
            proto.string_data = v.iter().map(|s| s.as_bytes().to_vec()).collect()
        }
        TensorData::Bool(v) => proto.int32_data = v.iter().map(|&b| i32::from(b)).collect(),
        TensorData::I8(v) => proto.int32_data = v.iter().map(|&x| x as i32).collect(),
        TensorData::I16(v) => proto.int32_data = v.iter().map(|&x| x as i32).collect(),
        TensorData::I32(v) => proto.int32_data = v.clone(),
        TensorData::U8(v) => proto.int32_data = v.iter().map(|&x| x as i32).collect(),
        TensorData::U16(v) => proto.int32_data = v.iter().map(|&x| x as i32).collect(),
        TensorData::F16(v) => proto.int32_data = v.iter().map(|&x| x as i32).collect(),
        TensorData::Bf16(v) => proto.int32_data = v.iter().map(|&x| x as i32).collect(),
        TensorData::Float8e4m3fn(v)
        | TensorData::Float8e4m3fnuz(v)
        | TensorData::Float8e5m2(v)
        | TensorData::Float8e5m2fnuz(v)
        | TensorData::Float8e8m0(v)
        | TensorData::Uint4(v)
        | TensorData::Int4(v)
        | TensorData::Float4e2m1(v)
        | TensorData::Uint2(v)
        | TensorData::Int2(v) => proto.int32_data = v.iter().map(|&x| x as i32).collect(),
        TensorData::RawBytes { bytes, .. } => proto.raw_data = bytes.clone(),
    }
}

fn decode_tensor(proto: &TensorProto) -> Result<Tensor> {
    let name = proto.name.clone();
    let elem =
        elem_from_disc(proto.data_type).map_err(|detail| ProtoCodecError::TensorDataMismatch {
            name: name.clone(),
            detail,
        })?;
    let shape: Vec<Dim> = proto.dims.iter().map(|&d| Dim::Fixed(d)).collect();

    if !proto.external_data.is_empty()
        || proto.data_location == tensor_proto::DataLocation::External as i32
    {
        let external = ExternalData::from_entries(
            &proto
                .external_data
                .iter()
                .map(|e| (e.key.clone(), e.value.clone()))
                .collect::<Vec<_>>(),
        )
        .map_err(|e| ProtoCodecError::InvalidExternalData {
            name: name.clone(),
            detail: e.to_string(),
        })?;
        return Ok(Tensor {
            name,
            elem,
            shape,
            data: TensorStorage::External(external),
            doc_string: proto.doc_string.clone(),
        });
    }

    let data = decode_tensor_data(elem, proto, &name)?;
    Ok(Tensor {
        name,
        elem,
        shape,
        data: TensorStorage::Inline(data),
        doc_string: proto.doc_string.clone(),
    })
}

fn decode_tensor_data(elem: ElemType, proto: &TensorProto, name: &str) -> Result<TensorData> {
    let expected = proto
        .dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d.max(0) as usize))
        .unwrap_or(usize::MAX);

    let check_len = |got: usize| -> Result<()> {
        if !proto.dims.is_empty() && got != expected {
            return Err(ProtoCodecError::TensorSizeMismatch {
                name: name.to_string(),
                expected,
                got,
            });
        }
        Ok(())
    };

    let raw = proto.raw_data.as_slice();
    if !raw.is_empty() {
        if elem == ElemType::String {
            return Err(ProtoCodecError::TensorDataMismatch {
                name: name.to_string(),
                detail: "STRING tensors must use string_data, not raw_data".to_string(),
            });
        }
        let data = decode_raw_data(elem, raw, name)?;
        if let Some(n) = data.len()
            && !proto.dims.is_empty()
            && n != expected
        {
            return Err(ProtoCodecError::TensorSizeMismatch {
                name: name.to_string(),
                expected,
                got: n,
            });
        }
        return Ok(data);
    }
    if !proto.string_data.is_empty() {
        if elem != ElemType::String {
            return Err(ProtoCodecError::TensorDataMismatch {
                name: name.to_string(),
                detail: format!(
                    "string_data present but dtype is {} ({elem:?})",
                    elem.name()
                ),
            });
        }
        let strings: Vec<String> = proto
            .string_data
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect();
        check_len(strings.len())?;
        return Ok(TensorData::String(strings));
    }

    match elem {
        ElemType::Float => {
            check_len(proto.float_data.len())?;
            Ok(TensorData::F32(proto.float_data.clone()))
        }
        ElemType::Double => {
            check_len(proto.double_data.len())?;
            Ok(TensorData::F64(proto.double_data.clone()))
        }
        ElemType::Complex64 => {
            check_len(proto.float_data.len())?;
            Ok(TensorData::Complex64(proto.float_data.clone()))
        }
        ElemType::Complex128 => {
            check_len(proto.double_data.len())?;
            Ok(TensorData::Complex128(proto.double_data.clone()))
        }
        ElemType::Int64 => {
            check_len(proto.int64_data.len())?;
            Ok(TensorData::I64(proto.int64_data.clone()))
        }
        ElemType::Uint32 => {
            check_len(proto.uint64_data.len())?;
            Ok(TensorData::U32(
                proto.uint64_data.iter().map(|&x| x as u32).collect(),
            ))
        }
        ElemType::Uint64 => {
            check_len(proto.uint64_data.len())?;
            Ok(TensorData::U64(proto.uint64_data.clone()))
        }
        ElemType::Int8
        | ElemType::Int16
        | ElemType::Int32
        | ElemType::Uint8
        | ElemType::Uint16
        | ElemType::Bool
        | ElemType::Float16
        | ElemType::Bfloat16
        | ElemType::Float8e4m3fn
        | ElemType::Float8e4m3fnuz
        | ElemType::Float8e5m2
        | ElemType::Float8e5m2fnuz
        | ElemType::Float8e8m0
        | ElemType::Uint4
        | ElemType::Int4
        | ElemType::Float4e2m1
        | ElemType::Uint2
        | ElemType::Int2 => decode_int32_data(elem, &proto.int32_data, name),
        ElemType::String => Err(ProtoCodecError::TensorDataMismatch {
            name: name.to_string(),
            detail: "STRING tensor carries no string_data".to_string(),
        }),
        ElemType::Undefined => Err(ProtoCodecError::TensorDataMismatch {
            name: name.to_string(),
            detail: "tensor with UNDEFINED data type".to_string(),
        }),
    }
}

fn decode_raw_data(elem: ElemType, raw: &[u8], name: &str) -> Result<TensorData> {
    if !elem.is_fixed_width() {
        check_raw_len(elem, raw.len(), raw.len(), name)?;
        return Ok(TensorData::RawBytes {
            elem,
            bytes: raw.to_vec(),
        });
    }
    let width = raw_elem_width(elem);
    if !raw.len().is_multiple_of(width) {
        return Err(ProtoCodecError::TensorSizeMismatch {
            name: name.to_string(),
            expected: raw.len().div_ceil(width),
            got: raw.len(),
        });
    }
    let n = raw.len() / width;
    check_raw_len(elem, n, n, name)?;
    macro_rules! le {
        ($t:ty) => {
            chunk_map(raw, width, |b| <$t>::from_le_bytes(b.try_into().unwrap()))
        };
    }
    let data = match elem {
        ElemType::Float => TensorData::F32(le!(f32)),
        ElemType::Double => TensorData::F64(le!(f64)),
        ElemType::Int8 => TensorData::I8(le!(i8)),
        ElemType::Int16 => TensorData::I16(le!(i16)),
        ElemType::Int32 => TensorData::I32(le!(i32)),
        ElemType::Int64 => TensorData::I64(le!(i64)),
        ElemType::Uint8 => TensorData::U8(le!(u8)),
        ElemType::Uint16 => TensorData::U16(le!(u16)),
        ElemType::Uint32 => TensorData::U32(le!(u32)),
        ElemType::Uint64 => TensorData::U64(le!(u64)),
        ElemType::Bool => TensorData::Bool(raw.iter().map(|&b| b != 0).collect()),
        ElemType::Float16 => TensorData::F16(le!(u16)),
        ElemType::Bfloat16 => TensorData::Bf16(le!(u16)),

        ElemType::Complex64 => TensorData::Complex64(le!(f32)),
        ElemType::Complex128 => TensorData::Complex128(le!(f64)),
        _ => TensorData::RawBytes {
            elem,
            bytes: raw.to_vec(),
        },
    };
    Ok(data)
}

fn check_raw_len(elem: ElemType, expected: usize, got: usize, name: &str) -> Result<()> {
    if expected != got {
        return Err(ProtoCodecError::TensorSizeMismatch {
            name: name.to_string(),
            expected,
            got,
        });
    }
    let _ = elem;
    Ok(())
}

fn raw_elem_width(elem: ElemType) -> usize {
    match elem {
        ElemType::Float | ElemType::Int32 | ElemType::Uint32 | ElemType::Complex64 => 4,
        ElemType::Double | ElemType::Int64 | ElemType::Uint64 | ElemType::Complex128 => 8,
        ElemType::Int16 | ElemType::Uint16 | ElemType::Float16 | ElemType::Bfloat16 => 2,
        _ => 1,
    }
}

fn chunk_map<T>(raw: &[u8], width: usize, f: impl Fn(&[u8]) -> T) -> Vec<T> {
    raw.chunks_exact(width).map(f).collect()
}

fn decode_int32_data(elem: ElemType, data: &[i32], name: &str) -> Result<TensorData> {
    let cast = |v: i32| -> u8 { v as u8 };
    let d = match elem {
        ElemType::Int8 => TensorData::I8(data.iter().map(|&v| v as i8).collect()),
        ElemType::Int16 => TensorData::I16(data.iter().map(|&v| v as i16).collect()),
        ElemType::Int32 => TensorData::I32(data.to_vec()),
        ElemType::Uint8 => TensorData::U8(data.iter().map(|&v| cast(v)).collect()),
        ElemType::Uint16 => TensorData::U16(data.iter().map(|&v| v as u16).collect()),
        ElemType::Bool => TensorData::Bool(data.iter().map(|&v| v != 0).collect()),
        ElemType::Float16 => TensorData::F16(data.iter().map(|&v| v as u16).collect()),
        ElemType::Bfloat16 => TensorData::Bf16(data.iter().map(|&v| v as u16).collect()),
        ElemType::Float8e4m3fn => TensorData::Float8e4m3fn(data.iter().map(|&v| cast(v)).collect()),
        ElemType::Float8e4m3fnuz => {
            TensorData::Float8e4m3fnuz(data.iter().map(|&v| cast(v)).collect())
        }
        ElemType::Float8e5m2 => TensorData::Float8e5m2(data.iter().map(|&v| cast(v)).collect()),
        ElemType::Float8e5m2fnuz => {
            TensorData::Float8e5m2fnuz(data.iter().map(|&v| cast(v)).collect())
        }
        ElemType::Float8e8m0 => TensorData::Float8e8m0(data.iter().map(|&v| cast(v)).collect()),
        ElemType::Uint4 => TensorData::Uint4(data.iter().map(|&v| cast(v)).collect()),
        ElemType::Int4 => TensorData::Int4(data.iter().map(|&v| cast(v)).collect()),
        ElemType::Float4e2m1 => TensorData::Float4e2m1(data.iter().map(|&v| cast(v)).collect()),
        ElemType::Uint2 => TensorData::Uint2(data.iter().map(|&v| cast(v)).collect()),
        ElemType::Int2 => TensorData::Int2(data.iter().map(|&v| cast(v)).collect()),
        _ => {
            return Err(ProtoCodecError::TensorDataMismatch {
                name: name.to_string(),
                detail: format!("dtype {} cannot be carried in int32_data", elem.name()),
            });
        }
    };
    Ok(d)
}

pub fn read_external(tensor: &Tensor, base_dir: &Path) -> Result<Tensor> {
    let external = match &tensor.data {
        TensorStorage::External(e) => e,
        TensorStorage::Inline(_) => {
            return Err(ProtoCodecError::InvalidExternalData {
                name: tensor.name.clone(),
                detail: "tensor does not reference external data".to_string(),
            });
        }
    };
    let path = resolve_external_path(&external.location, base_dir)?;
    let file_len = std::fs::metadata(&path)?.len();
    let offset = external.offset.unwrap_or(0);
    if offset > file_len {
        return Err(ProtoCodecError::InvalidExternalData {
            name: tensor.name.clone(),
            detail: format!("offset {offset} beyond file length {file_len}"),
        });
    }
    let avail = file_len - offset;
    let len = match external.length {
        Some(l) => {
            if l > avail {
                return Err(ProtoCodecError::InvalidExternalData {
                    name: tensor.name.clone(),
                    detail: format!("length {l} exceeds remaining file bytes {avail}"),
                });
            }
            l
        }
        None => avail,
    };
    let mut file = std::fs::File::open(&path)?;
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0u8; len as usize];
    file.read_exact(&mut bytes)?;
    let data = decode_raw_data(tensor.elem, &bytes, &tensor.name)?;
    Ok(Tensor {
        name: tensor.name.clone(),
        elem: tensor.elem,
        shape: tensor.shape.clone(),
        data: TensorStorage::Inline(data),
        doc_string: tensor.doc_string.clone(),
    })
}

fn resolve_external_path(location: &str, base_dir: &Path) -> Result<std::path::PathBuf> {
    if location.is_empty() {
        return Err(ProtoCodecError::InvalidExternalData {
            name: String::new(),
            detail: "empty external data location".to_string(),
        });
    }
    let rel = Path::new(location);
    if rel.is_absolute() {
        return Err(ProtoCodecError::InvalidExternalData {
            name: location.to_string(),
            detail: "external data location must be relative".to_string(),
        });
    }
    for comp in rel.components() {
        match comp {
            std::path::Component::Normal(_) => {}
            other => {
                return Err(ProtoCodecError::InvalidExternalData {
                    name: location.to_string(),
                    detail: format!("external data location contains unsafe component {other:?}"),
                });
            }
        }
    }
    Ok(base_dir.join(rel))
}

fn encode_value_type(vt: &ValueType) -> TypeProto {
    let value = match vt {
        ValueType::Tensor(t) => type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: t.elem.disc(),
            shape: t.shape.as_deref().map(encode_shape),
        }),
        ValueType::SparseTensor(t) => {
            type_proto::Value::SparseTensorType(type_proto::SparseTensor {
                elem_type: t.elem.disc(),
                shape: t.shape.as_deref().map(encode_shape),
            })
        }
        ValueType::Sequence(inner) => {
            type_proto::Value::SequenceType(Box::new(type_proto::Sequence {
                elem_type: Some(Box::new(encode_value_type(inner))),
            }))
        }
        ValueType::Map { key, value } => type_proto::Value::MapType(Box::new(type_proto::Map {
            key_type: key.disc(),
            value_type: Some(Box::new(encode_value_type(value))),
        })),
        ValueType::Optional(inner) => {
            type_proto::Value::OptionalType(Box::new(type_proto::Optional {
                elem_type: Some(Box::new(encode_value_type(inner))),
            }))
        }
        ValueType::Opaque { domain, name } => type_proto::Value::OpaqueType(type_proto::Opaque {
            domain: domain.clone().unwrap_or_default(),
            name: name.clone().unwrap_or_default(),
        }),
    };
    TypeProto {
        denotation: String::new(),
        value: Some(value),
    }
}

fn encode_shape(shape: &[Dim]) -> TensorShapeProto {
    TensorShapeProto {
        dim: shape
            .iter()
            .map(|d| tensor_shape_proto::Dimension {
                denotation: String::new(),
                value: match d {
                    Dim::Fixed(v) => Some(tensor_shape_proto::dimension::Value::DimValue(*v)),
                    Dim::Param(p) => {
                        Some(tensor_shape_proto::dimension::Value::DimParam(p.clone()))
                    }
                    Dim::Unknown => None,
                },
            })
            .collect(),
    }
}

fn decode_value_type(proto: &TypeProto) -> core::result::Result<ValueType, String> {
    match &proto.value {
        Some(type_proto::Value::TensorType(t)) => {
            let elem = elem_from_disc(t.elem_type)?;
            let shape = decode_shape(t.shape.as_ref());
            Ok(ValueType::Tensor(crate::ir::TensorType { elem, shape }))
        }
        Some(type_proto::Value::SparseTensorType(t)) => {
            let elem = elem_from_disc(t.elem_type)?;
            let shape = decode_shape(t.shape.as_ref());
            Ok(ValueType::SparseTensor(crate::ir::TensorType {
                elem,
                shape,
            }))
        }
        Some(type_proto::Value::SequenceType(s)) => {
            let inner = s
                .elem_type
                .as_ref()
                .ok_or_else(|| "sequence without element type".to_string())?;
            Ok(ValueType::Sequence(Box::new(decode_value_type(inner)?)))
        }
        Some(type_proto::Value::MapType(m)) => {
            let key = elem_from_disc(m.key_type)?;
            let value = m
                .value_type
                .as_ref()
                .ok_or_else(|| "map without value type".to_string())?;
            Ok(ValueType::Map {
                key,
                value: Box::new(decode_value_type(value)?),
            })
        }
        Some(type_proto::Value::OptionalType(o)) => {
            let inner = o
                .elem_type
                .as_ref()
                .ok_or_else(|| "optional without element type".to_string())?;
            Ok(ValueType::Optional(Box::new(decode_value_type(inner)?)))
        }
        Some(type_proto::Value::OpaqueType(o)) => Ok(ValueType::Opaque {
            domain: if o.domain.is_empty() {
                None
            } else {
                Some(o.domain.clone())
            },
            name: if o.name.is_empty() {
                None
            } else {
                Some(o.name.clone())
            },
        }),
        None => Err("type proto without value".to_string()),
    }
}

fn decode_shape(proto: Option<&TensorShapeProto>) -> Option<Vec<Dim>> {
    proto.map(|s| {
        s.dim
            .iter()
            .map(|d| match &d.value {
                Some(tensor_shape_proto::dimension::Value::DimValue(v)) => Dim::Fixed(*v),
                Some(tensor_shape_proto::dimension::Value::DimParam(p)) => Dim::Param(p.clone()),
                None => Dim::Unknown,
            })
            .collect()
    })
}

fn elem_from_disc(v: i32) -> core::result::Result<ElemType, String> {
    ElemType::from_disc(v).ok_or_else(|| format!("unknown element type discriminant {v}"))
}

fn encode_graph(graph: &Graph) -> GraphProto {
    GraphProto {
        node: graph.nodes.iter().map(encode_node).collect(),
        name: graph.name.clone(),
        initializer: graph.initializers.iter().map(encode_tensor).collect(),
        sparse_initializer: graph
            .sparse_initializers_raw
            .iter()
            .map(|s| SparseTensorProto::decode(s.proto.as_slice()).unwrap_or_default())
            .collect(),
        doc_string: graph.doc_string.clone(),
        input: graph.inputs.iter().map(encode_value_info).collect(),
        output: graph.outputs.iter().map(encode_value_info).collect(),
        value_info: graph.value_info.iter().map(encode_value_info).collect(),
        quantization_annotation: graph
            .quantization_annotations_raw
            .iter()
            .map(|a| crate::proto::TensorAnnotation {
                tensor_name: String::new(),
                quant_parameter_tensor_names: kv_entries(&a.entries),
            })
            .collect(),
        metadata_props: kv_entries(&graph.metadata),
    }
}

fn decode_graph(proto: &GraphProto) -> Result<Graph> {
    let initializers = proto
        .initializer
        .iter()
        .map(decode_tensor)
        .collect::<core::result::Result<Vec<_>, _>>()?;
    Ok(Graph {
        name: proto.name.clone(),
        nodes: proto
            .node
            .iter()
            .map(decode_node)
            .collect::<core::result::Result<Vec<_>, _>>()?,
        initializers,
        sparse_initializers_raw: proto
            .sparse_initializer
            .iter()
            .map(|s| SparseTensorRaw {
                name: s
                    .values
                    .as_ref()
                    .and_then(|v| {
                        if v.name.is_empty() {
                            None
                        } else {
                            Some(v.name.clone())
                        }
                    })
                    .unwrap_or_default(),
                proto: s.encode_to_vec(),
            })
            .collect(),
        inputs: proto
            .input
            .iter()
            .map(decode_value_info)
            .collect::<core::result::Result<Vec<_>, _>>()?,
        outputs: proto
            .output
            .iter()
            .map(decode_value_info)
            .collect::<core::result::Result<Vec<_>, _>>()?,
        value_info: proto
            .value_info
            .iter()
            .map(decode_value_info)
            .collect::<core::result::Result<Vec<_>, _>>()?,
        quantization_annotations_raw: proto
            .quantization_annotation
            .iter()
            .map(|a| crate::ir::RawKvEntries {
                entries: decode_kv_entries(&a.quant_parameter_tensor_names),
            })
            .collect(),
        metadata: decode_kv_entries(&proto.metadata_props),
        doc_string: proto.doc_string.clone(),
    })
}

fn encode_model_proto(model: &Model) -> ModelProto {
    ModelProto {
        ir_version: model.ir_version,
        opset_import: model.opset_import.iter().map(encode_opset_id).collect(),
        producer_name: model.producer_name.clone(),
        producer_version: model.producer_version.clone(),
        domain: model.domain.clone(),
        model_version: model.model_version,
        doc_string: model.doc_string.clone(),
        graph: Some(encode_graph(&model.graph)),
        metadata_props: kv_entries(&model.metadata),
        training_info: model
            .training_info_raw
            .iter()
            .map(|t| TrainingInfoProto::decode(t.proto.as_slice()).unwrap_or_default())
            .collect(),
        functions: model
            .functions_raw
            .iter()
            .map(|f| FunctionProto {
                domain: f.domain.clone(),
                name: f.name.clone(),
                overload: f.overload.clone(),
                ..Default::default()
            })
            .collect(),
        configuration: model
            .configurations
            .iter()
            .map(|c| DeviceConfigurationProto {
                name: c.name.clone(),
                num_devices: c.num_devices,
                device: c.devices.clone(),
            })
            .collect(),
    }
}
