use super::attribute::{Attribute, GraphRef};
use super::tensor::Tensor;
use super::types::ValueType;

pub const DEFAULT_DOMAIN: &str = "";
pub const ML_DOMAIN: &str = "ai.onnx.ml";

#[derive(Debug, Clone, PartialEq)]
pub struct NodeDeviceConfiguration {
    pub configuration_id: String,
    pub pipeline_stage: i32,
    pub sharding_specs_raw: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub name: Option<String>,
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<Attribute>,
    pub overload: String,
    pub doc_string: String,
    pub metadata: Vec<(String, String)>,
    pub device_configurations: Vec<NodeDeviceConfiguration>,
}

impl Node {
    pub fn new(
        op_type: impl Into<String>,
        domain: impl Into<String>,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: Vec<Attribute>,
    ) -> Self {
        Node {
            name: None,
            op_type: op_type.into(),
            domain: domain.into(),
            inputs,
            outputs,
            attributes,
            overload: String::new(),
            doc_string: String::new(),
            metadata: Vec::new(),
            device_configurations: Vec::new(),
        }
    }

    pub fn attr(&self, name: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    pub fn attr_mut(&mut self, name: &str) -> Option<&mut Attribute> {
        self.attributes.iter_mut().find(|a| a.name == name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValueInfo {
    pub name: String,
    pub value_type: Option<ValueType>,
    pub doc_string: String,
}

impl ValueInfo {
    pub fn new(name: impl Into<String>, value_type: ValueType) -> Self {
        ValueInfo {
            name: name.into(),
            value_type: Some(value_type),
            doc_string: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    pub name: String,
    pub nodes: Vec<Node>,
    pub initializers: Vec<Tensor>,
    pub sparse_initializers_raw: Vec<SparseTensorRaw>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub value_info: Vec<ValueInfo>,
    pub quantization_annotations_raw: Vec<RawKvEntries>,
    pub metadata: Vec<(String, String)>,
    pub doc_string: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseTensorRaw {
    pub name: String,
    pub proto: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RawKvEntries {
    pub entries: Vec<(String, String)>,
}

impl Graph {
    pub fn new(name: impl Into<String>) -> Self {
        Graph {
            name: name.into(),
            nodes: Vec::new(),
            initializers: Vec::new(),
            sparse_initializers_raw: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            value_info: Vec::new(),
            quantization_annotations_raw: Vec::new(),
            metadata: Vec::new(),
            doc_string: String::new(),
        }
    }

    pub fn initializer(&self, name: &str) -> Option<&Tensor> {
        self.initializers.iter().find(|t| t.name == name)
    }

    pub fn value_info(&self, name: &str) -> Option<&ValueInfo> {
        self.value_info.iter().find(|v| v.name == name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionRaw {
    pub domain: String,
    pub name: String,
    pub overload: String,
    pub proto: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingInfoRaw {
    pub proto: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceConfiguration {
    pub name: String,
    pub num_devices: i32,
    pub devices: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub ir_version: i64,
    pub opset_import: Vec<OpsetId>,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub metadata: Vec<(String, String)>,
    pub graph: Graph,
    pub functions_raw: Vec<FunctionRaw>,
    pub training_info_raw: Vec<TrainingInfoRaw>,
    pub configurations: Vec<DeviceConfiguration>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpsetId {
    pub domain: String,
    pub version: i64,
}

impl Model {
    pub fn new(graph: Graph, opset_import: Vec<OpsetId>) -> Self {
        Model {
            ir_version: super::types::IR_VERSION,
            opset_import,
            producer_name: "serde-onnx".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            metadata: Vec::new(),
            graph,
            functions_raw: Vec::new(),
            training_info_raw: Vec::new(),
            configurations: Vec::new(),
        }
    }

    pub fn opset(&self, domain: &str) -> Option<i64> {
        self.opset_import
            .iter()
            .find(|o| o.domain == domain)
            .map(|o| o.version)
    }
}

impl From<GraphRef> for Vec<u8> {
    fn from(g: GraphRef) -> Self {
        g.proto
    }
}
