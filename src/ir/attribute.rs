use super::tensor::Tensor;
use super::types::ValueType;

#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    Float(f32),
    Int(i64),
    String(String),
    Tensor(Box<Tensor>),
    Graph(GraphRef),
    SparseTensor(SparseRef),
    TypeProto(Box<ValueType>),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
    Tensors(Vec<Tensor>),
    Graphs(Vec<GraphRef>),
    SparseTensors(Vec<SparseRef>),
    TypeProtos(Vec<ValueType>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphRef {
    pub name: String,
    pub proto: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseRef {
    pub name: String,
    pub proto: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    pub name: String,
    pub value: AttributeValue,
    pub doc_string: String,
}

impl Attribute {
    pub fn new(name: impl Into<String>, value: AttributeValue) -> Self {
        Attribute {
            name: name.into(),
            value,
            doc_string: String::new(),
        }
    }

    pub fn float(name: impl Into<String>, v: f32) -> Self {
        Self::new(name, AttributeValue::Float(v))
    }

    pub fn int(name: impl Into<String>, v: i64) -> Self {
        Self::new(name, AttributeValue::Int(v))
    }

    pub fn string(name: impl Into<String>, v: impl Into<String>) -> Self {
        Self::new(name, AttributeValue::String(v.into()))
    }

    pub fn tensor(name: impl Into<String>, t: Tensor) -> Self {
        Self::new(name, AttributeValue::Tensor(Box::new(t)))
    }

    pub fn graph(name: impl Into<String>, g: GraphRef) -> Self {
        Self::new(name, AttributeValue::Graph(g))
    }

    pub fn floats(name: impl Into<String>, v: Vec<f32>) -> Self {
        Self::new(name, AttributeValue::Floats(v))
    }

    pub fn ints(name: impl Into<String>, v: Vec<i64>) -> Self {
        Self::new(name, AttributeValue::Ints(v))
    }

    pub fn strings(name: impl Into<String>, v: Vec<String>) -> Self {
        Self::new(name, AttributeValue::Strings(v))
    }

    pub fn tensors(name: impl Into<String>, v: Vec<Tensor>) -> Self {
        Self::new(name, AttributeValue::Tensors(v))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttributeFields {
    pub float: Option<f32>,
    pub int: Option<i64>,
    pub string_bytes: Option<Vec<u8>>,
    pub tensor: Option<Tensor>,
    pub graph: Option<GraphRef>,
    pub floats: Option<Vec<f32>>,
    pub ints: Option<Vec<i64>>,
    pub strings: Option<Vec<String>>,
    pub tensors: Option<Vec<Tensor>>,
    pub graphs: Option<Vec<GraphRef>>,
}

pub fn attribute_from_fields(
    name: String,
    fields: AttributeFields,
) -> Result<Attribute, AttributeError> {
    let count = [
        fields.float.is_some(),
        fields.int.is_some(),
        fields.string_bytes.is_some(),
        fields.tensor.is_some(),
        fields.graph.is_some(),
        fields.floats.is_some(),
        fields.ints.is_some(),
        fields.strings.is_some(),
        fields.tensors.is_some(),
        fields.graphs.is_some(),
    ]
    .iter()
    .filter(|b| **b)
    .count();

    if count > 1 {
        return Err(AttributeError::ConflictingValues { name });
    }

    let value = if let Some(float_value) = fields.float {
        AttributeValue::Float(float_value)
    } else if let Some(int_value) = fields.int {
        AttributeValue::Int(int_value)
    } else if let Some(string_bytes) = fields.string_bytes {
        match String::from_utf8(string_bytes) {
            Ok(v) => AttributeValue::String(v),
            Err(_) => return Err(AttributeError::InvalidUtf8String(name)),
        }
    } else if let Some(tensor) = fields.tensor {
        AttributeValue::Tensor(Box::new(tensor))
    } else if let Some(graph) = fields.graph {
        AttributeValue::Graph(graph)
    } else if let Some(v) = fields.floats {
        AttributeValue::Floats(v)
    } else if let Some(v) = fields.ints {
        AttributeValue::Ints(v)
    } else if let Some(v) = fields.strings {
        AttributeValue::Strings(v)
    } else if let Some(v) = fields.tensors {
        AttributeValue::Tensors(v)
    } else if let Some(v) = fields.graphs {
        AttributeValue::Graphs(v)
    } else {
        return Err(AttributeError::NoValue { name });
    };

    Ok(Attribute {
        name,
        value,
        doc_string: String::new(),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum AttributeError {
    ConflictingValues { name: String },
    NoValue { name: String },
    InvalidUtf8String(String),
}

impl core::fmt::Display for AttributeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ConflictingValues { name } => {
                write!(f, "attribute {name:?} has more than one value field set")
            }
            Self::NoValue { name } => write!(f, "attribute {name:?} has no value field set"),
            Self::InvalidUtf8String(name) => {
                write!(f, "attribute {name:?} string value is not valid UTF-8")
            }
        }
    }
}

impl std::error::Error for AttributeError {}
