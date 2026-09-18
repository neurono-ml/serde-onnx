mod aionnx;
mod linear;
mod payload;
mod preproc;
mod trees;

#[cfg(test)]
mod tests;

pub use aionnx::{Cast, Concat, Gather, Identity, Reshape};
pub use linear::{
    ClassLabels, LinearClassifier, LinearRegressor, SvmClassifier, SvmCommon, SvmRegressor,
};
pub use payload::{NodePayload, TypedNode, make_node_payload};
pub use preproc::{
    DictVectorizer, FeatureVectorizer, Imputer, LabelEncoder, Normalizer, OneHotEncoder, Scaler,
};
pub use trees::{TreeEnsembleClassifier, TreeEnsembleRegressor, TreeNodes, ZipMap};

use crate::ir::{Attribute, AttributeValue, Node, Tensor};

pub const ML_EXPORT_OPSET_TARGET: i64 = 4;

pub const CORE_TYPED_OPSET_TARGET: i64 = 21;

pub const CODEC_MAX_IR_VERSION: i64 = crate::proto::SUPPORTED_IR_VERSION;

pub const CODEC_MAX_ONNX_OPSET: i64 = crate::proto::SUPPORTED_ONNX_OPSET;

pub const CODEC_MAX_ML_OPSET: i64 = crate::proto::SUPPORTED_ML_OPSET;

pub const ML_DOMAIN_STR: &str = crate::ir::ML_DOMAIN;

pub const ONNX_DOMAIN_STR: &str = "";

#[derive(Debug, Clone, PartialEq)]
pub enum OpError {
    WrongOp {
        expected_domain: &'static str,
        expected_op: &'static str,
        got_domain: String,
        got_op: String,
    },
    MissingAttribute {
        op: &'static str,
        attr: &'static str,
    },
    WrongAttributeType {
        op: &'static str,
        attr: String,
        expected: &'static str,
    },
    DuplicateAttribute {
        op: &'static str,
        attr: String,
    },
    UnknownAttribute {
        op: &'static str,
        attr: String,
    },
    WrongInputCount {
        op: &'static str,
        expected: String,
        got: usize,
    },
    WrongOutputCount {
        op: &'static str,
        expected: String,
        got: usize,
    },
    InvalidValue {
        op: &'static str,
        attr: String,
        detail: String,
    },
}

impl core::fmt::Display for OpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongOp {
                expected_domain,
                expected_op,
                got_domain,
                got_op,
            } => write!(
                f,
                "node ({got_domain:?}, {got_op:?}) is not ({expected_domain:?}, {expected_op:?})"
            ),
            Self::MissingAttribute { op, attr } => {
                write!(f, "op {op} is missing required attribute {attr:?}")
            }
            Self::WrongAttributeType { op, attr, expected } => write!(
                f,
                "op {op} attribute {attr:?} has the wrong type, expected {expected}"
            ),
            Self::DuplicateAttribute { op, attr } => {
                write!(f, "op {op} has duplicate attribute {attr:?}")
            }
            Self::UnknownAttribute { op, attr } => {
                write!(f, "op {op} has unrecognized attribute {attr:?}")
            }
            Self::WrongInputCount { op, expected, got } => {
                write!(f, "op {op} expects {expected} inputs, got {got}")
            }
            Self::WrongOutputCount { op, expected, got } => {
                write!(f, "op {op} expects {expected} outputs, got {got}")
            }
            Self::InvalidValue { op, attr, detail } => {
                write!(f, "op {op} attribute {attr:?} is invalid: {detail}")
            }
        }
    }
}

impl std::error::Error for OpError {}

#[derive(Debug, Clone, PartialEq)]
pub struct Emitted {
    pub node: Node,
    pub initializers: Vec<Tensor>,
}

impl Emitted {
    pub fn single(node: Node) -> Self {
        Emitted {
            node,
            initializers: Vec::new(),
        }
    }
}

pub trait OnnxOp: Sized + Clone + PartialEq + core::fmt::Debug {
    const OP_TYPE: &'static str;
    const DOMAIN: &'static str;
    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError>;
    fn from_node(node: &Node) -> Result<Self, OpError>;
}

#[cfg(feature = "export")]
pub trait OpSink {
    type Error: core::fmt::Debug;
    fn emit(&mut self, emitted: Emitted) -> Result<(Vec<String>, Vec<String>), Self::Error>;
}

#[cfg(feature = "export")]
pub fn emit_op<S: OpSink, O: OnnxOp>(
    sink: &mut S,
    op: &O,
    inputs: Vec<String>,
    outputs: Vec<String>,
) -> Result<(Vec<String>, Vec<String>), S::Error>
where
    S::Error: From<OpError>,
{
    let emitted = op.to_node(inputs, outputs)?;
    sink.emit(emitted)
}

pub(crate) struct AttrTable<'a> {
    op: &'static str,
    entries: Vec<(&'a str, &'a AttributeValue)>,
}

impl<'a> AttrTable<'a> {
    pub(crate) fn new(op: &'static str, node: &'a Node) -> Result<Self, OpError> {
        let mut entries = Vec::with_capacity(node.attributes.len());
        for attr in &node.attributes {
            if entries.iter().any(|(name, _)| *name == attr.name) {
                return Err(OpError::DuplicateAttribute {
                    op,
                    attr: attr.name.clone(),
                });
            }
            entries.push((attr.name.as_str(), &attr.value));
        }
        Ok(AttrTable { op, entries })
    }

    fn take(&mut self, name: &str) -> Option<&'a AttributeValue> {
        self.entries
            .iter()
            .position(|(n, _)| *n == name)
            .map(|i| self.entries.remove(i).1)
    }

    pub(crate) fn opt_floats(&mut self, name: &str) -> Result<Option<Vec<f32>>, OpError> {
        match self.take(name) {
            None => Ok(None),
            Some(AttributeValue::Floats(v)) => Ok(Some(v.clone())),
            _ => Err(OpError::WrongAttributeType {
                op: self.op,
                attr: name.to_string(),
                expected: "floats",
            }),
        }
    }

    pub(crate) fn opt_ints(&mut self, name: &str) -> Result<Option<Vec<i64>>, OpError> {
        match self.take(name) {
            None => Ok(None),
            Some(AttributeValue::Ints(v)) => Ok(Some(v.clone())),
            _ => Err(OpError::WrongAttributeType {
                op: self.op,
                attr: name.to_string(),
                expected: "ints",
            }),
        }
    }

    pub(crate) fn opt_strings(&mut self, name: &str) -> Result<Option<Vec<String>>, OpError> {
        match self.take(name) {
            None => Ok(None),
            Some(AttributeValue::Strings(v)) => Ok(Some(v.clone())),
            _ => Err(OpError::WrongAttributeType {
                op: self.op,
                attr: name.to_string(),
                expected: "strings",
            }),
        }
    }

    pub(crate) fn opt_float(&mut self, name: &str) -> Result<Option<f32>, OpError> {
        match self.take(name) {
            None => Ok(None),
            Some(AttributeValue::Float(v)) => Ok(Some(*v)),
            _ => Err(OpError::WrongAttributeType {
                op: self.op,
                attr: name.to_string(),
                expected: "float",
            }),
        }
    }

    pub(crate) fn opt_int(&mut self, name: &str) -> Result<Option<i64>, OpError> {
        match self.take(name) {
            None => Ok(None),
            Some(AttributeValue::Int(v)) => Ok(Some(*v)),
            _ => Err(OpError::WrongAttributeType {
                op: self.op,
                attr: name.to_string(),
                expected: "int",
            }),
        }
    }

    pub(crate) fn opt_string(&mut self, name: &str) -> Result<Option<String>, OpError> {
        match self.take(name) {
            None => Ok(None),
            Some(AttributeValue::String(v)) => Ok(Some(v.clone())),
            _ => Err(OpError::WrongAttributeType {
                op: self.op,
                attr: name.to_string(),
                expected: "string",
            }),
        }
    }

    pub(crate) fn opt_tensor(&mut self, name: &str) -> Result<Option<Tensor>, OpError> {
        match self.take(name) {
            None => Ok(None),
            Some(AttributeValue::Tensor(v)) => Ok(Some((**v).clone())),
            _ => Err(OpError::WrongAttributeType {
                op: self.op,
                attr: name.to_string(),
                expected: "tensor",
            }),
        }
    }

    pub(crate) fn req_floats(&mut self, name: &'static str) -> Result<Vec<f32>, OpError> {
        self.opt_floats(name)?.ok_or(OpError::MissingAttribute {
            op: self.op,
            attr: name,
        })
    }

    pub(crate) fn req_int(&mut self, name: &'static str) -> Result<i64, OpError> {
        self.opt_int(name)?.ok_or(OpError::MissingAttribute {
            op: self.op,
            attr: name,
        })
    }

    pub(crate) fn finish(self) -> Result<(), OpError> {
        match self.entries.into_iter().next() {
            None => Ok(()),
            Some((name, _)) => Err(OpError::UnknownAttribute {
                op: self.op,
                attr: name.to_string(),
            }),
        }
    }
}

pub(crate) fn check_op<O: OnnxOp>(node: &Node) -> Result<(), OpError> {
    if node.domain != O::DOMAIN || node.op_type != O::OP_TYPE {
        return Err(OpError::WrongOp {
            expected_domain: O::DOMAIN,
            expected_op: O::OP_TYPE,
            got_domain: node.domain.clone(),
            got_op: node.op_type.clone(),
        });
    }
    Ok(())
}

pub(crate) fn check_counts(
    op: &'static str,
    node: &Node,
    inputs: fn(usize) -> bool,
    inputs_desc: &str,
    outputs: fn(usize) -> bool,
    outputs_desc: &str,
) -> Result<(), OpError> {
    if !inputs(node.inputs.len()) {
        return Err(OpError::WrongInputCount {
            op,
            expected: inputs_desc.to_string(),
            got: node.inputs.len(),
        });
    }
    if !outputs(node.outputs.len()) {
        return Err(OpError::WrongOutputCount {
            op,
            expected: outputs_desc.to_string(),
            got: node.outputs.len(),
        });
    }
    Ok(())
}

pub(crate) fn build_node<O: OnnxOp>(
    op_attrs: Vec<Attribute>,
    inputs: Vec<String>,
    outputs: Vec<String>,
) -> Emitted {
    Emitted::single(Node::new(O::OP_TYPE, O::DOMAIN, inputs, outputs, op_attrs))
}

pub(crate) fn push_floats(attrs: &mut Vec<Attribute>, name: &str, v: &Option<Vec<f32>>) {
    if let Some(x) = v {
        attrs.push(Attribute::floats(name, x.clone()));
    }
}

pub(crate) fn push_ints(attrs: &mut Vec<Attribute>, name: &str, v: &Option<Vec<i64>>) {
    if let Some(x) = v {
        attrs.push(Attribute::ints(name, x.clone()));
    }
}

pub(crate) fn push_strings(attrs: &mut Vec<Attribute>, name: &str, v: &Option<Vec<String>>) {
    if let Some(x) = v {
        attrs.push(Attribute::strings(name, x.clone()));
    }
}

pub(crate) fn push_float(attrs: &mut Vec<Attribute>, name: &str, v: Option<f32>) {
    if let Some(x) = v {
        attrs.push(Attribute::float(name, x));
    }
}

pub(crate) fn push_int(attrs: &mut Vec<Attribute>, name: &str, v: Option<i64>) {
    if let Some(x) = v {
        attrs.push(Attribute::int(name, x));
    }
}

pub(crate) fn push_string(attrs: &mut Vec<Attribute>, name: &str, v: &Option<String>) {
    if let Some(x) = v {
        attrs.push(Attribute::string(name, x.clone()));
    }
}

pub(crate) fn push_tensor(attrs: &mut Vec<Attribute>, name: &str, v: &Option<Tensor>) {
    if let Some(x) = v {
        attrs.push(Attribute::tensor(name, x.clone()));
    }
}
