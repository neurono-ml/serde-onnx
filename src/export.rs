use std::collections::BTreeMap;
use std::collections::HashSet;

use crate::ir::Attribute;
use crate::ir::Dim;
use crate::ir::Graph;
use crate::ir::Model;
use crate::ir::OpsetId;
use crate::ir::Scalar;
use crate::ir::Shape;
use crate::ir::Tensor;
use crate::ir::ValidationError;
use crate::ir::ValueInfo;
use crate::ir::ValueType;
use crate::ml::CORE_TYPED_OPSET_TARGET;
use crate::ml::ML_DOMAIN_STR;
use crate::ml::ML_EXPORT_OPSET_TARGET;
use crate::ml::ONNX_DOMAIN_STR;
use crate::ml::OnnxOp;
use crate::ml::OpError;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValueRef(pub String);

impl ValueRef {
    pub fn name(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl From<String> for ValueRef {
    fn from(v: String) -> Self {
        ValueRef(v)
    }
}

impl From<&str> for ValueRef {
    fn from(v: &str) -> Self {
        ValueRef(v.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExportError {
    DuplicateName(String),
    LengthMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    Op(OpError),
    Validation(Vec<ValidationError>),
}

impl core::fmt::Display for ExportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DuplicateName(name) => write!(f, "exported value {name:?} is already defined"),
            Self::LengthMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "initializer {name:?}: shape implies {expected} elements but data carries {got}"
            ),
            Self::Op(e) => write!(f, "invalid op for export: {e}"),
            Self::Validation(errors) => {
                write!(
                    f,
                    "exported graph failed validation ({} error(s)):",
                    errors.len()
                )?;
                for e in errors {
                    write!(f, "\n- {e}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ExportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Op(e) => Some(e),
            Self::Validation(_) => None,
            _ => None,
        }
    }
}

impl From<OpError> for ExportError {
    fn from(e: OpError) -> Self {
        ExportError::Op(e)
    }
}

pub struct GraphBuilder {
    graph: Graph,
    defined: HashSet<String>,
    opsets: BTreeMap<String, i64>,
}

impl GraphBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        GraphBuilder {
            graph: Graph::new(name),
            defined: HashSet::new(),
            opsets: BTreeMap::new(),
        }
    }

    pub fn graph_name(&self) -> &str {
        &self.graph.name
    }

    pub fn declare_opset(&mut self, domain: impl Into<String>, version: i64) {
        let domain = domain.into();
        self.opsets
            .entry(domain)
            .and_modify(|v| {
                if version > *v {
                    *v = version;
                }
            })
            .or_insert(version);
    }

    fn track_opset_for_domain(&mut self, domain: &str) {
        if domain == ONNX_DOMAIN_STR {
            self.declare_opset(domain, CORE_TYPED_OPSET_TARGET);
        } else if domain == ML_DOMAIN_STR {
            self.declare_opset(domain, ML_EXPORT_OPSET_TARGET);
        }
    }

    fn claim(&mut self, name: &str) -> Result<(), ExportError> {
        if name.is_empty() {
            return Ok(());
        }
        if !self.defined.insert(name.to_string()) {
            return Err(ExportError::DuplicateName(name.to_string()));
        }
        Ok(())
    }

    pub fn input(
        &mut self,
        name: impl Into<String>,
        value_type: ValueType,
    ) -> Result<ValueRef, ExportError> {
        let name = name.into();
        self.claim(&name)?;
        self.graph
            .inputs
            .push(ValueInfo::new(name.clone(), value_type));
        Ok(ValueRef(name))
    }

    pub fn output(
        &mut self,
        name: impl Into<String>,
        value_type: ValueType,
    ) -> Result<ValueRef, ExportError> {
        let name = name.into();
        self.graph
            .outputs
            .push(ValueInfo::new(name.clone(), value_type));
        Ok(ValueRef(name))
    }

    pub fn initializer<T: Scalar>(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        values: Vec<T>,
    ) -> Result<ValueRef, ExportError> {
        let name = name.into();
        if let Some(expected) = fixed_len(&shape)
            && expected != values.len()
        {
            return Err(ExportError::LengthMismatch {
                name,
                expected,
                got: values.len(),
            });
        }
        self.claim(&name)?;
        let tensor = Tensor::new(name.clone(), T::ELEM_TYPE, shape, T::into_data(values));
        self.graph.initializers.push(tensor);
        Ok(ValueRef(name))
    }

    pub fn initializer_tensor(&mut self, tensor: Tensor) -> Result<ValueRef, ExportError> {
        let name = tensor.name.clone();
        self.claim(&name)?;
        self.graph.initializers.push(tensor);
        Ok(ValueRef(name))
    }

    pub fn push_node(
        &mut self,
        op_type: &str,
        domain: &str,
        inputs: Vec<String>,
        attributes: Vec<Attribute>,
        num_outputs: usize,
    ) -> Result<Vec<ValueRef>, ExportError> {
        let index = self.graph.nodes.len();
        let stem: String = op_type
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect();
        let stem = if stem.is_empty() {
            "Op".to_string()
        } else {
            stem
        };
        let mut outputs = Vec::with_capacity(num_outputs);
        for k in 0..num_outputs {
            let candidate = if num_outputs == 1 {
                format!("node_{index}_{stem}_out")
            } else {
                format!("node_{index}_{stem}_out_{k}")
            };
            self.claim(&candidate)?;
            outputs.push(candidate);
        }
        let node = crate::ir::Node::new(op_type, domain, inputs, outputs.clone(), attributes);
        self.track_opset_for_domain(domain);
        self.graph.nodes.push(node);
        Ok(outputs.into_iter().map(ValueRef).collect())
    }

    pub fn emit_op<O: OnnxOp>(
        &mut self,
        op: &O,
        inputs: Vec<ValueRef>,
        num_outputs: usize,
    ) -> Result<Vec<ValueRef>, ExportError> {
        let index = self.graph.nodes.len();
        let stem: String = O::OP_TYPE
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect();
        let stem = if stem.is_empty() {
            "Op".to_string()
        } else {
            stem
        };
        let mut outputs = Vec::with_capacity(num_outputs);
        for k in 0..num_outputs {
            let candidate = if num_outputs == 1 {
                format!("node_{index}_{stem}_out")
            } else {
                format!("node_{index}_{stem}_out_{k}")
            };
            outputs.push(candidate);
        }
        let emitted = op.to_node(inputs.into_iter().map(|v| v.0).collect(), outputs.clone())?;
        self.push_emitted(emitted)?;
        Ok(outputs.into_iter().map(ValueRef).collect())
    }

    fn push_emitted(&mut self, emitted: crate::ml::Emitted) -> Result<(), ExportError> {
        for tensor in emitted.initializers {
            let name = tensor.name.clone();
            self.claim(&name)?;
            self.graph.initializers.push(tensor);
        }
        for out in &emitted.node.outputs {
            self.claim(out)?;
        }
        self.track_opset_for_domain(&emitted.node.domain);
        self.graph.nodes.push(emitted.node);
        Ok(())
    }

    pub fn finish(self) -> (Graph, Vec<OpsetId>) {
        let opsets = self
            .opsets
            .into_iter()
            .map(|(domain, version)| OpsetId { domain, version })
            .collect();
        (self.graph, opsets)
    }

    pub fn opset_import(&self) -> Vec<OpsetId> {
        self.opsets
            .iter()
            .map(|(domain, version)| OpsetId {
                domain: domain.clone(),
                version: *version,
            })
            .collect()
    }
}

impl crate::ml::OpSink for GraphBuilder {
    type Error = ExportError;

    fn emit(
        &mut self,
        emitted: crate::ml::Emitted,
    ) -> Result<(Vec<String>, Vec<String>), Self::Error> {
        let inputs = emitted.node.inputs.clone();
        let outputs = emitted.node.outputs.clone();
        self.push_emitted(emitted)?;
        Ok((inputs, outputs))
    }
}

fn fixed_len(shape: &[Dim]) -> Option<usize> {
    let mut acc = 1usize;
    for d in shape {
        match d {
            Dim::Fixed(v) => {
                if *v < 0 {
                    return None;
                }
                acc = acc.checked_mul(*v as usize)?;
            }
            _ => return None,
        }
    }
    Some(acc)
}

pub trait ToOnnx {
    fn to_graph(&self, builder: &mut GraphBuilder) -> Result<ValueRef, ExportError>;
}

pub fn export_model<T: ToOnnx + ?Sized>(
    value: &T,
    graph_name: impl Into<String>,
) -> Result<Model, ExportError> {
    let mut builder = GraphBuilder::new(graph_name);
    value.to_graph(&mut builder)?;
    let (graph, opsets) = builder.finish();
    let model = Model::new(graph, opsets);
    let report = crate::ir::validate_model(&model);
    if !report.is_ok() {
        return Err(ExportError::Validation(report.errors));
    }
    Ok(model)
}
