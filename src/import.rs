use std::path::Path;

use crate::ir::{AttributeValue, Model, Node, OpsetId, Tensor, ValueInfo};
use crate::ml::{NodePayload, make_node_payload};
use crate::proto::{ProtoCodecError, decode_model};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportWarningKind {
    SubgraphAttribute,
    Function,
    TrainingInfo,
}

impl core::fmt::Display for ImportWarningKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SubgraphAttribute => write!(f, "subgraph-attribute"),
            Self::Function => write!(f, "function"),
            Self::TrainingInfo => write!(f, "training-info"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportWarning {
    pub node_index: Option<usize>,
    pub kind: ImportWarningKind,
    pub message: String,
}

impl ImportWarning {
    pub fn subgraph_attribute(node_index: usize, node: &Node, attr: &str) -> Self {
        ImportWarning {
            node_index: Some(node_index),
            kind: ImportWarningKind::SubgraphAttribute,
            message: format!(
                "node {node_index} ({}:{}) attribute {attr:?} carries a subgraph preserved as raw bytes",
                node.domain, node.op_type,
            ),
        }
    }

    pub fn function(domain: &str, name: &str) -> Self {
        ImportWarning {
            node_index: None,
            kind: ImportWarningKind::Function,
            message: format!(
                "model function {domain:?}::{name:?} preserved as raw bytes without resolution"
            ),
        }
    }

    pub fn training_info(index: usize) -> Self {
        ImportWarning {
            node_index: None,
            kind: ImportWarningKind::TrainingInfo,
            message: format!("model training_info entry {index} preserved as raw bytes"),
        }
    }
}

impl core::fmt::Display for ImportWarning {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.node_index {
            Some(i) => write!(f, "[{} at node {i}] {}", self.kind, self.message),
            None => write!(f, "[{}] {}", self.kind, self.message),
        }
    }
}

impl std::error::Error for ImportWarning {}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedModel {
    pub model: Model,
    pub payloads: Vec<NodePayload>,
    pub warnings: Vec<ImportWarning>,
}

impl DecodedModel {
    pub fn from_model(model: Model) -> Self {
        let payloads = model.graph.nodes.iter().map(make_node_payload).collect();
        let warnings = collect_warnings(&model);
        DecodedModel {
            model,
            payloads,
            warnings,
        }
    }

    pub fn decode_bytes(bytes: &[u8]) -> Result<Self, ProtoCodecError> {
        Ok(Self::from_model(decode_model(bytes)?))
    }

    pub fn decode_file(path: impl AsRef<Path>) -> Result<Self, ProtoCodecError> {
        let bytes = std::fs::read(path.as_ref())?;
        Self::decode_bytes(&bytes)
    }

    pub fn inputs(&self) -> &[ValueInfo] {
        &self.model.graph.inputs
    }

    pub fn outputs(&self) -> &[ValueInfo] {
        &self.model.graph.outputs
    }

    pub fn initializers(&self) -> &[Tensor] {
        &self.model.graph.initializers
    }

    pub fn graph_name(&self) -> &str {
        &self.model.graph.name
    }

    pub fn opset_import(&self) -> &[OpsetId] {
        &self.model.opset_import
    }

    pub fn typed(&self) -> impl Iterator<Item = (usize, &NodePayload)> {
        self.payloads
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_raw())
    }

    pub fn raw_nodes(&self) -> impl Iterator<Item = (usize, &Node)> {
        self.payloads
            .iter()
            .enumerate()
            .filter_map(|(i, p)| match p {
                NodePayload::Raw(node) => Some((i, node)),
                _ => None,
            })
    }

    pub fn visit(&self, visitor: &mut impl PayloadVisitor) {
        for (index, payload) in self.payloads.iter().enumerate() {
            match payload {
                NodePayload::Raw(node) => visitor.visit_raw(index, node),
                typed => visitor.visit_typed(index, typed),
            }
        }
    }
}

pub trait PayloadVisitor {
    fn visit_typed(&mut self, index: usize, payload: &NodePayload);
    fn visit_raw(&mut self, index: usize, node: &Node);
}

pub fn decode_bytes(bytes: &[u8]) -> Result<DecodedModel, ProtoCodecError> {
    DecodedModel::decode_bytes(bytes)
}

pub fn decode_file(path: impl AsRef<Path>) -> Result<DecodedModel, ProtoCodecError> {
    DecodedModel::decode_file(path)
}

fn collect_warnings(model: &Model) -> Vec<ImportWarning> {
    let mut warnings = Vec::new();
    for (index, node) in model.graph.nodes.iter().enumerate() {
        for attr in &node.attributes {
            match &attr.value {
                AttributeValue::Graph(_) | AttributeValue::Graphs(_) => {
                    warnings.push(ImportWarning::subgraph_attribute(index, node, &attr.name));
                }
                _ => {}
            }
        }
    }
    for f in &model.functions_raw {
        warnings.push(ImportWarning::function(&f.domain, &f.name));
    }
    for (index, _) in model.training_info_raw.iter().enumerate() {
        warnings.push(ImportWarning::training_info(index));
    }
    warnings
}
