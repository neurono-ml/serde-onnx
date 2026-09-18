use super::{
    Cast, Concat, DictVectorizer, FeatureVectorizer, Gather, Identity, Imputer, LabelEncoder,
    LinearClassifier, LinearRegressor, Normalizer, OneHotEncoder, OnnxOp, Reshape, Scaler,
    SvmClassifier, SvmRegressor, TreeEnsembleClassifier, TreeEnsembleRegressor, ZipMap,
};
use crate::ir::Node;

#[derive(Debug, Clone, PartialEq)]
pub struct TypedNode<O> {
    pub op: O,
    pub node: Node,
}

impl<O: OnnxOp> TypedNode<O> {
    pub fn recognize(node: &Node) -> Result<Self, super::OpError> {
        Ok(TypedNode {
            op: O::from_node(node)?,
            node: node.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodePayload {
    Scaler(TypedNode<Scaler>),
    Imputer(TypedNode<Imputer>),
    Normalizer(TypedNode<Normalizer>),
    LabelEncoder(TypedNode<LabelEncoder>),
    OneHotEncoder(TypedNode<OneHotEncoder>),
    DictVectorizer(TypedNode<DictVectorizer>),
    FeatureVectorizer(TypedNode<FeatureVectorizer>),
    LinearClassifier(TypedNode<LinearClassifier>),
    LinearRegressor(TypedNode<LinearRegressor>),
    SvmClassifier(TypedNode<SvmClassifier>),
    SvmRegressor(TypedNode<SvmRegressor>),
    TreeEnsembleClassifier(TypedNode<TreeEnsembleClassifier>),
    TreeEnsembleRegressor(TypedNode<TreeEnsembleRegressor>),
    ZipMap(TypedNode<ZipMap>),
    Cast(TypedNode<Cast>),
    Reshape(TypedNode<Reshape>),
    Concat(TypedNode<Concat>),
    Gather(TypedNode<Gather>),
    Identity(TypedNode<Identity>),
    Raw(Node),
}

impl NodePayload {
    pub fn node(&self) -> &Node {
        match self {
            Self::Scaler(v) => &v.node,
            Self::Imputer(v) => &v.node,
            Self::Normalizer(v) => &v.node,
            Self::LabelEncoder(v) => &v.node,
            Self::OneHotEncoder(v) => &v.node,
            Self::DictVectorizer(v) => &v.node,
            Self::FeatureVectorizer(v) => &v.node,
            Self::LinearClassifier(v) => &v.node,
            Self::LinearRegressor(v) => &v.node,
            Self::SvmClassifier(v) => &v.node,
            Self::SvmRegressor(v) => &v.node,
            Self::TreeEnsembleClassifier(v) => &v.node,
            Self::TreeEnsembleRegressor(v) => &v.node,
            Self::ZipMap(v) => &v.node,
            Self::Cast(v) => &v.node,
            Self::Reshape(v) => &v.node,
            Self::Concat(v) => &v.node,
            Self::Gather(v) => &v.node,
            Self::Identity(v) => &v.node,
            Self::Raw(node) => node,
        }
    }

    pub fn is_raw(&self) -> bool {
        matches!(self, Self::Raw(_))
    }
}

pub fn make_node_payload(node: &Node) -> NodePayload {
    const ML: &str = crate::ir::ML_DOMAIN;
    const CORE: &str = super::ONNX_DOMAIN_STR;
    match (node.domain.as_str(), node.op_type.as_str()) {
        (ML, "Scaler") => TypedNode::recognize(node).map(NodePayload::Scaler),
        (ML, "Imputer") => TypedNode::recognize(node).map(NodePayload::Imputer),
        (ML, "Normalizer") => TypedNode::recognize(node).map(NodePayload::Normalizer),
        (ML, "LabelEncoder") => TypedNode::recognize(node).map(NodePayload::LabelEncoder),
        (ML, "OneHotEncoder") => TypedNode::recognize(node).map(NodePayload::OneHotEncoder),
        (ML, "DictVectorizer") => TypedNode::recognize(node).map(NodePayload::DictVectorizer),
        (ML, "FeatureVectorizer") => TypedNode::recognize(node).map(NodePayload::FeatureVectorizer),
        (ML, "LinearClassifier") => TypedNode::recognize(node).map(NodePayload::LinearClassifier),
        (ML, "LinearRegressor") => TypedNode::recognize(node).map(NodePayload::LinearRegressor),
        (ML, "SVMClassifier") => TypedNode::recognize(node).map(NodePayload::SvmClassifier),
        (ML, "SVMRegressor") => TypedNode::recognize(node).map(NodePayload::SvmRegressor),
        (ML, "TreeEnsembleClassifier") => {
            TypedNode::recognize(node).map(NodePayload::TreeEnsembleClassifier)
        }
        (ML, "TreeEnsembleRegressor") => {
            TypedNode::recognize(node).map(NodePayload::TreeEnsembleRegressor)
        }
        (ML, "ZipMap") => TypedNode::recognize(node).map(NodePayload::ZipMap),
        (CORE, "Cast") => TypedNode::recognize(node).map(NodePayload::Cast),
        (CORE, "Reshape") => TypedNode::recognize(node).map(NodePayload::Reshape),
        (CORE, "Concat") => TypedNode::recognize(node).map(NodePayload::Concat),
        (CORE, "Gather") => TypedNode::recognize(node).map(NodePayload::Gather),
        (CORE, "Identity") => TypedNode::recognize(node).map(NodePayload::Identity),
        _ => Err(super::OpError::WrongOp {
            expected_domain: "known",
            expected_op: "known",
            got_domain: node.domain.clone(),
            got_op: node.op_type.clone(),
        }),
    }
    .unwrap_or_else(|_| NodePayload::Raw(node.clone()))
}
