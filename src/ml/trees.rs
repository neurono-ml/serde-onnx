use super::{
    AttrTable, Emitted, OnnxOp, OpError, build_node, check_counts, check_op, push_floats, push_int,
    push_ints, push_string, push_strings, push_tensor,
};
use crate::ir::{Attribute, ML_DOMAIN, Node, Tensor};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TreeNodes {
    pub falsenodeids: Vec<i64>,
    pub featureids: Vec<i64>,
    pub hitrates: Option<Vec<f32>>,
    pub hitrates_as_tensor: Option<Tensor>,
    pub missing_value_tracks_true: Option<Vec<i64>>,
    pub modes: Vec<String>,
    pub nodeids: Vec<i64>,
    pub treeids: Vec<i64>,
    pub truenodeids: Vec<i64>,
    pub values: Option<Vec<f32>>,
    pub values_as_tensor: Option<Tensor>,
}

impl TreeNodes {
    fn check(&self, op: &'static str) -> Result<(), OpError> {
        let n = self.modes.len();
        if n == 0 {
            return Err(OpError::MissingAttribute {
                op,
                attr: "nodes_modes",
            });
        }
        for (attr, len) in [
            ("nodes_falsenodeids", self.falsenodeids.len()),
            ("nodes_featureids", self.featureids.len()),
            ("nodes_nodeids", self.nodeids.len()),
            ("nodes_treeids", self.treeids.len()),
            ("nodes_truenodeids", self.truenodeids.len()),
        ] {
            if len != n {
                return Err(OpError::InvalidValue {
                    op,
                    attr: attr.to_string(),
                    detail: "all nodes_* arrays must share the length of nodes_modes".to_string(),
                });
            }
        }
        if let Some(v) = &self.missing_value_tracks_true
            && v.len() != n
        {
            return Err(OpError::InvalidValue {
                op,
                attr: "nodes_missing_value_tracks_true".to_string(),
                detail: "all nodes_* arrays must share the length of nodes_modes".to_string(),
            });
        }
        if let Some(v) = &self.hitrates
            && v.len() != n
        {
            return Err(OpError::InvalidValue {
                op,
                attr: "nodes_hitrates".to_string(),
                detail: "all nodes_* arrays must share the length of nodes_modes".to_string(),
            });
        }
        if let Some(v) = &self.values
            && v.len() != n
        {
            return Err(OpError::InvalidValue {
                op,
                attr: "nodes_values".to_string(),
                detail: "all nodes_* arrays must share the length of nodes_modes".to_string(),
            });
        }
        Ok(())
    }

    fn emit(&self, attrs: &mut Vec<Attribute>) {
        push_ints(
            attrs,
            "nodes_falsenodeids",
            &Some(self.falsenodeids.clone()),
        );
        push_ints(attrs, "nodes_featureids", &Some(self.featureids.clone()));
        push_floats(attrs, "nodes_hitrates", &self.hitrates);
        push_tensor(attrs, "nodes_hitrates_as_tensor", &self.hitrates_as_tensor);
        push_ints(
            attrs,
            "nodes_missing_value_tracks_true",
            &self.missing_value_tracks_true,
        );
        push_strings(attrs, "nodes_modes", &Some(self.modes.clone()));
        push_ints(attrs, "nodes_nodeids", &Some(self.nodeids.clone()));
        push_ints(attrs, "nodes_treeids", &Some(self.treeids.clone()));
        push_ints(attrs, "nodes_truenodeids", &Some(self.truenodeids.clone()));
        push_floats(attrs, "nodes_values", &self.values);
        push_tensor(attrs, "nodes_values_as_tensor", &self.values_as_tensor);
    }

    fn parse(op: &'static str, table: &mut AttrTable) -> Result<Self, OpError> {
        let nodes = TreeNodes {
            falsenodeids: table.opt_ints("nodes_falsenodeids")?.unwrap_or_default(),
            featureids: table.opt_ints("nodes_featureids")?.unwrap_or_default(),
            hitrates: table.opt_floats("nodes_hitrates")?,
            hitrates_as_tensor: table.opt_tensor("nodes_hitrates_as_tensor")?,
            missing_value_tracks_true: table.opt_ints("nodes_missing_value_tracks_true")?,
            modes: table.opt_strings("nodes_modes")?.unwrap_or_default(),
            nodeids: table.opt_ints("nodes_nodeids")?.unwrap_or_default(),
            treeids: table.opt_ints("nodes_treeids")?.unwrap_or_default(),
            truenodeids: table.opt_ints("nodes_truenodeids")?.unwrap_or_default(),
            values: table.opt_floats("nodes_values")?,
            values_as_tensor: table.opt_tensor("nodes_values_as_tensor")?,
        };
        nodes.check(op)?;
        Ok(nodes)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TreeEnsembleClassifier {
    pub base_values: Option<Vec<f32>>,
    pub base_values_as_tensor: Option<Tensor>,
    pub class_ids: Vec<i64>,
    pub class_nodeids: Vec<i64>,
    pub class_treeids: Vec<i64>,
    pub class_weights: Option<Vec<f32>>,
    pub class_weights_as_tensor: Option<Tensor>,
    pub classlabels_int64s: Option<Vec<i64>>,
    pub classlabels_strings: Option<Vec<String>>,
    pub nodes: TreeNodes,
    pub post_transform: Option<String>,
}

impl TreeEnsembleClassifier {
    fn check(&self) -> Result<(), OpError> {
        const OP: &str = "TreeEnsembleClassifier";
        if self.classlabels_int64s.is_some() == self.classlabels_strings.is_some() {
            return Err(OpError::InvalidValue {
                op: OP,
                attr: "classlabels_*".to_string(),
                detail: "exactly one of classlabels_int64s and classlabels_strings must be set"
                    .to_string(),
            });
        }
        self.nodes.check(OP)?;
        let n = self.class_ids.len();
        if n == 0 {
            return Err(OpError::MissingAttribute {
                op: OP,
                attr: "class_ids",
            });
        }
        for (attr, len) in [
            ("class_nodeids", self.class_nodeids.len()),
            ("class_treeids", self.class_treeids.len()),
        ] {
            if len != n {
                return Err(OpError::InvalidValue {
                    op: OP,
                    attr: attr.to_string(),
                    detail: "all class_* arrays must share the length of class_ids".to_string(),
                });
            }
        }
        if let Some(v) = &self.class_weights
            && v.len() != n
        {
            return Err(OpError::InvalidValue {
                op: OP,
                attr: "class_weights".to_string(),
                detail: "all class_* arrays must share the length of class_ids".to_string(),
            });
        }
        Ok(())
    }
}

impl OnnxOp for TreeEnsembleClassifier {
    const OP_TYPE: &'static str = "TreeEnsembleClassifier";
    const DOMAIN: &'static str = ML_DOMAIN;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
        self.check()?;
        if inputs.len() != 1 {
            return Err(OpError::WrongInputCount {
                op: Self::OP_TYPE,
                expected: "1".to_string(),
                got: inputs.len(),
            });
        }
        if outputs.len() != 2 {
            return Err(OpError::WrongOutputCount {
                op: Self::OP_TYPE,
                expected: "2".to_string(),
                got: outputs.len(),
            });
        }
        let mut attrs = Vec::new();
        push_floats(&mut attrs, "base_values", &self.base_values);
        push_tensor(
            &mut attrs,
            "base_values_as_tensor",
            &self.base_values_as_tensor,
        );
        push_ints(&mut attrs, "class_ids", &Some(self.class_ids.clone()));
        push_ints(
            &mut attrs,
            "class_nodeids",
            &Some(self.class_nodeids.clone()),
        );
        push_ints(
            &mut attrs,
            "class_treeids",
            &Some(self.class_treeids.clone()),
        );
        push_floats(&mut attrs, "class_weights", &self.class_weights);
        push_tensor(
            &mut attrs,
            "class_weights_as_tensor",
            &self.class_weights_as_tensor,
        );
        push_ints(&mut attrs, "classlabels_int64s", &self.classlabels_int64s);
        push_strings(&mut attrs, "classlabels_strings", &self.classlabels_strings);
        self.nodes.emit(&mut attrs);
        push_string(&mut attrs, "post_transform", &self.post_transform);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 2, "2")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let ensemble = TreeEnsembleClassifier {
            base_values: table.opt_floats("base_values")?,
            base_values_as_tensor: table.opt_tensor("base_values_as_tensor")?,
            class_ids: table.opt_ints("class_ids")?.unwrap_or_default(),
            class_nodeids: table.opt_ints("class_nodeids")?.unwrap_or_default(),
            class_treeids: table.opt_ints("class_treeids")?.unwrap_or_default(),
            class_weights: table.opt_floats("class_weights")?,
            class_weights_as_tensor: table.opt_tensor("class_weights_as_tensor")?,
            classlabels_int64s: table.opt_ints("classlabels_int64s")?,
            classlabels_strings: table.opt_strings("classlabels_strings")?,
            nodes: TreeNodes::parse(Self::OP_TYPE, &mut table)?,
            post_transform: table.opt_string("post_transform")?,
        };
        table.finish()?;
        ensemble.check()?;
        Ok(ensemble)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TreeEnsembleRegressor {
    pub aggregate_function: Option<String>,
    pub base_values: Option<Vec<f32>>,
    pub base_values_as_tensor: Option<Tensor>,
    pub n_targets: Option<i64>,
    pub nodes: TreeNodes,
    pub post_transform: Option<String>,
    pub target_ids: Vec<i64>,
    pub target_nodeids: Vec<i64>,
    pub target_treeids: Vec<i64>,
    pub target_weights: Option<Vec<f32>>,
    pub target_weights_as_tensor: Option<Tensor>,
}

impl TreeEnsembleRegressor {
    fn check(&self) -> Result<(), OpError> {
        const OP: &str = "TreeEnsembleRegressor";
        self.nodes.check(OP)?;
        let n = self.target_ids.len();
        if n == 0 {
            return Err(OpError::MissingAttribute {
                op: OP,
                attr: "target_ids",
            });
        }
        for (attr, len) in [
            ("target_nodeids", self.target_nodeids.len()),
            ("target_treeids", self.target_treeids.len()),
        ] {
            if len != n {
                return Err(OpError::InvalidValue {
                    op: OP,
                    attr: attr.to_string(),
                    detail: "all target_* arrays must share the length of target_ids".to_string(),
                });
            }
        }
        if let Some(v) = &self.target_weights
            && v.len() != n
        {
            return Err(OpError::InvalidValue {
                op: OP,
                attr: "target_weights".to_string(),
                detail: "all target_* arrays must share the length of target_ids".to_string(),
            });
        }
        Ok(())
    }
}

impl OnnxOp for TreeEnsembleRegressor {
    const OP_TYPE: &'static str = "TreeEnsembleRegressor";
    const DOMAIN: &'static str = ML_DOMAIN;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
        self.check()?;
        if inputs.len() != 1 {
            return Err(OpError::WrongInputCount {
                op: Self::OP_TYPE,
                expected: "1".to_string(),
                got: inputs.len(),
            });
        }
        if outputs.len() != 1 {
            return Err(OpError::WrongOutputCount {
                op: Self::OP_TYPE,
                expected: "1".to_string(),
                got: outputs.len(),
            });
        }
        let mut attrs = Vec::new();
        push_string(&mut attrs, "aggregate_function", &self.aggregate_function);
        push_floats(&mut attrs, "base_values", &self.base_values);
        push_tensor(
            &mut attrs,
            "base_values_as_tensor",
            &self.base_values_as_tensor,
        );
        push_int(&mut attrs, "n_targets", self.n_targets);
        self.nodes.emit(&mut attrs);
        push_string(&mut attrs, "post_transform", &self.post_transform);
        push_ints(&mut attrs, "target_ids", &Some(self.target_ids.clone()));
        push_ints(
            &mut attrs,
            "target_nodeids",
            &Some(self.target_nodeids.clone()),
        );
        push_ints(
            &mut attrs,
            "target_treeids",
            &Some(self.target_treeids.clone()),
        );
        push_floats(&mut attrs, "target_weights", &self.target_weights);
        push_tensor(
            &mut attrs,
            "target_weights_as_tensor",
            &self.target_weights_as_tensor,
        );
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let ensemble = TreeEnsembleRegressor {
            aggregate_function: table.opt_string("aggregate_function")?,
            base_values: table.opt_floats("base_values")?,
            base_values_as_tensor: table.opt_tensor("base_values_as_tensor")?,
            n_targets: table.opt_int("n_targets")?,
            nodes: TreeNodes::parse(Self::OP_TYPE, &mut table)?,
            post_transform: table.opt_string("post_transform")?,
            target_ids: table.opt_ints("target_ids")?.unwrap_or_default(),
            target_nodeids: table.opt_ints("target_nodeids")?.unwrap_or_default(),
            target_treeids: table.opt_ints("target_treeids")?.unwrap_or_default(),
            target_weights: table.opt_floats("target_weights")?,
            target_weights_as_tensor: table.opt_tensor("target_weights_as_tensor")?,
        };
        table.finish()?;
        ensemble.check()?;
        Ok(ensemble)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ZipMap {
    pub classlabels_int64s: Option<Vec<i64>>,
    pub classlabels_strings: Option<Vec<String>>,
}

impl OnnxOp for ZipMap {
    const OP_TYPE: &'static str = "ZipMap";
    const DOMAIN: &'static str = ML_DOMAIN;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
        if self.classlabels_int64s.is_some() == self.classlabels_strings.is_some() {
            return Err(OpError::InvalidValue {
                op: Self::OP_TYPE,
                attr: "classlabels_*".to_string(),
                detail: "exactly one of classlabels_int64s and classlabels_strings must be set"
                    .to_string(),
            });
        }
        if inputs.len() != 1 {
            return Err(OpError::WrongInputCount {
                op: Self::OP_TYPE,
                expected: "1".to_string(),
                got: inputs.len(),
            });
        }
        if outputs.len() != 1 {
            return Err(OpError::WrongOutputCount {
                op: Self::OP_TYPE,
                expected: "1".to_string(),
                got: outputs.len(),
            });
        }
        let mut attrs = Vec::new();
        push_ints(&mut attrs, "classlabels_int64s", &self.classlabels_int64s);
        push_strings(&mut attrs, "classlabels_strings", &self.classlabels_strings);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let map = ZipMap {
            classlabels_int64s: table.opt_ints("classlabels_int64s")?,
            classlabels_strings: table.opt_strings("classlabels_strings")?,
        };
        table.finish()?;
        if map.classlabels_int64s.is_some() == map.classlabels_strings.is_some() {
            return Err(OpError::InvalidValue {
                op: Self::OP_TYPE,
                attr: "classlabels_*".to_string(),
                detail: "exactly one of classlabels_int64s and classlabels_strings must be set"
                    .to_string(),
            });
        }
        Ok(map)
    }
}
