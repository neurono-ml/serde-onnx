use super::{
    AttrTable, Emitted, ONNX_DOMAIN_STR, OnnxOp, OpError, build_node, check_counts, check_op,
    push_int,
};
use crate::ir::{Attribute, ElemType, Node};

#[derive(Debug, Clone, PartialEq)]
pub struct Cast {
    pub to: ElemType,
    pub saturate: Option<i64>,
}

impl OnnxOp for Cast {
    const OP_TYPE: &'static str = "Cast";
    const DOMAIN: &'static str = ONNX_DOMAIN_STR;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
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
        let mut attrs = vec![Attribute::int("to", self.to.disc() as i64)];
        push_int(&mut attrs, "saturate", self.saturate);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let to_raw = table.req_int("to")?;
        let to = ElemType::from_disc(to_raw as i32).ok_or_else(|| OpError::InvalidValue {
            op: Self::OP_TYPE,
            attr: "to".to_string(),
            detail: "unknown element type discriminant".to_string(),
        })?;
        let saturate = table.opt_int("saturate")?;
        table.finish()?;
        Ok(Cast { to, saturate })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reshape {
    pub allowzero: Option<i64>,
}

impl OnnxOp for Reshape {
    const OP_TYPE: &'static str = "Reshape";
    const DOMAIN: &'static str = ONNX_DOMAIN_STR;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
        if inputs.is_empty() || inputs.len() > 2 {
            return Err(OpError::WrongInputCount {
                op: Self::OP_TYPE,
                expected: "1..=2".to_string(),
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
        push_int(&mut attrs, "allowzero", self.allowzero);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(
            Self::OP_TYPE,
            node,
            |n| n == 1 || n == 2,
            "1..=2",
            |n| n == 1,
            "1",
        )?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let allowzero = table.opt_int("allowzero")?;
        table.finish()?;
        Ok(Reshape { allowzero })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Concat {
    pub axis: i64,
}

impl OnnxOp for Concat {
    const OP_TYPE: &'static str = "Concat";
    const DOMAIN: &'static str = ONNX_DOMAIN_STR;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
        if inputs.is_empty() {
            return Err(OpError::WrongInputCount {
                op: Self::OP_TYPE,
                expected: "1..".to_string(),
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
        Ok(build_node::<Self>(
            vec![Attribute::int("axis", self.axis)],
            inputs,
            outputs,
        ))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n >= 1, "1..", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let axis = table.req_int("axis")?;
        table.finish()?;
        Ok(Concat { axis })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gather {
    pub axis: Option<i64>,
}

impl OnnxOp for Gather {
    const OP_TYPE: &'static str = "Gather";
    const DOMAIN: &'static str = ONNX_DOMAIN_STR;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
        if inputs.len() != 2 {
            return Err(OpError::WrongInputCount {
                op: Self::OP_TYPE,
                expected: "2".to_string(),
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
        push_int(&mut attrs, "axis", self.axis);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 2, "2", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let axis = table.opt_int("axis")?;
        table.finish()?;
        Ok(Gather { axis })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Identity;

impl OnnxOp for Identity {
    const OP_TYPE: &'static str = "Identity";
    const DOMAIN: &'static str = ONNX_DOMAIN_STR;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
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
        Ok(build_node::<Self>(Vec::new(), inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 1, "1")?;
        AttrTable::new(Self::OP_TYPE, node)?.finish()?;
        Ok(Identity)
    }
}
