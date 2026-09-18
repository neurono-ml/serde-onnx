use super::{
    AttrTable, Emitted, OnnxOp, OpError, build_node, check_counts, check_op, push_float,
    push_floats, push_int, push_ints, push_string, push_strings, push_tensor,
};
use crate::ir::{Attribute, ML_DOMAIN, Node, Tensor};

#[derive(Debug, Clone, PartialEq)]
pub struct Scaler {
    pub offset: Option<Vec<f32>>,
    pub scale: Option<Vec<f32>>,
}

impl OnnxOp for Scaler {
    const OP_TYPE: &'static str = "Scaler";
    const DOMAIN: &'static str = ML_DOMAIN;

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
        let mut attrs = Vec::new();
        push_floats(&mut attrs, "offset", &self.offset);
        push_floats(&mut attrs, "scale", &self.scale);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let offset = table.opt_floats("offset")?;
        let scale = table.opt_floats("scale")?;
        table.finish()?;
        Ok(Scaler { offset, scale })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Imputer {
    pub imputed_value_floats: Option<Vec<f32>>,
    pub imputed_value_int64s: Option<Vec<i64>>,
    pub replaced_value_float: Option<f32>,
    pub replaced_value_int64: Option<i64>,
}

impl OnnxOp for Imputer {
    const OP_TYPE: &'static str = "Imputer";
    const DOMAIN: &'static str = ML_DOMAIN;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
        if self.imputed_value_floats.is_some() && self.imputed_value_int64s.is_some() {
            return Err(OpError::InvalidValue {
                op: Self::OP_TYPE,
                attr: "imputed_value_*".to_string(),
                detail: "only one of imputed_value_floats and imputed_value_int64s may be set"
                    .to_string(),
            });
        }
        if self.replaced_value_float.is_some() && self.replaced_value_int64.is_some() {
            return Err(OpError::InvalidValue {
                op: Self::OP_TYPE,
                attr: "replaced_value_*".to_string(),
                detail: "only one of replaced_value_float and replaced_value_int64 may be set"
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
        push_floats(
            &mut attrs,
            "imputed_value_floats",
            &self.imputed_value_floats,
        );
        push_ints(
            &mut attrs,
            "imputed_value_int64s",
            &self.imputed_value_int64s,
        );
        push_float(
            &mut attrs,
            "replaced_value_float",
            self.replaced_value_float,
        );
        push_int(
            &mut attrs,
            "replaced_value_int64",
            self.replaced_value_int64,
        );
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let imputed_value_floats = table.opt_floats("imputed_value_floats")?;
        let imputed_value_int64s = table.opt_ints("imputed_value_int64s")?;
        let replaced_value_float = table.opt_float("replaced_value_float")?;
        let replaced_value_int64 = table.opt_int("replaced_value_int64")?;
        table.finish()?;
        if imputed_value_floats.is_some() && imputed_value_int64s.is_some() {
            return Err(OpError::InvalidValue {
                op: Self::OP_TYPE,
                attr: "imputed_value_*".to_string(),
                detail: "only one of imputed_value_floats and imputed_value_int64s may be set"
                    .to_string(),
            });
        }
        if replaced_value_float.is_some() && replaced_value_int64.is_some() {
            return Err(OpError::InvalidValue {
                op: Self::OP_TYPE,
                attr: "replaced_value_*".to_string(),
                detail: "only one of replaced_value_float and replaced_value_int64 may be set"
                    .to_string(),
            });
        }
        Ok(Imputer {
            imputed_value_floats,
            imputed_value_int64s,
            replaced_value_float,
            replaced_value_int64,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Normalizer {
    pub norm: Option<String>,
}

impl OnnxOp for Normalizer {
    const OP_TYPE: &'static str = "Normalizer";
    const DOMAIN: &'static str = ML_DOMAIN;

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
        let mut attrs = Vec::new();
        push_string(&mut attrs, "norm", &self.norm);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let norm = table.opt_string("norm")?;
        table.finish()?;
        Ok(Normalizer { norm })
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LabelEncoder {
    pub default_float: Option<f32>,
    pub default_int64: Option<i64>,
    pub default_string: Option<String>,
    pub default_tensor: Option<Tensor>,
    pub keys_floats: Option<Vec<f32>>,
    pub keys_int64s: Option<Vec<i64>>,
    pub keys_strings: Option<Vec<String>>,
    pub keys_tensor: Option<Tensor>,
    pub values_floats: Option<Vec<f32>>,
    pub values_int64s: Option<Vec<i64>>,
    pub values_strings: Option<Vec<String>>,
    pub values_tensor: Option<Tensor>,
}

impl LabelEncoder {
    fn check_keys_values(&self) -> Result<(), OpError> {
        let keys = [
            self.keys_floats.is_some(),
            self.keys_int64s.is_some(),
            self.keys_strings.is_some(),
            self.keys_tensor.is_some(),
        ]
        .iter()
        .filter(|b| **b)
        .count();
        if keys != 1 {
            return Err(OpError::InvalidValue {
                op: Self::OP_TYPE,
                attr: "keys_*".to_string(),
                detail:
                    "exactly one of keys_floats, keys_int64s, keys_strings, keys_tensor must be set"
                        .to_string(),
            });
        }
        let values = [
            self.values_floats.is_some(),
            self.values_int64s.is_some(),
            self.values_strings.is_some(),
            self.values_tensor.is_some(),
        ]
        .iter()
        .filter(|b| **b)
        .count();
        if values != 1 {
            return Err(OpError::InvalidValue {
                op: Self::OP_TYPE,
                attr: "values_*".to_string(),
                detail: "exactly one of values_floats, values_int64s, values_strings, values_tensor must be set"
                    .to_string(),
            });
        }
        Ok(())
    }
}

impl OnnxOp for LabelEncoder {
    const OP_TYPE: &'static str = "LabelEncoder";
    const DOMAIN: &'static str = ML_DOMAIN;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
        self.check_keys_values()?;
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
        push_float(&mut attrs, "default_float", self.default_float);
        push_int(&mut attrs, "default_int64", self.default_int64);
        push_string(&mut attrs, "default_string", &self.default_string);
        push_tensor(&mut attrs, "default_tensor", &self.default_tensor);
        push_floats(&mut attrs, "keys_floats", &self.keys_floats);
        push_ints(&mut attrs, "keys_int64s", &self.keys_int64s);
        push_strings(&mut attrs, "keys_strings", &self.keys_strings);
        push_tensor(&mut attrs, "keys_tensor", &self.keys_tensor);
        push_floats(&mut attrs, "values_floats", &self.values_floats);
        push_ints(&mut attrs, "values_int64s", &self.values_int64s);
        push_strings(&mut attrs, "values_strings", &self.values_strings);
        push_tensor(&mut attrs, "values_tensor", &self.values_tensor);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let encoder = LabelEncoder {
            default_float: table.opt_float("default_float")?,
            default_int64: table.opt_int("default_int64")?,
            default_string: table.opt_string("default_string")?,
            default_tensor: table.opt_tensor("default_tensor")?,
            keys_floats: table.opt_floats("keys_floats")?,
            keys_int64s: table.opt_ints("keys_int64s")?,
            keys_strings: table.opt_strings("keys_strings")?,
            keys_tensor: table.opt_tensor("keys_tensor")?,
            values_floats: table.opt_floats("values_floats")?,
            values_int64s: table.opt_ints("values_int64s")?,
            values_strings: table.opt_strings("values_strings")?,
            values_tensor: table.opt_tensor("values_tensor")?,
        };
        table.finish()?;
        encoder.check_keys_values()?;
        Ok(encoder)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OneHotEncoder {
    pub cats_int64s: Option<Vec<i64>>,
    pub cats_strings: Option<Vec<String>>,
    pub zeros: Option<i64>,
}

impl OnnxOp for OneHotEncoder {
    const OP_TYPE: &'static str = "OneHotEncoder";
    const DOMAIN: &'static str = ML_DOMAIN;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
        check_oneof_cats(
            Self::OP_TYPE,
            self.cats_int64s.is_some(),
            self.cats_strings.is_some(),
        )?;
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
        push_ints(&mut attrs, "cats_int64s", &self.cats_int64s);
        push_strings(&mut attrs, "cats_strings", &self.cats_strings);
        push_int(&mut attrs, "zeros", self.zeros);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let cats_int64s = table.opt_ints("cats_int64s")?;
        let cats_strings = table.opt_strings("cats_strings")?;
        let zeros = table.opt_int("zeros")?;
        table.finish()?;
        check_oneof_cats(Self::OP_TYPE, cats_int64s.is_some(), cats_strings.is_some())?;
        Ok(OneHotEncoder {
            cats_int64s,
            cats_strings,
            zeros,
        })
    }
}

fn check_oneof_cats(op: &'static str, ints: bool, strings: bool) -> Result<(), OpError> {
    if ints == strings {
        return Err(OpError::InvalidValue {
            op,
            attr: "cats_*".to_string(),
            detail: "exactly one of cats_int64s and cats_strings must be set".to_string(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct DictVectorizer {
    pub int64_vocabulary: Option<Vec<i64>>,
    pub string_vocabulary: Option<Vec<String>>,
}

impl OnnxOp for DictVectorizer {
    const OP_TYPE: &'static str = "DictVectorizer";
    const DOMAIN: &'static str = ML_DOMAIN;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
        check_oneof_vocab(
            Self::OP_TYPE,
            self.int64_vocabulary.is_some(),
            self.string_vocabulary.is_some(),
        )?;
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
        push_ints(&mut attrs, "int64_vocabulary", &self.int64_vocabulary);
        push_strings(&mut attrs, "string_vocabulary", &self.string_vocabulary);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let int64_vocabulary = table.opt_ints("int64_vocabulary")?;
        let string_vocabulary = table.opt_strings("string_vocabulary")?;
        table.finish()?;
        check_oneof_vocab(
            Self::OP_TYPE,
            int64_vocabulary.is_some(),
            string_vocabulary.is_some(),
        )?;
        Ok(DictVectorizer {
            int64_vocabulary,
            string_vocabulary,
        })
    }
}

fn check_oneof_vocab(op: &'static str, ints: bool, strings: bool) -> Result<(), OpError> {
    if ints == strings {
        return Err(OpError::InvalidValue {
            op,
            attr: "*_vocabulary".to_string(),
            detail: "exactly one of int64_vocabulary and string_vocabulary must be set".to_string(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureVectorizer {
    pub inputdimensions: Option<Vec<i64>>,
}

impl OnnxOp for FeatureVectorizer {
    const OP_TYPE: &'static str = "FeatureVectorizer";
    const DOMAIN: &'static str = ML_DOMAIN;

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
        let mut attrs: Vec<Attribute> = Vec::new();
        push_ints(&mut attrs, "inputdimensions", &self.inputdimensions);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n >= 1, "1..", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let inputdimensions = table.opt_ints("inputdimensions")?;
        table.finish()?;
        Ok(FeatureVectorizer { inputdimensions })
    }
}
