use super::{
    AttrTable, Emitted, OnnxOp, OpError, build_node, check_counts, check_op, push_floats, push_int,
    push_ints, push_string, push_strings,
};
use crate::ir::{ML_DOMAIN, Node};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClassLabels {
    pub ints: Option<Vec<i64>>,
    pub strings: Option<Vec<String>>,
}

impl ClassLabels {
    fn check(&self, op: &'static str) -> Result<(), OpError> {
        if self.ints.is_some() == self.strings.is_some() {
            return Err(OpError::InvalidValue {
                op,
                attr: "classlabels_*".to_string(),
                detail: "exactly one of classlabels_ints and classlabels_strings must be set"
                    .to_string(),
            });
        }
        Ok(())
    }

    fn emit(&self, attrs: &mut Vec<crate::ir::Attribute>) {
        push_ints(attrs, "classlabels_ints", &self.ints);
        push_strings(attrs, "classlabels_strings", &self.strings);
    }

    fn parse(op: &'static str, table: &mut AttrTable) -> Result<Self, OpError> {
        let labels = ClassLabels {
            ints: table.opt_ints("classlabels_ints")?,
            strings: table.opt_strings("classlabels_strings")?,
        };
        labels.check(op)?;
        Ok(labels)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearClassifier {
    pub classlabels: ClassLabels,
    pub coefficients: Vec<f32>,
    pub intercepts: Option<Vec<f32>>,
    pub multi_class: Option<i64>,
    pub post_transform: Option<String>,
}

impl LinearClassifier {
    pub fn with_int_labels(labels: Vec<i64>, coefficients: Vec<f32>) -> Result<Self, OpError> {
        let classifier = LinearClassifier {
            classlabels: ClassLabels {
                ints: Some(labels),
                strings: None,
            },
            coefficients,
            intercepts: None,
            multi_class: None,
            post_transform: None,
        };
        classifier.check()?;
        Ok(classifier)
    }

    pub fn with_string_labels(
        labels: Vec<String>,
        coefficients: Vec<f32>,
    ) -> Result<Self, OpError> {
        let classifier = LinearClassifier {
            classlabels: ClassLabels {
                ints: None,
                strings: Some(labels),
            },
            coefficients,
            intercepts: None,
            multi_class: None,
            post_transform: None,
        };
        classifier.check()?;
        Ok(classifier)
    }

    fn check(&self) -> Result<(), OpError> {
        self.classlabels.check(Self::OP_TYPE)?;
        if self.coefficients.is_empty() {
            return Err(OpError::MissingAttribute {
                op: Self::OP_TYPE,
                attr: "coefficients",
            });
        }
        Ok(())
    }
}

impl OnnxOp for LinearClassifier {
    const OP_TYPE: &'static str = "LinearClassifier";
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
        self.classlabels.emit(&mut attrs);
        attrs.push(crate::ir::Attribute::floats(
            "coefficients",
            self.coefficients.clone(),
        ));
        push_floats(&mut attrs, "intercepts", &self.intercepts);
        push_int(&mut attrs, "multi_class", self.multi_class);
        push_string(&mut attrs, "post_transform", &self.post_transform);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 2, "2")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let classlabels = ClassLabels::parse(Self::OP_TYPE, &mut table)?;
        let coefficients = table.req_floats("coefficients")?;
        let intercepts = table.opt_floats("intercepts")?;
        let multi_class = table.opt_int("multi_class")?;
        let post_transform = table.opt_string("post_transform")?;
        table.finish()?;
        let classifier = LinearClassifier {
            classlabels,
            coefficients,
            intercepts,
            multi_class,
            post_transform,
        };
        classifier.check()?;
        Ok(classifier)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearRegressor {
    pub coefficients: Vec<f32>,
    pub intercepts: Option<Vec<f32>>,
    pub post_transform: Option<String>,
    pub targets: Option<i64>,
}

impl OnnxOp for LinearRegressor {
    const OP_TYPE: &'static str = "LinearRegressor";
    const DOMAIN: &'static str = ML_DOMAIN;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
        if self.coefficients.is_empty() {
            return Err(OpError::MissingAttribute {
                op: Self::OP_TYPE,
                attr: "coefficients",
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
        attrs.push(crate::ir::Attribute::floats(
            "coefficients",
            self.coefficients.clone(),
        ));
        push_floats(&mut attrs, "intercepts", &self.intercepts);
        push_string(&mut attrs, "post_transform", &self.post_transform);
        push_int(&mut attrs, "targets", self.targets);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let coefficients = table.req_floats("coefficients")?;
        let intercepts = table.opt_floats("intercepts")?;
        let post_transform = table.opt_string("post_transform")?;
        let targets = table.opt_int("targets")?;
        table.finish()?;
        Ok(LinearRegressor {
            coefficients,
            intercepts,
            post_transform,
            targets,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SvmCommon {
    pub coefficients: Option<Vec<f32>>,
    pub kernel_params: Option<Vec<f32>>,
    pub kernel_type: Option<String>,
    pub post_transform: Option<String>,
    pub prob_a: Option<Vec<f32>>,
    pub prob_b: Option<Vec<f32>>,
    pub rho: Option<Vec<f32>>,
    pub support_vectors: Option<Vec<f32>>,
}

impl SvmCommon {
    fn emit(&self, attrs: &mut Vec<crate::ir::Attribute>) {
        push_floats(attrs, "coefficients", &self.coefficients);
        push_floats(attrs, "kernel_params", &self.kernel_params);
        push_string(attrs, "kernel_type", &self.kernel_type);
        push_string(attrs, "post_transform", &self.post_transform);
        push_floats(attrs, "prob_a", &self.prob_a);
        push_floats(attrs, "prob_b", &self.prob_b);
        push_floats(attrs, "rho", &self.rho);
        push_floats(attrs, "support_vectors", &self.support_vectors);
    }

    fn parse(table: &mut AttrTable) -> Result<Self, OpError> {
        Ok(SvmCommon {
            coefficients: table.opt_floats("coefficients")?,
            kernel_params: table.opt_floats("kernel_params")?,
            kernel_type: table.opt_string("kernel_type")?,
            post_transform: table.opt_string("post_transform")?,
            prob_a: table.opt_floats("prob_a")?,
            prob_b: table.opt_floats("prob_b")?,
            rho: table.opt_floats("rho")?,
            support_vectors: table.opt_floats("support_vectors")?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SvmClassifier {
    pub classlabels: ClassLabels,
    pub common: SvmCommon,
    pub vectors_per_class: Option<Vec<i64>>,
}

impl OnnxOp for SvmClassifier {
    const OP_TYPE: &'static str = "SVMClassifier";
    const DOMAIN: &'static str = ML_DOMAIN;

    fn to_node(&self, inputs: Vec<String>, outputs: Vec<String>) -> Result<Emitted, OpError> {
        self.classlabels.check(Self::OP_TYPE)?;
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
        self.classlabels.emit(&mut attrs);
        self.common.emit(&mut attrs);
        push_ints(&mut attrs, "vectors_per_class", &self.vectors_per_class);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 2, "2")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let classlabels = ClassLabels::parse(Self::OP_TYPE, &mut table)?;
        let common = SvmCommon::parse(&mut table)?;
        let vectors_per_class = table.opt_ints("vectors_per_class")?;
        table.finish()?;
        Ok(SvmClassifier {
            classlabels,
            common,
            vectors_per_class,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SvmRegressor {
    pub common: SvmCommon,
    pub n_supports: Option<i64>,
    pub one_class: Option<i64>,
}

impl OnnxOp for SvmRegressor {
    const OP_TYPE: &'static str = "SVMRegressor";
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
        self.common.emit(&mut attrs);
        push_int(&mut attrs, "n_supports", self.n_supports);
        push_int(&mut attrs, "one_class", self.one_class);
        Ok(build_node::<Self>(attrs, inputs, outputs))
    }

    fn from_node(node: &Node) -> Result<Self, OpError> {
        check_op::<Self>(node)?;
        check_counts(Self::OP_TYPE, node, |n| n == 1, "1", |n| n == 1, "1")?;
        let mut table = AttrTable::new(Self::OP_TYPE, node)?;
        let common = SvmCommon::parse(&mut table)?;
        let n_supports = table.opt_int("n_supports")?;
        let one_class = table.opt_int("one_class")?;
        table.finish()?;
        Ok(SvmRegressor {
            common,
            n_supports,
            one_class,
        })
    }
}
