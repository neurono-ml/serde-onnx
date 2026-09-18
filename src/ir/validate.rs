use super::graph::{Graph, Model};
use super::types::Dim;

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    GraphMissingName,
    DuplicateOutput { node_index: usize, name: String },
    DuplicateInput { node_index: usize, name: String },
    DuplicateValueInfo { kind: &'static str, name: String },
    DuplicateInitializer { name: String },
    UndefinedValue { node_index: usize, name: String },
    Cycle { path: Vec<String> },
    MissingShape { kind: &'static str, name: String },
    MultipleGraphOutputsForValue { name: String },
}

impl core::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::GraphMissingName => write!(f, "graph has no name"),
            Self::DuplicateOutput { node_index, name } => {
                write!(
                    f,
                    "node {node_index} declares duplicate output name {name:?}"
                )
            }
            Self::DuplicateInput { node_index, name } => {
                write!(
                    f,
                    "node {node_index} declares duplicate input name {name:?}"
                )
            }
            Self::DuplicateValueInfo { kind, name } => {
                write!(f, "duplicate {kind} value info name {name:?}")
            }
            Self::DuplicateInitializer { name } => {
                write!(f, "duplicate initializer name {name:?}")
            }
            Self::UndefinedValue { node_index, name } => {
                write!(f, "node {node_index} references undefined value {name:?}")
            }
            Self::Cycle { path } => {
                write!(f, "graph contains a cycle: {}", path.join(" -> "))
            }
            Self::MissingShape { kind, name } => {
                write!(
                    f,
                    "{kind} {name:?} of the main graph has no type/shape (rank)"
                )
            }
            Self::MultipleGraphOutputsForValue { name } => {
                write!(f, "value {name:?} appears more than once in graph outputs")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationReport {
    pub errors: Vec<ValidationError>,
}

impl ValidationReport {
    pub const fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn into_result(self) -> Result<(), Vec<ValidationError>> {
        if self.is_ok() {
            Ok(())
        } else {
            Err(self.errors)
        }
    }
}

impl core::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (i, e) in self.errors.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationReport {}

pub fn validate_graph(graph: &Graph) -> ValidationReport {
    let mut errors = Vec::new();

    if graph.name.is_empty() {
        errors.push(ValidationError::GraphMissingName);
    }

    for vi in graph
        .inputs
        .iter()
        .chain(&graph.outputs)
        .chain(&graph.value_info)
    {
        if vi.name.is_empty() {
            errors.push(ValidationError::DuplicateValueInfo {
                kind: "unnamed",
                name: vi.name.clone(),
            });
        }
    }

    for (kind, list) in [
        ("input", &graph.inputs),
        ("output", &graph.outputs),
        ("value_info", &graph.value_info),
    ] {
        let mut seen = std::collections::HashSet::new();
        for vi in list {
            if !seen.insert(vi.name.clone()) {
                errors.push(ValidationError::DuplicateValueInfo {
                    kind,
                    name: vi.name.clone(),
                });
            }
        }
    }

    let mut init_names = std::collections::HashSet::new();
    for t in &graph.initializers {
        if !init_names.insert(t.name.clone()) {
            errors.push(ValidationError::DuplicateInitializer {
                name: t.name.clone(),
            });
        }
    }

    let mut defined: std::collections::HashSet<String> = init_names;
    for vi in &graph.inputs {
        defined.insert(vi.name.clone());
    }

    let mut produced_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        for inp in &node.inputs {
            if inp.is_empty() {
                continue;
            }
            if !defined.contains(inp) {
                errors.push(ValidationError::UndefinedValue {
                    node_index: idx,
                    name: inp.clone(),
                });
            }
        }
        for out in &node.outputs {
            if out.is_empty() {
                continue;
            }
            if produced_names.contains(out) {
                errors.push(ValidationError::DuplicateOutput {
                    node_index: idx,
                    name: out.clone(),
                });
            } else {
                produced_names.insert(out.clone());
                defined.insert(out.clone());
            }
        }
    }

    let has_cycle = detect_cycle(graph);
    if let Some(path) = has_cycle {
        errors.push(ValidationError::Cycle { path });
    }

    for vi in &graph.outputs {
        let unshaped = vi
            .value_type
            .as_ref()
            .is_none_or(|vt| !value_type_has_rank(vt));
        if unshaped {
            errors.push(ValidationError::MissingShape {
                kind: "output",
                name: vi.name.clone(),
            });
        }
    }
    for vi in &graph.inputs {
        let unshaped = vi
            .value_type
            .as_ref()
            .is_none_or(|vt| !value_type_has_rank(vt));
        if unshaped {
            errors.push(ValidationError::MissingShape {
                kind: "input",
                name: vi.name.clone(),
            });
        }
    }

    ValidationReport { errors }
}

fn value_type_has_rank(vt: &super::types::ValueType) -> bool {
    match vt {
        super::types::ValueType::Tensor(t) | super::types::ValueType::SparseTensor(t) => {
            t.shape.is_some()
        }
        super::types::ValueType::Sequence(inner) | super::types::ValueType::Optional(inner) => {
            value_type_has_rank(inner)
        }
        super::types::ValueType::Map { value, .. } => value_type_has_rank(value),
        super::types::ValueType::Opaque { .. } => true,
    }
}

fn detect_cycle(graph: &Graph) -> Option<Vec<String>> {
    let n = graph.nodes.len();
    if n == 0 {
        return None;
    }
    let mut index_of: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        for out in &node.outputs {
            if !out.is_empty() {
                index_of.entry(out.as_str()).or_insert(i);
            }
        }
    }

    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node) in graph.nodes.iter().enumerate() {
        for inp in &node.inputs {
            if inp.is_empty() {
                continue;
            }
            if let Some(&j) = index_of.get(inp.as_str())
                && j != i
            {
                deps[i].push(j);
            }
        }
    }

    const WHITE: u8 = 0;
    const GRAY: u8 = 1;
    const BLACK: u8 = 2;
    let mut color = vec![WHITE; n];
    let mut stack: Vec<usize> = Vec::new();

    for start in 0..n {
        if color[start] != WHITE {
            continue;
        }
        stack.clear();
        stack.push(start);
        color[start] = GRAY;
        let mut iter_pos = vec![0usize; n];

        while let Some(&cur) = stack.last() {
            if iter_pos[cur] < deps[cur].len() {
                let next = deps[cur][iter_pos[cur]];
                iter_pos[cur] += 1;
                match color[next] {
                    GRAY => {
                        let pos = stack.iter().rposition(|&x| x == next).unwrap_or(0);
                        let mut path: Vec<String> = stack[pos..]
                            .iter()
                            .map(|&i| node_label(&graph.nodes[i]))
                            .collect();
                        path.push(node_label(&graph.nodes[next]));
                        return Some(path);
                    }
                    WHITE => {
                        color[next] = GRAY;
                        stack.push(next);
                    }
                    _ => {}
                }
            } else {
                color[cur] = BLACK;
                stack.pop();
            }
        }
    }
    None
}

fn node_label(node: &super::graph::Node) -> String {
    node.name
        .clone()
        .unwrap_or_else(|| format!("{}({})", node.op_type, node.outputs.join(",")))
}

pub fn validate_model(model: &Model) -> ValidationReport {
    validate_graph(&model.graph)
}

pub fn is_dim_fixed(d: &Dim) -> bool {
    matches!(d, Dim::Fixed(_))
}
