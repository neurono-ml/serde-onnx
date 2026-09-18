use crate::proto::tensor_proto::DataType;

pub const IR_VERSION: i64 = 10;

pub const ONNX_OPSET_VERSION: i64 = 21;

pub const ML_OPSET_VERSION: i64 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum ElemType {
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    Bfloat16 = 16,
    Float8e4m3fn = 17,
    Float8e4m3fnuz = 18,
    Float8e5m2 = 19,
    Float8e5m2fnuz = 20,
    Uint4 = 21,
    Int4 = 22,
    Float4e2m1 = 23,
    Float8e8m0 = 24,
    Uint2 = 25,
    Int2 = 26,
}

impl ElemType {
    pub const fn disc(self) -> i32 {
        self as i32
    }

    pub fn from_disc(v: i32) -> Option<Self> {
        DataType::try_from(v).ok().and_then(|d| match d {
            DataType::Undefined => Some(Self::Undefined),
            DataType::Float => Some(Self::Float),
            DataType::Uint8 => Some(Self::Uint8),
            DataType::Int8 => Some(Self::Int8),
            DataType::Uint16 => Some(Self::Uint16),
            DataType::Int16 => Some(Self::Int16),
            DataType::Int32 => Some(Self::Int32),
            DataType::Int64 => Some(Self::Int64),
            DataType::String => Some(Self::String),
            DataType::Bool => Some(Self::Bool),
            DataType::Float16 => Some(Self::Float16),
            DataType::Double => Some(Self::Double),
            DataType::Uint32 => Some(Self::Uint32),
            DataType::Uint64 => Some(Self::Uint64),
            DataType::Complex64 => Some(Self::Complex64),
            DataType::Complex128 => Some(Self::Complex128),
            DataType::Bfloat16 => Some(Self::Bfloat16),
            DataType::Float8e4m3fn => Some(Self::Float8e4m3fn),
            DataType::Float8e4m3fnuz => Some(Self::Float8e4m3fnuz),
            DataType::Float8e5m2 => Some(Self::Float8e5m2),
            DataType::Float8e5m2fnuz => Some(Self::Float8e5m2fnuz),
            DataType::Uint4 => Some(Self::Uint4),
            DataType::Int4 => Some(Self::Int4),
            DataType::Float4e2m1 => Some(Self::Float4e2m1),
            DataType::Float8e8m0 => Some(Self::Float8e8m0),
            DataType::Uint2 => Some(Self::Uint2),
            _ => None,
        })
    }

    pub const fn is_fixed_width(self) -> bool {
        !matches!(
            self,
            Self::Undefined
                | Self::Uint4
                | Self::Int4
                | Self::Float4e2m1
                | Self::Uint2
                | Self::Int2
        )
    }

    pub const fn is_numeric(self) -> bool {
        !matches!(self, Self::Undefined | Self::String)
    }

    pub const fn is_quantized_integer(self) -> bool {
        matches!(
            self,
            Self::Uint2
                | Self::Int2
                | Self::Uint4
                | Self::Int4
                | Self::Uint8
                | Self::Int8
                | Self::Uint16
                | Self::Int16
                | Self::Int32
                | Self::Int64
        )
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Undefined => "UNDEFINED",
            Self::Float => "FLOAT",
            Self::Uint8 => "UINT8",
            Self::Int8 => "INT8",
            Self::Uint16 => "UINT16",
            Self::Int16 => "INT16",
            Self::Int32 => "INT32",
            Self::Int64 => "INT64",
            Self::String => "STRING",
            Self::Bool => "BOOL",
            Self::Float16 => "FLOAT16",
            Self::Double => "DOUBLE",
            Self::Uint32 => "UINT32",
            Self::Uint64 => "UINT64",
            Self::Complex64 => "COMPLEX64",
            Self::Complex128 => "COMPLEX128",
            Self::Bfloat16 => "BFLOAT16",
            Self::Float8e4m3fn => "FLOAT8E4M3FN",
            Self::Float8e4m3fnuz => "FLOAT8E4M3FNUZ",
            Self::Float8e5m2 => "FLOAT8E5M2",
            Self::Float8e5m2fnuz => "FLOAT8E5M2FNUZ",
            Self::Uint4 => "UINT4",
            Self::Int4 => "INT4",
            Self::Float4e2m1 => "FLOAT4E2M1",
            Self::Float8e8m0 => "FLOAT8E8M0",
            Self::Uint2 => "UINT2",
            Self::Int2 => "INT2",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Fixed(i64),
    Param(String),
    Unknown,
}

impl Dim {
    pub fn is_fixed(&self) -> bool {
        matches!(self, Dim::Fixed(_))
    }

    pub fn fixed_value(&self) -> Option<i64> {
        match self {
            Dim::Fixed(v) => Some(*v),
            _ => None,
        }
    }
}

pub type Shape = Vec<Dim>;

pub type MaybeRanked = Option<Shape>;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorType {
    pub elem: ElemType,
    pub shape: MaybeRanked,
}

impl TensorType {
    pub fn scalar(elem: ElemType) -> Self {
        TensorType {
            elem,
            shape: Some(Vec::new()),
        }
    }

    pub fn unranked(elem: ElemType) -> Self {
        TensorType { elem, shape: None }
    }

    pub fn shaped(elem: ElemType, dims: impl IntoIterator<Item = Dim>) -> Self {
        TensorType {
            elem,
            shape: Some(dims.into_iter().collect()),
        }
    }

    pub const fn is_scalar(&self) -> bool {
        matches!(&self.shape, Some(s) if s.is_empty())
    }

    pub fn rank(&self) -> Option<usize> {
        self.shape.as_ref().map(|s| s.len())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueType {
    Tensor(TensorType),
    SparseTensor(TensorType),
    Sequence(Box<ValueType>),
    Map {
        key: ElemType,
        value: Box<ValueType>,
    },
    Optional(Box<ValueType>),
    Opaque {
        domain: Option<String>,
        name: Option<String>,
    },
}

impl ValueType {
    pub fn tensor(elem: ElemType, shape: MaybeRanked) -> Self {
        ValueType::Tensor(TensorType { elem, shape })
    }

    pub fn sequence(inner: ValueType) -> Self {
        ValueType::Sequence(Box::new(inner))
    }

    pub fn optional(inner: ValueType) -> Self {
        ValueType::Optional(Box::new(inner))
    }
}
