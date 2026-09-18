use super::numeric::Scalar;
use super::types::{Dim, ElemType, Shape};
use num_traits::NumCast;

#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    Bool(Vec<bool>),
    String(Vec<String>),
    F16(Vec<u16>),
    Bf16(Vec<u16>),
    Complex64(Vec<f32>),
    Complex128(Vec<f64>),
    Float8e4m3fn(Vec<u8>),
    Float8e4m3fnuz(Vec<u8>),
    Float8e5m2(Vec<u8>),
    Float8e5m2fnuz(Vec<u8>),
    Float8e8m0(Vec<u8>),
    Uint4(Vec<u8>),
    Int4(Vec<u8>),
    Float4e2m1(Vec<u8>),
    Uint2(Vec<u8>),
    Int2(Vec<u8>),
    RawBytes { elem: ElemType, bytes: Vec<u8> },
}

impl TensorData {
    pub const fn elem_type(&self) -> ElemType {
        match self {
            Self::F32(_) => ElemType::Float,
            Self::F64(_) => ElemType::Double,
            Self::I8(_) => ElemType::Int8,
            Self::I16(_) => ElemType::Int16,
            Self::I32(_) => ElemType::Int32,
            Self::I64(_) => ElemType::Int64,
            Self::U8(_) => ElemType::Uint8,
            Self::U16(_) => ElemType::Uint16,
            Self::U32(_) => ElemType::Uint32,
            Self::U64(_) => ElemType::Uint64,
            Self::Bool(_) => ElemType::Bool,
            Self::String(_) => ElemType::String,
            Self::F16(_) => ElemType::Float16,
            Self::Bf16(_) => ElemType::Bfloat16,
            Self::Complex64(_) => ElemType::Complex64,
            Self::Complex128(_) => ElemType::Complex128,
            Self::Float8e4m3fn(_) => ElemType::Float8e4m3fn,
            Self::Float8e4m3fnuz(_) => ElemType::Float8e4m3fnuz,
            Self::Float8e5m2(_) => ElemType::Float8e5m2,
            Self::Float8e5m2fnuz(_) => ElemType::Float8e5m2fnuz,
            Self::Float8e8m0(_) => ElemType::Float8e8m0,
            Self::Uint4(_) => ElemType::Uint4,
            Self::Int4(_) => ElemType::Int4,
            Self::Float4e2m1(_) => ElemType::Float4e2m1,
            Self::Uint2(_) => ElemType::Uint2,
            Self::Int2(_) => ElemType::Int2,
            Self::RawBytes { elem, .. } => *elem,
        }
    }

    pub fn len(&self) -> Option<usize> {
        match self {
            Self::F32(v) => Some(v.len()),
            Self::F64(v) => Some(v.len()),
            Self::I8(v) => Some(v.len()),
            Self::I16(v) => Some(v.len()),
            Self::I32(v) => Some(v.len()),
            Self::I64(v) => Some(v.len()),
            Self::U8(v) => Some(v.len()),
            Self::U16(v) => Some(v.len()),
            Self::U32(v) => Some(v.len()),
            Self::U64(v) => Some(v.len()),
            Self::Bool(v) => Some(v.len()),
            Self::String(v) => Some(v.len()),
            Self::F16(v) => Some(v.len()),
            Self::Bf16(v) => Some(v.len()),
            Self::Complex64(v) => Some(v.len()),
            Self::Complex128(v) => Some(v.len()),
            Self::Float8e4m3fn(v) => Some(v.len()),
            Self::Float8e4m3fnuz(v) => Some(v.len()),
            Self::Float8e5m2(v) => Some(v.len()),
            Self::Float8e5m2fnuz(v) => Some(v.len()),
            Self::Float8e8m0(v) => Some(v.len()),
            Self::Uint4(v) => Some(v.len()),
            Self::Int4(v) => Some(v.len()),
            Self::Float4e2m1(v) => Some(v.len()),
            Self::Uint2(v) => Some(v.len()),
            Self::Int2(v) => Some(v.len()),
            Self::RawBytes { .. } => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len().is_some_and(|n| n == 0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternalData {
    pub location: String,
    pub offset: Option<u64>,
    pub length: Option<u64>,
    pub checksum: Option<String>,
}

impl ExternalData {
    pub fn from_entries(entries: &[(String, String)]) -> Result<Self, TensorError> {
        let mut location = None;
        let mut offset = None;
        let mut length = None;
        let mut checksum = None;
        for (k, v) in entries {
            match k.as_str() {
                "location" => location = Some(v.clone()),
                "offset" => {
                    offset = Some(v.parse::<u64>().map_err(|_| {
                        TensorError::InvalidExternalData(format!("non-numeric offset {v:?}"))
                    })?)
                }
                "length" => {
                    length = Some(v.parse::<u64>().map_err(|_| {
                        TensorError::InvalidExternalData(format!("non-numeric length {v:?}"))
                    })?)
                }
                "checksum" => checksum = Some(v.clone()),
                _ => {}
            }
        }
        Ok(ExternalData {
            location: location.ok_or_else(|| {
                TensorError::InvalidExternalData("external_data without location key".to_string())
            })?,
            offset,
            length,
            checksum,
        })
    }

    pub fn to_entries(&self) -> Vec<(String, String)> {
        let mut entries = Vec::with_capacity(4);
        entries.push(("location".to_string(), self.location.clone()));
        if let Some(o) = self.offset {
            entries.push(("offset".to_string(), o.to_string()));
        }
        if let Some(l) = self.length {
            entries.push(("length".to_string(), l.to_string()));
        }
        if let Some(c) = &self.checksum {
            entries.push(("checksum".to_string(), c.clone()));
        }
        entries
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorStorage {
    Inline(TensorData),
    External(ExternalData),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub elem: ElemType,
    pub shape: Shape,
    pub data: TensorStorage,
    pub doc_string: String,
}

impl Tensor {
    pub fn new(name: impl Into<String>, elem: ElemType, shape: Shape, data: TensorData) -> Self {
        Tensor {
            name: name.into(),
            elem,
            shape,
            data: TensorStorage::Inline(data),
            doc_string: String::new(),
        }
    }

    pub fn external(
        name: impl Into<String>,
        elem: ElemType,
        shape: Shape,
        external: ExternalData,
    ) -> Self {
        Tensor {
            name: name.into(),
            elem,
            shape,
            data: TensorStorage::External(external),
            doc_string: String::new(),
        }
    }

    pub fn expected_len(&self) -> Option<usize> {
        self.shape
            .iter()
            .try_fold(1i64, |acc, d| match d {
                Dim::Fixed(v) => acc.checked_mul(*v),
                _ => None,
            })
            .and_then(|n| usize::try_from(n).ok())
    }

    pub fn data(&self) -> Option<&TensorData> {
        match &self.data {
            TensorStorage::Inline(d) => Some(d),
            TensorStorage::External(_) => None,
        }
    }

    pub fn as_slice<T: Scalar>(&self) -> Result<&[T], TensorError> {
        let data = self.data().ok_or(TensorError::TypeError {
            expected: T::ELEM_TYPE,
            got: self.elem,
        })?;
        match data {
            TensorData::F32(v) if T::ELEM_TYPE == ElemType::Float => as_slice_ref::<T, f32>(v),
            TensorData::F64(v) if T::ELEM_TYPE == ElemType::Double => as_slice_ref::<T, f64>(v),
            TensorData::I8(v) if T::ELEM_TYPE == ElemType::Int8 => as_slice_ref::<T, i8>(v),
            TensorData::I16(v) if T::ELEM_TYPE == ElemType::Int16 => as_slice_ref::<T, i16>(v),
            TensorData::I32(v) if T::ELEM_TYPE == ElemType::Int32 => as_slice_ref::<T, i32>(v),
            TensorData::I64(v) if T::ELEM_TYPE == ElemType::Int64 => as_slice_ref::<T, i64>(v),
            TensorData::U8(v) if T::ELEM_TYPE == ElemType::Uint8 => as_slice_ref::<T, u8>(v),
            TensorData::U16(v) if T::ELEM_TYPE == ElemType::Uint16 => as_slice_ref::<T, u16>(v),
            TensorData::U32(v) if T::ELEM_TYPE == ElemType::Uint32 => as_slice_ref::<T, u32>(v),
            TensorData::U64(v) if T::ELEM_TYPE == ElemType::Uint64 => as_slice_ref::<T, u64>(v),
            TensorData::Bool(v) if T::ELEM_TYPE == ElemType::Bool => as_slice_ref::<T, bool>(v),
            _ => Err(TensorError::TypeError {
                expected: T::ELEM_TYPE,
                got: data.elem_type(),
            }),
        }
    }

    pub fn to_vec<T: Scalar + NumCast>(&self) -> Result<Vec<T>, TensorError> {
        let data = self.data().ok_or(TensorError::TypeError {
            expected: T::ELEM_TYPE,
            got: self.elem,
        })?;
        match data {
            TensorData::F32(v) => to_vec_checked::<T, f32>(v),
            TensorData::F64(v) => to_vec_checked::<T, f64>(v),
            TensorData::I8(v) => to_vec_checked::<T, i8>(v),
            TensorData::I16(v) => to_vec_checked::<T, i16>(v),
            TensorData::I32(v) => to_vec_checked::<T, i32>(v),
            TensorData::I64(v) => to_vec_checked::<T, i64>(v),
            TensorData::U8(v) => to_vec_checked::<T, u8>(v),
            TensorData::U16(v) => to_vec_checked::<T, u16>(v),
            TensorData::U32(v) => to_vec_checked::<T, u32>(v),
            TensorData::U64(v) => to_vec_checked::<T, u64>(v),
            TensorData::Bool(v) if T::ELEM_TYPE == ElemType::Bool => to_vec_bool::<T>(v),
            TensorData::String(_) => Err(TensorError::TypeError {
                expected: T::ELEM_TYPE,
                got: ElemType::String,
            }),
            TensorData::F16(bits) => to_vec_f16::<T, half::f16>(bits),
            TensorData::Bf16(bits) => to_vec_f16::<T, half::bf16>(bits),
            _ => Err(TensorError::TypeError {
                expected: T::ELEM_TYPE,
                got: data.elem_type(),
            }),
        }
    }
}

fn as_slice_ref<T: Scalar, S: Scalar>(v: &[S]) -> Result<&[T], TensorError> {
    if core::mem::size_of::<T>() == core::mem::size_of::<S>() {
        Ok(unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<T>(), v.len()) })
    } else {
        Err(TensorError::TypeError {
            expected: T::ELEM_TYPE,
            got: S::ELEM_TYPE,
        })
    }
}

fn to_vec_checked<T: Scalar + NumCast, S: Scalar + NumCast>(
    v: &[S],
) -> Result<Vec<T>, TensorError> {
    let mut out = Vec::with_capacity(v.len());
    for (i, item) in v.iter().enumerate() {
        match try_cast_lossless::<S, T>(*item) {
            Ok(x) => out.push(x),
            Err(reason) => {
                return Err(TensorError::CastError {
                    index: i,
                    from: S::ELEM_TYPE,
                    to: T::ELEM_TYPE,
                    reason,
                });
            }
        }
    }
    Ok(out)
}

fn try_cast_lossless<S: Scalar + NumCast, T: Scalar + NumCast>(v: S) -> Result<T, CastLossReason> {
    let src_float = elem_is_float(S::ELEM_TYPE);
    let dst_float = elem_is_float(T::ELEM_TYPE);
    let to = match NumCast::from(v) {
        Some(x) => x,
        None => return Err(CastLossReason::OutOfRange),
    };
    if src_float && !dst_float {
        let f = v.to_f64().unwrap_or(0.0);
        if f.fract() != 0.0 {
            return Err(CastLossReason::NonIntegral);
        }
    }
    if !round_identifies(v, &to) {
        return Err(CastLossReason::OutOfRange);
    }
    Ok(to)
}

fn elem_is_float(e: ElemType) -> bool {
    matches!(
        e,
        ElemType::Float | ElemType::Double | ElemType::Float16 | ElemType::Bfloat16
    )
}

fn round_identifies<S: NumCast, T: NumCast + Copy>(from: S, to: &T) -> bool {
    let back = match S::from(*to) {
        Some(b) => b,
        None => return false,
    };
    match (from.to_f64(), back.to_f64()) {
        (Some(f), Some(b)) => f == b,
        _ => true,
    }
}

fn to_vec_bool<T: Scalar + NumCast>(v: &[bool]) -> Result<Vec<T>, TensorError> {
    let mut out = Vec::with_capacity(v.len());
    for (i, item) in v.iter().enumerate() {
        match NumCast::from(<u8 as From<bool>>::from(*item)) {
            Some(x) => out.push(x),
            None => {
                return Err(TensorError::CastError {
                    index: i,
                    from: ElemType::Bool,
                    to: T::ELEM_TYPE,
                    reason: CastLossReason::OutOfRange,
                });
            }
        }
    }
    Ok(out)
}

fn to_vec_f16<T: Scalar + NumCast, H: HalfFloat>(bits: &[u16]) -> Result<Vec<T>, TensorError> {
    let vals: Vec<f32> = bits.iter().map(|&b| H::from_bits(b).to_f32()).collect();
    to_vec_checked::<T, f32>(&vals)
}

fn to_vec_f16_from_f32<H: HalfFloat>(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| H::from_f32(x).to_bits()).collect()
}

pub trait HalfFloat: Copy {
    fn from_bits(bits: u16) -> Self;

    fn to_f32(self) -> f32;

    fn from_f32(v: f32) -> Self;

    fn to_bits(self) -> u16;
}

impl HalfFloat for half::f16 {
    fn from_bits(bits: u16) -> Self {
        half::f16::from_bits(bits)
    }

    fn to_f32(self) -> f32 {
        half::f16::to_f32(self)
    }

    fn from_f32(v: f32) -> Self {
        half::f16::from_f32(v)
    }

    fn to_bits(self) -> u16 {
        half::f16::to_bits(self)
    }
}

impl HalfFloat for half::bf16 {
    fn from_bits(bits: u16) -> Self {
        half::bf16::from_bits(bits)
    }

    fn to_f32(self) -> f32 {
        half::bf16::to_f32(self)
    }

    fn from_f32(v: f32) -> Self {
        half::bf16::from_f32(v)
    }

    fn to_bits(self) -> u16 {
        half::bf16::to_bits(self)
    }
}

impl TensorData {
    pub fn f16_from_f32(v: &[f32]) -> Self {
        TensorData::F16(to_vec_f16_from_f32::<half::f16>(v))
    }

    pub fn bf16_from_f32(v: &[f32]) -> Self {
        TensorData::Bf16(to_vec_f16_from_f32::<half::bf16>(v))
    }
}

impl Tensor {
    pub fn f16_bits(&self) -> Option<&[u16]> {
        self.data().and_then(|d| match d {
            TensorData::F16(bits) => Some(bits.as_slice()),
            _ => None,
        })
    }

    pub fn bf16_bits(&self) -> Option<&[u16]> {
        self.data().and_then(|d| match d {
            TensorData::Bf16(bits) => Some(bits.as_slice()),
            _ => None,
        })
    }

    pub fn f16_to_f32_vec(&self) -> Result<Vec<f32>, TensorError> {
        let data = self.data().ok_or(TensorError::TypeError {
            expected: ElemType::Float16,
            got: self.elem,
        })?;
        match data {
            TensorData::F16(bits) => Ok(bits
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect()),
            _ => Err(TensorError::TypeError {
                expected: ElemType::Float16,
                got: data.elem_type(),
            }),
        }
    }

    pub fn bf16_to_f32_vec(&self) -> Result<Vec<f32>, TensorError> {
        let data = self.data().ok_or(TensorError::TypeError {
            expected: ElemType::Bfloat16,
            got: self.elem,
        })?;
        match data {
            TensorData::Bf16(bits) => Ok(bits
                .iter()
                .map(|&b| half::bf16::from_bits(b).to_f32())
                .collect()),
            _ => Err(TensorError::TypeError {
                expected: ElemType::Bfloat16,
                got: data.elem_type(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    InvalidExternalData(String),
    ElementCountMismatch {
        expected: usize,
        got: usize,
    },
    TypeError {
        expected: ElemType,
        got: ElemType,
    },
    CastError {
        index: usize,
        from: ElemType,
        to: ElemType,
        reason: CastLossReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastLossReason {
    OutOfRange,
    NonIntegral,
}

impl core::fmt::Display for CastLossReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfRange => write!(f, "value out of range for target type"),
            Self::NonIntegral => write!(f, "value has a fractional part"),
        }
    }
}

impl core::fmt::Display for TensorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidExternalData(m) => write!(f, "invalid external data: {m}"),
            Self::ElementCountMismatch { expected, got } => write!(
                f,
                "tensor element count mismatch: shape implies {expected} elements, data has {got}"
            ),
            Self::TypeError { expected, got } => write!(
                f,
                "tensor type mismatch: expected {} ({expected:?}), found {} ({got:?})",
                expected.name(),
                got.name()
            ),
            Self::CastError {
                index,
                from,
                to,
                reason,
            } => write!(
                f,
                "element {index}: cannot cast {} ({from:?}) to {} ({to:?}): {reason}",
                from.name(),
                to.name()
            ),
        }
    }
}

impl std::error::Error for TensorError {}
