use super::tensor::TensorData;
use crate::ir::types::ElemType;

pub trait Sealed {}

pub trait Scalar: Copy + Sealed {
    const ELEM_TYPE: ElemType;

    const SIZE: usize;

    fn to_le_bytes(self) -> Vec<u8>;

    fn from_le_bytes(bytes: &[u8]) -> Option<Self>;

    fn into_data(values: Vec<Self>) -> TensorData;
}

mod sealed {
    use super::Sealed;

    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for bool {}
}

impl Scalar for f32 {
    const ELEM_TYPE: ElemType = ElemType::Float;

    const SIZE: usize = 4;

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
        Some(f32::from_le_bytes(bytes.try_into().ok()?))
    }

    fn into_data(values: Vec<Self>) -> TensorData {
        TensorData::F32(values)
    }
}

impl Scalar for f64 {
    const ELEM_TYPE: ElemType = ElemType::Double;

    const SIZE: usize = 8;

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
        Some(f64::from_le_bytes(bytes.try_into().ok()?))
    }

    fn into_data(values: Vec<Self>) -> TensorData {
        TensorData::F64(values)
    }
}

impl Scalar for i8 {
    const ELEM_TYPE: ElemType = ElemType::Int8;

    const SIZE: usize = 1;

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
        Some(i8::from_le_bytes(bytes.try_into().ok()?))
    }

    fn into_data(values: Vec<Self>) -> TensorData {
        TensorData::I8(values)
    }
}

impl Scalar for i16 {
    const ELEM_TYPE: ElemType = ElemType::Int16;

    const SIZE: usize = 2;

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
        Some(i16::from_le_bytes(bytes.try_into().ok()?))
    }

    fn into_data(values: Vec<Self>) -> TensorData {
        TensorData::I16(values)
    }
}

impl Scalar for i32 {
    const ELEM_TYPE: ElemType = ElemType::Int32;

    const SIZE: usize = 4;

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
        Some(i32::from_le_bytes(bytes.try_into().ok()?))
    }

    fn into_data(values: Vec<Self>) -> TensorData {
        TensorData::I32(values)
    }
}

impl Scalar for i64 {
    const ELEM_TYPE: ElemType = ElemType::Int64;

    const SIZE: usize = 8;

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
        Some(i64::from_le_bytes(bytes.try_into().ok()?))
    }

    fn into_data(values: Vec<Self>) -> TensorData {
        TensorData::I64(values)
    }
}

impl Scalar for u8 {
    const ELEM_TYPE: ElemType = ElemType::Uint8;

    const SIZE: usize = 1;

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
        Some(u8::from_le_bytes(bytes.try_into().ok()?))
    }

    fn into_data(values: Vec<Self>) -> TensorData {
        TensorData::U8(values)
    }
}

impl Scalar for u16 {
    const ELEM_TYPE: ElemType = ElemType::Uint16;

    const SIZE: usize = 2;

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
        Some(u16::from_le_bytes(bytes.try_into().ok()?))
    }

    fn into_data(values: Vec<Self>) -> TensorData {
        TensorData::U16(values)
    }
}

impl Scalar for u32 {
    const ELEM_TYPE: ElemType = ElemType::Uint32;

    const SIZE: usize = 4;

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    }

    fn into_data(values: Vec<Self>) -> TensorData {
        TensorData::U32(values)
    }
}

impl Scalar for u64 {
    const ELEM_TYPE: ElemType = ElemType::Uint64;

    const SIZE: usize = 8;

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
        Some(u64::from_le_bytes(bytes.try_into().ok()?))
    }

    fn into_data(values: Vec<Self>) -> TensorData {
        TensorData::U64(values)
    }
}

impl Scalar for bool {
    const ELEM_TYPE: ElemType = ElemType::Bool;

    const SIZE: usize = 1;

    fn to_le_bytes(self) -> Vec<u8> {
        vec![u8::from(self)]
    }

    fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
        Some(bytes.first()? != &0)
    }

    fn into_data(values: Vec<Self>) -> TensorData {
        TensorData::Bool(values)
    }
}
