use proptest::prelude::*;

use serde_onnx::ir::{ElemType, Scalar, Tensor, TensorData};

fn le_roundtrip<T: Scalar + PartialEq + std::fmt::Debug>(values: Vec<T>) {
    let mut bytes = Vec::with_capacity(values.len() * T::SIZE);
    for v in &values {
        bytes.extend_from_slice(&v.to_le_bytes()[..T::SIZE]);
    }
    let mut decoded = Vec::with_capacity(values.len());
    for chunk in bytes.chunks(T::SIZE) {
        decoded.push(T::from_le_bytes(chunk).expect("decode must succeed"));
    }
    assert_eq!(decoded, values);
}

proptest! {
    #[test]
    fn le_roundtrip_f32(values in prop::collection::vec(any::<f32>(), 0..64)) {
        le_roundtrip(values);
    }

    #[test]
    fn le_roundtrip_f64(values in prop::collection::vec(any::<f64>(), 0..64)) {
        le_roundtrip(values);
    }

    #[test]
    fn le_roundtrip_i8(values in prop::collection::vec(any::<i8>(), 0..64)) {
        le_roundtrip(values);
    }

    #[test]
    fn le_roundtrip_i16(values in prop::collection::vec(any::<i16>(), 0..64)) {
        le_roundtrip(values);
    }

    #[test]
    fn le_roundtrip_i32(values in prop::collection::vec(any::<i32>(), 0..64)) {
        le_roundtrip(values);
    }

    #[test]
    fn le_roundtrip_i64(values in prop::collection::vec(any::<i64>(), 0..64)) {
        le_roundtrip(values);
    }

    #[test]
    fn le_roundtrip_u8(values in prop::collection::vec(any::<u8>(), 0..64)) {
        le_roundtrip(values);
    }

    #[test]
    fn le_roundtrip_u16(values in prop::collection::vec(any::<u16>(), 0..64)) {
        le_roundtrip(values);
    }

    #[test]
    fn le_roundtrip_u32(values in prop::collection::vec(any::<u32>(), 0..64)) {
        le_roundtrip(values);
    }

    #[test]
    fn le_roundtrip_u64(values in prop::collection::vec(any::<u64>(), 0..64)) {
        le_roundtrip(values);
    }

    #[test]
    fn le_roundtrip_bool(values in prop::collection::vec(any::<bool>(), 0..64)) {
        le_roundtrip(values);
    }

    #[test]
    fn as_slice_exact_is_lossless(values in prop::collection::vec(any::<f64>(), 0..64)) {
        let tensor = Tensor::new("t", ElemType::Double, vec![], TensorData::F64(values.clone()));
        let got = tensor.as_slice::<f64>().expect("exact access must succeed");
        assert_eq!(got, &values[..]);
    }

    #[test]
    fn to_vec_i32_i64_lossless(values in prop::collection::vec(any::<i32>(), 0..64)) {
        let tensor = Tensor::new("t", ElemType::Int32, vec![], TensorData::I32(values.clone()));
        let got = tensor.to_vec::<i64>().expect("i32 to i64 is lossless");
        assert_eq!(got, values.iter().map(|&v| v as i64).collect::<Vec<i64>>());
    }

    #[test]
    fn to_vec_u8_i64_lossless(values in prop::collection::vec(any::<u8>(), 0..64)) {
        let tensor = Tensor::new("t", ElemType::Uint8, vec![], TensorData::U8(values.clone()));
        let got = tensor.to_vec::<i64>().expect("u8 to i64 is lossless");
        assert_eq!(got, values.iter().map(|&v| v as i64).collect::<Vec<i64>>());
    }
}

mod unit {
    use super::*;

    #[test]
    fn as_slice_dtype_mismatch_fails() {
        let tensor = Tensor::new("t", ElemType::Float, vec![], TensorData::F32(vec![1.0]));
        let err = tensor.as_slice::<f64>().unwrap_err();
        assert_eq!(
            err,
            serde_onnx::ir::TensorError::TypeError {
                expected: ElemType::Double,
                got: ElemType::Float,
            }
        );
    }

    #[test]
    fn to_vec_float_to_int_rejects_fraction() {
        let tensor = Tensor::new("t", ElemType::Float, vec![], TensorData::F32(vec![1.5]));
        let err = tensor.to_vec::<i64>().expect_err("must reject fraction");
        match err {
            serde_onnx::ir::TensorError::CastError {
                index,
                from,
                to,
                reason,
            } => {
                assert_eq!(index, 0);
                assert_eq!(from, ElemType::Float);
                assert_eq!(to, ElemType::Int64);
                assert_eq!(reason, serde_onnx::ir::CastLossReason::NonIntegral);
            }
            other => panic!("expected CastError, got {other:?}"),
        }
    }

    #[test]
    fn to_vec_float_to_int_rejects_out_of_range() {
        let tensor = Tensor::new(
            "t",
            ElemType::Float,
            vec![],
            TensorData::F32(vec![f32::INFINITY]),
        );
        let err = tensor.to_vec::<i32>().expect_err("must reject inf");
        match err {
            serde_onnx::ir::TensorError::CastError { index, reason, .. } => {
                assert_eq!(index, 0);
                assert_eq!(reason, serde_onnx::ir::CastLossReason::OutOfRange);
            }
            other => panic!("expected CastError, got {other:?}"),
        }
    }

    #[test]
    fn to_vec_f64_to_u64_large_fails() {
        let tensor = Tensor::new(
            "t",
            ElemType::Double,
            vec![],
            TensorData::F64(vec![2f64.powi(64)]),
        );
        assert!(tensor.to_vec::<u64>().is_err());
    }

    #[test]
    fn to_vec_f64_to_f32_rejects_overflow() {
        let tensor = Tensor::new("t", ElemType::Double, vec![], TensorData::F64(vec![1e300]));
        let err = tensor
            .to_vec::<f32>()
            .expect_err("must reject f64 -> f32 overflow");
        match err {
            serde_onnx::ir::TensorError::CastError { reason, .. } => {
                assert_eq!(reason, serde_onnx::ir::CastLossReason::OutOfRange);
            }
            other => panic!("expected CastError, got {other:?}"),
        }
    }

    #[test]
    fn to_vec_string_always_fails() {
        let tensor = Tensor::new(
            "t",
            ElemType::String,
            vec![],
            TensorData::String(vec!["x".to_string()]),
        );
        assert!(matches!(
            tensor.to_vec::<f32>(),
            Err(serde_onnx::ir::TensorError::TypeError { .. })
        ));
    }

    #[test]
    fn f16_mismatch_fails() {
        let tensor = Tensor::new("t", ElemType::Float16, vec![], TensorData::F16(vec![0]));
        assert!(tensor.bf16_to_f32_vec().is_err());
        assert!(tensor.f16_to_f32_vec().is_ok());
    }

    #[test]
    fn to_vec_bool_exact() {
        let tensor = Tensor::new(
            "t",
            ElemType::Bool,
            vec![],
            TensorData::Bool(vec![true, false]),
        );
        assert_eq!(tensor.as_slice::<bool>().unwrap(), &[true, false]);
    }

    #[test]
    fn f16_bits_roundtrip_f32() {
        let vals = vec![0.0f32, 1.0, -2.5, 65504.0];
        let tensor = Tensor::new(
            "t",
            ElemType::Float16,
            vec![],
            TensorData::f16_from_f32(&vals),
        );
        assert_eq!(tensor.f16_bits().unwrap().len(), 4);
        assert_eq!(
            tensor.f16_to_f32_vec().unwrap(),
            vec![0.0, 1.0, -2.5, 65504.0]
        );
    }

    #[test]
    fn bf16_bits_roundtrip_f32() {
        let vals = vec![0.0f32, 1.0, -2.5, 3.0];
        let tensor = Tensor::new(
            "t",
            ElemType::Bfloat16,
            vec![],
            TensorData::bf16_from_f32(&vals),
        );
        assert_eq!(tensor.bf16_bits().unwrap().len(), 4);
        assert_eq!(tensor.bf16_to_f32_vec().unwrap(), vec![0.0, 1.0, -2.5, 3.0]);
    }
}
