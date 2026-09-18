# Vendored ONNX proto files

Source: https://github.com/onnx/onnx
Pinned upstream commit: `bca0315ff3e56bb5847509ca32fadb8d87f680e1` (tag `v1.20.0`)

## Vendored files

| File | Upstream path | SHA-256 |
|------|---------------|---------|
| onnx.proto3 | `onnx/onnx.proto3` | 470e64dfc5338477d3adc1853f5875618f70cc698306d9bed1232305680b121f |
| onnx-operators.proto3 | `onnx/onnx-operators.proto3` | 0dc840c63fc2deeab089a0ce94381f82be80c31574eb597a56ba92f999c19c39 |
| onnx-ml.proto3 | `onnx/onnx-ml.proto3` | 8790fa95816474026168a7e2a657007540ec36f758ff048ff9f0a34ed0e449e6 |
| onnx-operators-ml.proto3 | `onnx/onnx-operators-ml.proto3` | 573c423722059d0dd64e48c2d8d2f99901c2d8d7c1bf2ff3fc5b0af2065fa231 |

## Changelog

- 2026-09-17: vendored at upstream tag `v1.20.0` (commit `bca0315ff3e56bb5847509ca32fadb8d87f680e1`).
  This is the latest upstream release whose generated proto files carry the full
  Apache-2.0 license header text. `v1.21.0` truncated the header in generated files
  to `SPDX-License-Identifier` only; `v1.22.0` is current but does not restore the
  full header. Update by replacing these four files and refreshing the pinned
  commit, checksums and date above (manual step, never automated).

## Compilation pair

`onnx.proto3` + `onnx-operators.proto3` and `onnx-ml.proto3` +
`onnx-operators-ml.proto3` define overlapping messages (`ModelProto`,
`GraphProto`, `NodeProto`, ...). Only one pair is passed to `prost-build` at a
time; this project compiles the **ML pair** (`onnx-ml.proto3` +
`onnx-operators-ml.proto3`), which is a superset covering `ai.onnx.ml`
definitions. The non-ML pair stays vendored for provenance/diffing when
updating, and so consumers of this file set can see the upstream layout.

## ElemType constants target

Pinned proto defines `TensorProto.DataType` with values `UINT2 = 25` and
`INT2 = 26` (2-bit ints, IR version 13 in the same file). `v1.21`/`v1.22` do not
add further data types, so the constants set is identical across
v1.20.0..v1.22.0; the pin at v1.20.0 targets constants up to and including
2-bit types.
