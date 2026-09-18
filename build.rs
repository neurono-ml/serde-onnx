use std::path::Path;

fn main() -> std::io::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let protos = root.join("third_party/onnx");
    let includes = root.join("third_party");

    println!("cargo:rerun-if-changed=third_party/onnx/onnx-ml.proto3");
    println!("cargo:rerun-if-changed=third_party/onnx/onnx-operators-ml.proto3");

    let mut config = prost_build::Config::new();
    config.protoc_executable(
        protoc_bin_vendored::protoc_bin_path().expect("vendored protoc binary available"),
    );
    config.compile_protos(
        &[
            protos.join("onnx-ml.proto3"),
            protos.join("onnx-operators-ml.proto3"),
        ],
        &[&includes],
    )?;

    Ok(())
}
