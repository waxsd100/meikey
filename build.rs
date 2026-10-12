//! feature cuda のときだけ、GPU のカーネル（src/service/gpu/kernel.cu）を C++ として CPU 向けにコンパイルする。
//! テスト（service::gpu::tests）で、GPU が無くてもカーネルの計算を Rust の実装と突き合わせるため

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=src/service/gpu/kernel.cu");
        println!("cargo:rerun-if-changed=src/service/gpu/host.cpp");
        cc::Build::new()
            .cpp(true)
            .file("src/service/gpu/host.cpp")
            .flag_if_supported("-std=c++17")
            .flag_if_supported("/std:c++17")
            .opt_level(2)
            .compile("meikey_gpu_host");
    }
}
