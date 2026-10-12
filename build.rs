//! feature cuda のとき、GPU のカーネル（src/service/gpu/kernel.cu）を C++ として CPU 向けにもコンパイルする。
//! テスト（service::gpu::tests）で、GPU が無くてもカーネルの計算を Rust の実装と突き合わせるため。
//! C++ コンパイラが無ければコンパイルせず、そのテストだけ飛ばす（cfg meikey_gpu_host が立たない）

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(meikey_gpu_host)");
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=src/service/gpu/kernel.cu");
        println!("cargo:rerun-if-changed=src/service/gpu/host.cpp");
        let built = cc::Build::new()
            .cpp(true)
            .file("src/service/gpu/host.cpp")
            .flag_if_supported("-std=c++17")
            .flag_if_supported("/std:c++17")
            .opt_level(2)
            .cargo_warnings(false)
            .try_compile("meikey_gpu_host");
        match built {
            Ok(()) => println!("cargo:rustc-cfg=meikey_gpu_host"),
            Err(e) => println!(
                "cargo:warning=C++ コンパイラが使えないので、GPU カーネルを CPU で確かめるテストを飛ばす（{e}）"
            ),
        }
    }
}
