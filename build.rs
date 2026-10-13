//! feature cuda のとき、GPU のカーネル（src/service/gpu/kernel.cu）を C++ として CPU 向けにもコンパイルする。
//! テスト（service::gpu::tests）で、GPU が無くてもカーネルの計算を Rust の実装と突き合わせるため。
//! 埋め込む PTX の設定（service::gpu::EMBEDDED）ごとに、表の幅・まとめる鍵の数とテスト用の関数名の接頭辞を
//! 変えてコンパイルする。C++ コンパイラが無ければコンパイルせず、そのテストだけ飛ばす（cfg meikey_gpu_host が立たない）

/// (表の幅, まとめる鍵の数, テスト用の関数名の接頭辞)。service::gpu::EMBEDDED と同じ並び（テストで確かめる）
#[cfg(feature = "cuda")]
const VARIANTS: [(u32, u32, &str); 6] = [
    (12, 16, "meikey_test_"),
    (12, 32, "meikey_w12_b32_test_"),
    (14, 16, "meikey_w14_b16_test_"),
    (14, 32, "meikey_w14_b32_test_"),
    (16, 16, "meikey_w16_b16_test_"),
    (16, 32, "meikey_w16_b32_test_"),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(meikey_gpu_host)");
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=src/service/gpu/kernel.cu");
        println!("cargo:rerun-if-changed=src/service/gpu/host.cpp");
        let mut all = true;
        for (width, batch, prefix) in VARIANTS {
            let built = cc::Build::new()
                .cpp(true)
                .file("src/service/gpu/host.cpp")
                .define("MEIKEY_W", width.to_string().as_str())
                .define("MEIKEY_BATCH", batch.to_string().as_str())
                .define("MEIKEY_TEST_PREFIX", prefix)
                .flag_if_supported("-std=c++17")
                .flag_if_supported("/std:c++17")
                .opt_level(2)
                .cargo_warnings(false)
                .try_compile(&format!("meikey_gpu_host_w{width}_b{batch}"));
            if let Err(e) = built {
                println!(
                    "cargo:warning=C++ コンパイラが使えないので、GPU カーネルを CPU で確かめるテストを飛ばす（{e}）"
                );
                all = false;
                break;
            }
        }
        if all {
            println!("cargo:rustc-cfg=meikey_gpu_host");
        }
    }
}
