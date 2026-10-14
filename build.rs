//! feature cuda のとき、GPU のカーネル（src/service/gpu/kernel.cu）を C++ として CPU 向けにもコンパイルする。
//! テスト（service::gpu::tests）で、GPU が無くてもカーネルの計算を Rust の実装と突き合わせるため。
//! 埋め込む PTX の設定（service::gpu::EMBEDDED）ごとに、表の幅・まとめる鍵の数・体演算のリムの数と
//! テスト用の関数名の接頭辞を変えてコンパイルし、その関数を並べた Rust のコード（OUT_DIR/gpu_hosts.rs）も作る。
//! C++ コンパイラが無ければコンパイルせず、そのテストだけ飛ばす（cfg meikey_gpu_host が立たない）

/// (表の幅, まとめる鍵の数, リムの数)。service::gpu::EMBEDDED と同じ並び（テストで確かめる）。
/// 先頭は既定の設定で、テスト用の関数名の接頭辞は meikey_test_（ほかは meikey_w<幅>_b<数>_l<リム>_test_）
#[cfg(feature = "cuda")]
const VARIANTS: [(u32, u32, u32); 20] = [
    (12, 32, 10),
    (12, 32, 8),
    (12, 64, 10),
    (12, 64, 8),
    (14, 32, 10),
    (14, 32, 8),
    (14, 64, 10),
    (14, 64, 8),
    (16, 32, 10),
    (16, 32, 8),
    (16, 64, 10),
    (16, 64, 8),
    (18, 32, 10),
    (18, 32, 8),
    (18, 64, 10),
    (18, 64, 8),
    (20, 32, 10),
    (20, 32, 8),
    (20, 64, 10),
    (20, 64, 8),
];

#[cfg(feature = "cuda")]
fn prefix(i: usize, (width, batch, limbs): (u32, u32, u32)) -> String {
    match i {
        0 => "meikey_test_".into(),
        _ => format!("meikey_w{width}_b{batch}_l{limbs}_test_"),
    }
}

/// テストが使う、設定ごとの関数の宣言と、それを並べた HOSTS
#[cfg(feature = "cuda")]
fn hosts_rs() -> String {
    let mut decls = String::new();
    let mut entries = String::new();
    for (i, v) in VARIANTS.into_iter().enumerate() {
        let p = prefix(i, v);
        decls += &format!(
            "    fn {p}derive_batch(table: *const u32, key: *const u32, counter: u32, first: u64, out: *mut u8);\n\
             \x20   fn {p}search(nthreads: u64, table: *const u32, key: *const u32, counter: u32, needle: *const u8, \
             needle_len: u32, fold: u32, mode: u32, base: u64, iters: u32, stop_on_hit: u32, result: *mut u64);\n\
             \x20   fn {p}fe_op_words(op: u32, a: *const u32, b: *const u32, out: *mut u8);\n\
             \x20   fn {p}batch() -> u32;\n\
             \x20   fn {p}width() -> u32;\n\
             \x20   fn {p}limbs() -> u32;\n"
        );
        let (width, batch, limbs) = v;
        entries += &format!(
            "    HostKernel {{\n\
             \x20       options: GpuOptions::new({width}, {batch}, {limbs}),\n\
             \x20       derive: {p}derive_batch,\n\
             \x20       search: {p}search,\n\
             \x20       fe_op: {p}fe_op_words,\n\
             \x20       batch: {p}batch,\n\
             \x20       width: {p}width,\n\
             \x20       limbs: {p}limbs,\n\
             \x20   }},\n"
        );
    }
    format!(
        "// build.rs が作る。埋め込む設定ごとに CPU 向けにコンパイルしたカーネルの関数\n\
         unsafe extern \"C\" {{\n{decls}}}\n\n\
         /// [`EMBEDDED`] と同じ並び（options_match_kernel で確かめる）\n\
         static HOSTS: [HostKernel; {}] = [\n{entries}];\n",
        VARIANTS.len()
    )
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(meikey_gpu_host)");
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=src/service/gpu/kernel.cu");
        println!("cargo:rerun-if-changed=src/service/gpu/host.cpp");
        let mut all = true;
        for (i, v) in VARIANTS.into_iter().enumerate() {
            let (width, batch, limbs) = v;
            let built = cc::Build::new()
                .cpp(true)
                .file("src/service/gpu/host.cpp")
                .define("MEIKEY_W", width.to_string().as_str())
                .define("MEIKEY_BATCH", batch.to_string().as_str())
                .define("MEIKEY_LIMBS", limbs.to_string().as_str())
                .define("MEIKEY_TEST_PREFIX", prefix(i, v).as_str())
                .flag_if_supported("-std=c++17")
                .flag_if_supported("/std:c++17")
                // MSVC は既定でソースを OS の文字コード（日本語の Windows なら CP932）として読み、
                // 日本語のコメントの後ろの改行が食われて次の行までコメントになる
                .flag_if_supported("/utf-8")
                .opt_level(2)
                .cargo_warnings(false)
                .try_compile(&format!("meikey_gpu_host_w{width}_b{batch}_l{limbs}"));
            if let Err(e) = built {
                println!(
                    "cargo:warning=C++ コンパイラが使えないので、GPU カーネルを CPU で確かめるテストを飛ばす（{e}）"
                );
                all = false;
                break;
            }
        }
        if all {
            let dir = std::env::var("OUT_DIR").expect("cargo が OUT_DIR を渡す");
            let out = std::path::Path::new(&dir).join("gpu_hosts.rs");
            std::fs::write(out, hosts_rs()).expect("OUT_DIR に書ける");
            println!("cargo:rustc-cfg=meikey_gpu_host");
        }
    }
}
