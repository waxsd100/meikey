//! feature cuda のとき、GPU のカーネル（src/service/gpu/kernel.cu）を C++ として CPU 向けにもコンパイルする。
//! テスト（service::gpu::tests）で、GPU が無くてもカーネルの計算を Rust の実装と突き合わせるため。
//! 埋め込む PTX の組（service::gpu::BUILDS）ごとに、まとめる鍵の数・体演算の種類と
//! テスト用の関数名の接頭辞を変えてコンパイルし、その関数を並べた Rust のコード（OUT_DIR/gpu_hosts.rs）も作る。
//! C++ コンパイラが無ければコンパイルせず、そのテストだけ飛ばす（cfg meikey_gpu_host が立たない）

/// (まとめる鍵の数, 体演算の種類（service::gpu::Limbs の名前）, MEIKEY_FIELD, MEIKEY_UNPACKED, --gpu-limbs の値)。
/// service::gpu::BUILDS と同じ並び（テストで確かめる）。先頭は既定の組で、テスト用の関数名の接頭辞は
/// meikey_test_（ほかは meikey_b<数>_l<値>_test_）
#[cfg(feature = "cuda")]
const VARIANTS: [(u32, &str, u32, u32, &str); 12] = [
    (32, "R25", 10, 0, "10"),
    (32, "R25Table", 10, 1, "10u"),
    (32, "R32", 8, 0, "8"),
    (32, "K26", 26, 0, "10k"),
    (32, "K26Table", 26, 1, "10ku"),
    (32, "F51", 51, 0, "5f"),
    (16, "R25", 10, 0, "10"),
    (16, "R25Table", 10, 1, "10u"),
    (16, "R32", 8, 0, "8"),
    (16, "K26", 26, 0, "10k"),
    (16, "K26Table", 26, 1, "10ku"),
    (16, "F51", 51, 0, "5f"),
];

#[cfg(feature = "cuda")]
fn prefix(i: usize, (batch, _, _, _, code): (u32, &str, u32, u32, &str)) -> String {
    match i {
        0 => "meikey_test_".into(),
        _ => format!("meikey_b{batch}_l{code}_test_"),
    }
}

/// テストが使う、組ごとの関数の宣言と、それを並べた HOSTS
#[cfg(feature = "cuda")]
fn hosts_rs() -> String {
    let mut decls = String::new();
    let mut entries = String::new();
    for (i, v) in VARIANTS.into_iter().enumerate() {
        let p = prefix(i, v);
        decls += &format!(
            "    fn {p}derive_batch(table: *const u32, width: u32, key: *const u32, counter: u32, first: u64, \
             out: *mut u8);\n\
             \x20   fn {p}search(nthreads: u64, table: *const u32, width: u32, key: *const u32, counter: u32, \
             needle: *const u8, needle_len: u32, fold: u32, mode: u32, base: u64, iters: u32, stop_on_hit: u32, \
             result: *mut u64);\n\
             \x20   fn {p}fe_op_words(op: u32, a: *const u32, b: *const u32, out: *mut u8);\n\
             \x20   fn {p}invert_pooled(words: *const u32, n: u32, out: *mut u8);\n\
             \x20   fn {p}batch() -> u32;\n\
             \x20   fn {p}field() -> u32;\n\
             \x20   fn {p}unpacked() -> u32;\n\
             \x20   fn {p}entry_words() -> u32;\n"
        );
        let (batch, limbs, _, _, _) = v;
        entries += &format!(
            "    HostKernel {{\n\
             \x20       build: Build {{ batch: {batch}, limbs: Limbs::{limbs} }},\n\
             \x20       derive: {p}derive_batch,\n\
             \x20       search: {p}search,\n\
             \x20       fe_op: {p}fe_op_words,\n\
             \x20       invert_pooled: {p}invert_pooled,\n\
             \x20       batch: {p}batch,\n\
             \x20       field: {p}field,\n\
             \x20       unpacked: {p}unpacked,\n\
             \x20       entry_words: {p}entry_words,\n\
             \x20   }},\n"
        );
    }
    format!(
        "// build.rs が作る。組ごとに CPU 向けにコンパイルしたカーネルの関数\n\
         unsafe extern \"C\" {{\n{decls}}}\n\n\
         /// [`BUILDS`] と同じ並び（builds_match_kernel で確かめる）\n\
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
            let (batch, _, field, unpacked, code) = v;
            let built = cc::Build::new()
                .cpp(true)
                .file("src/service/gpu/host.cpp")
                .define("MEIKEY_BATCH", batch.to_string().as_str())
                .define("MEIKEY_FIELD", field.to_string().as_str())
                .define("MEIKEY_UNPACKED", unpacked.to_string().as_str())
                .define("MEIKEY_TEST_PREFIX", prefix(i, v).as_str())
                .flag_if_supported("-std=c++17")
                .flag_if_supported("/std:c++17")
                // MSVC は既定でソースを OS の文字コード（日本語の Windows なら CP932）として読み、
                // 日本語のコメントの後ろの改行が食われて次の行までコメントになる
                .flag_if_supported("/utf-8")
                // 倍精度の体演算（MEIKEY_FIELD 51）の a * b + c を FMA にまとめない（結果が変わる）
                .flag_if_supported("-ffp-contract=off")
                .flag_if_supported("/fp:precise")
                .opt_level(2)
                .cargo_warnings(false)
                .try_compile(&format!("meikey_gpu_host_b{batch}_l{code}"));
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
