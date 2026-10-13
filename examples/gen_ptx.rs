//! 埋め込む GPU カーネルの PTX（src/service/gpu/ptx/ の設定ごとのファイル）を kernel.cu から作る（`cargo make ptx`）。
//!
//! PTX の版はコンパイルした NVRTC の版で決まり、それより古いドライバでは読めない。なるべく多くのドライバで
//! 動くよう、古い NVRTC（CUDA 12.0）で作る。例: `pip install --target /tmp/nvrtc "nvidia-cuda-nvrtc-cu12==12.0.*"`
//! として、`LD_LIBRARY_PATH=/tmp/nvrtc/nvidia/cuda_nvrtc/lib cargo make ptx`

use meikey::service::gpu::{EMBEDDED, portable_ptx};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for options in EMBEDDED {
        let ptx = portable_ptx(options)?;
        let path = format!(
            "{}/src/service/gpu/ptx/{}",
            env!("CARGO_MANIFEST_DIR"),
            options.ptx_file()
        );
        std::fs::write(&path, ptx)?;
        println!("{path}");
    }
    Ok(())
}
