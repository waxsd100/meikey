//! 埋め込む GPU カーネルの PTX（src/service/gpu/ptx/ の、まとめる鍵の数と体演算の種類の組ごとのファイル）を kernel.cu から作る（`cargo make ptx`）。
//!
//! PTX の版はコンパイルした NVRTC の版で決まり、それより古いドライバでは読めない。なるべく多くのドライバで
//! 動くよう、古い NVRTC（CUDA 12.0）で作る。例: `pip install --target /tmp/nvrtc "nvidia-cuda-nvrtc-cu12==12.0.*"`
//! として、`LD_LIBRARY_PATH=/tmp/nvrtc/nvidia/cuda_nvrtc/lib cargo make ptx`

use meikey::service::gpu::{BUILDS, GpuError, portable_ptx};

/// PTX を作れない・書けないときのエラー
#[derive(Debug, thiserror::Error)]
enum GenError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error("{path}: {source}")]
    Write {
        path: String,
        source: std::io::Error,
    },
}

fn main() -> Result<(), GenError> {
    for build in BUILDS {
        let ptx = portable_ptx(build)?;
        let path = format!(
            "{}/src/service/gpu/ptx/{}",
            env!("CARGO_MANIFEST_DIR"),
            build.ptx_file()
        );
        std::fs::write(&path, ptx).map_err(|source| GenError::Write {
            path: path.clone(),
            source,
        })?;
        println!("{path}");
    }
    Ok(())
}
