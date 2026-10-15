"""Google Colab の GPU の VM で、meikey の GPU のテストを回す。

GitHub Actions（.github/workflows/colab.yml）が `colab run` でこのファイルを VM に送って実行する。
環境変数 MEIKEY_SHA のコミットを取ってきて、`cargo nextest run --profile ci` で GPU のテストを回し、
その終了コードで終わる（結果は /content/meikey/target/nextest/ci/junit.xml）。
"""

import glob
import os
import subprocess
import sys

REPO = os.environ.get("MEIKEY_REPO", "https://github.com/waxsd100/meikey")
SHA = os.environ["MEIKEY_SHA"]
FILTER = os.environ.get("MEIKEY_FILTER", "test(/^service::gpu::/)")
DIR = "/content/meikey"
CARGO_BIN = os.path.expanduser("~/.cargo/bin")

env = dict(os.environ, CARGO_TERM_COLOR="always")
env["PATH"] = CARGO_BIN + ":" + env["PATH"]


def sh(cmd, cwd=None, check=True):
    """コマンドの出力を 1 行ずつそのまま出す（出力が続くので、colab run が待ちくたびれない）"""
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end="", flush=True)
    code = p.wait()
    if check and code != 0:
        print(f"失敗（終了コード {code}）: {cmd}", flush=True)
        sys.exit(code)
    return code


def use_nvrtc():
    """libnvrtc を探して LD_LIBRARY_PATH に足す（colab/meikey.ipynb と同じ。NVRTC のテストで使う）"""
    candidates = sorted(
        p for d in ["/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib", "/usr/lib/x86_64-linux-gnu"]
                  + glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/cuda_nvrtc/lib")
                  + glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/cu*/lib")
        for p in glob.glob(d + "/libnvrtc.so*")
        if "builtins" not in p and ".alt." not in p
    )
    if not candidates:
        print("libnvrtc が見つからない", flush=True)
        return
    nvrtc = candidates[-1]
    libdir = "/content/meikey-libs"
    os.makedirs(libdir, exist_ok=True)
    link = os.path.join(libdir, "libnvrtc.so")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(nvrtc, link)
    print(f"libnvrtc: {nvrtc}", flush=True)
    paths = [libdir, os.path.dirname(nvrtc)]
    old = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(paths + [p for p in old.split(":") if p and p not in paths])


sh("nvidia-smi")

if not os.path.exists(os.path.join(CARGO_BIN, "cargo")):
    sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs"
       " | sh -s -- -y --profile minimal --default-toolchain none")
if not os.path.exists(os.path.join(CARGO_BIN, "cargo-nextest")):
    sh(f"curl -LsSf https://get.nexte.st/latest/linux | tar zxf - -C {CARGO_BIN}")

# GitHub Actions で動いているコミットを取ってくる
sh(f"rm -rf {DIR} && git init -q {DIR}")
sh(f"git fetch -q --depth 1 {REPO} {SHA} && git checkout -q FETCH_HEAD", cwd=DIR)
sh("git log -1 --oneline", cwd=DIR)
# rust-toolchain.toml の版を入れる
sh("rustup show", cwd=DIR)

use_nvrtc()
# GPU か NVRTC が見つからなければ、テストを飛ばさずに失敗させる
env["MEIKEY_REQUIRE_SIMD"] = "cuda,nvrtc"
code = sh(f"cargo nextest run --profile ci --release -E '{FILTER}'", cwd=DIR, check=False)
sys.exit(code)
