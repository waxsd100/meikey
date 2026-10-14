"""meikey と、同じように ed25519 の SSH 公開鍵に好きな文字列を入れるツールを、同じマシンでまとめて測る。

colab/compare.ipynb から使う（手元の Linux でも動く。作業ディレクトリは環境変数 COMPARE_DIR、既定は /content/compare）。

測り方は2通り:
- 速度を表示するツールは、見つからない文字列（NEVER）で決めた秒数だけ動かし、最後に表示した速度を読む
- 速度を表示しないツールは、短い文字列で何度も探させ、見つかった数・その文字列が当たる確率・かかった時間から
  速度を出す（1回の起動にかかる時間は、すぐ見つかる 1 文字で測って差し引く）
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # このファイルがある meikey のリポジトリ
WORK = os.environ.get("COMPARE_DIR", "/content/compare")
SRC = os.path.join(WORK, "src")
BIN = os.path.join(WORK, "bin")
LOGS = os.path.join(WORK, "logs")
NPROC = os.cpu_count() or 1

# 見つからない文字列（10 文字）。後始末で、この文字列を引数に持つプロセスを止める
NEVER = "zyxwvutsrq"
# 見つけて数えるときの文字列は、これの先頭 k 文字（全部違う文字で、A〜P を含まない）
LETTERS = NEVER


# ---------------------------------------------------------------------------
# コマンドの実行
# ---------------------------------------------------------------------------


def env() -> dict:
    e = dict(os.environ)
    paths = [BIN, os.path.join(WORK, "cargo", "bin"), os.path.expanduser("~/.cargo/bin"),
             os.path.join(WORK, "go", "bin"), "/usr/local/cuda/bin"]
    e["PATH"] = ":".join(paths + [e.get("PATH", "")])
    e["GOBIN"] = BIN
    e["GOPATH"] = os.path.join(WORK, "gopath")
    e["RUSTFLAGS"] = "-C target-cpu=native"
    # meikey の表示を英語にする（速度の行を読むため）
    e["LC_ALL"] = e["LANG"] = "C.UTF-8"
    return e


def run(cmd: str, cwd: Optional[str] = None, timeout: Optional[float] = None,
        tty: bool = False) -> tuple[int, str, float]:
    """cmd を bash で動かし、(終了コード, 出力, かかった秒数) を返す。timeout 秒で SIGINT を送り（速度の表示を
    最後まで出させる）、それでも終わらなければ止める。tty なら疑似端末の中で動かす（端末でないと速度を表示しない
    ツールのため）"""
    if tty:
        cmd = f"script -qfec {shlex.quote(cmd)} /dev/null"
    start = time.monotonic()
    p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, env=env(), stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _signal(p, signal.SIGINT)
        try:
            out, _ = p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            _signal(p, signal.SIGKILL)
            out, _ = p.communicate()
    elapsed = time.monotonic() - start
    _signal(p, signal.SIGKILL)  # 残った子プロセスも止める
    return p.returncode, out.decode("utf-8", "replace"), elapsed


def _signal(p: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(p.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def cleanup() -> None:
    """測ったツールのプロセスが残っていれば止める（疑似端末の中のものは別のプロセスグループになるため）"""
    subprocess.run(["pkill", "-9", "-f", LETTERS[:3]], capture_output=True)


ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[()][0-9A-Za-z]|\x1b[=>]")


def plain(out: str) -> str:
    """色や行の上書き（\r）を除いた出力"""
    return ANSI.sub("", out).replace("\r", "\n")


# ---------------------------------------------------------------------------
# ツールのビルド
# ---------------------------------------------------------------------------


def clone(url: str, name: str) -> str:
    return f"rm -rf {name} && git clone -q --depth 1 {url} {name} && cd {name}"


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and run("nvidia-smi -L", timeout=30)[0] == 0


def has_nvcc() -> bool:
    return run("command -v nvcc", timeout=10)[0] == 0


def has_cuda() -> bool:
    """GPU と nvcc の両方がある（-arch=native や、GPU の世代を調べてビルドするため）"""
    return has_gpu() and has_nvcc()


def has_opencl_gpu() -> bool:
    code, out, _ = run("clinfo -l", timeout=60)
    return code == 0 and "NVIDIA" in out


def gpu_cc() -> str:
    """GPU の compute capability（例: 12.0 なら "120"）。-arch=native を使えない Makefile に渡す"""
    out = run("nvidia-smi --query-gpu=compute_cap --format=csv,noheader", timeout=30)[1]
    return out.strip().splitlines()[0].replace(".", "") if out.strip() else "75"


def solanity(url: str, name: str) -> str:
    """solanity をもとにしたツール（ed25519-gpu-vanity）のビルド。GPU の世代は Makefile の既定（sm_50〜sm_75）では
    なく、この GPU の世代にする"""
    return (clone(url, name)
            + " && make V=release GPU_ARCHS=sm_{cc} GPU_PTX_ARCH=compute_{cc}"
            + " && test -x src/release/cuda_ed25519_vanity")


@dataclass
class Build:
    name: str
    url: str
    commands: str
    needs: Callable[[], bool] = lambda: True
    needs_text: str = ""


BUILDS = [
    Build("meikey", "https://github.com/waxsd100/meikey",
          f"cd {shlex.quote(REPO)} && cargo build --release"),
    Build("vanityssh-go", "https://github.com/danielewood/vanityssh-go",
          "go install github.com/danielewood/vanityssh-go@latest"),
    Build("vanity-ssh-keygen", "https://github.com/DumAdudus/vanity-ssh-keygen",
          clone("https://github.com/DumAdudus/vanity-ssh-keygen", "vanity-ssh-keygen")
          # crypto/ed25519 の内部の関数を go:linkname で呼んでいるので、新しい Go ではリンクの検査を外す
          + " && go build -ldflags=-checklinkname=0 -o vanity-ssh-keygen ./cmd/vanity-ssh-keygen"),
    Build("vanity-ssh-rs", "https://github.com/mogottsch/vanity-ssh-rs",
          f"cargo install --locked --root {WORK}/cargo vanity-ssh-rs"),
    Build("vanissh", "https://crates.io/crates/vanissh",
          f"cargo install --locked --root {WORK}/cargo vanissh"),
    Build("ssh-keygen-ed25519-vanity", "https://github.com/ionathanch/ssh-keygen-ed25519-vanity",
          clone("https://github.com/ionathanch/ssh-keygen-ed25519-vanity", "ssh-keygen-ed25519-vanity")
          + " && make && cargo build --release"),
    Build("ed25519brute", "https://github.com/akiym/ed25519brute",
          "go install github.com/akiym/ed25519brute@latest"),
    Build("SSH-Ed25519-VanityGen", "https://github.com/Aminuxer/SSH-Ed25519-VanityGen",
          clone("https://github.com/Aminuxer/SSH-Ed25519-VanityGen", "SSH-Ed25519-VanityGen")
          + " && pip install -q cryptography psutil && (command -v nvidia-smi >/dev/null && pip install -q pyopencl || true)"),
    Build("VaniSSH", "https://github.com/scottvr/vanissh",
          clone("https://github.com/scottvr/vanissh", "scottvr-vanissh") + " && pip install -q -r requirements.txt"),
    Build("vanity-key-ssh", "https://github.com/irdbl/vanity-key-ssh",
          clone("https://github.com/irdbl/vanity-key-ssh", "vanity-key-ssh")
          + " && python3 tools/gen_table.py table.bin && make host"
          + " && (command -v nvcc >/dev/null && python3 tools/gen_table.py table16.bin --wide"
          + " && make gpu GPU_ARCH=-arch=native || true)"),
    Build("vanity-keygen", "https://github.com/pcarrier/vanity-keygen",
          clone("https://github.com/pcarrier/vanity-keygen", "vanity-keygen") + " && cargo build --release",
          has_opencl_gpu, "OpenCL で使える NVIDIA の GPU"),
    Build("ed25519brute-cuda", "https://github.com/4equest/ed25519brute-cuda",
          clone("https://github.com/4equest/ed25519brute-cuda", "ed25519brute-cuda")
          # build.bat と同じ指定（-rdc=true が無いと、extern で宣言した __device__ の変数を定義したところで
          # ホストのコンパイラがエラーにする）。世代は -arch=native
          + " && nvcc -O3 --use_fast_math --extra-device-vectorization -arch=native -rdc=true -lcudadevrt -t 0"
          + " -o ed25519brute_cuda src/main.cu -I src",
          has_cuda, "GPU と nvcc（CUDA Toolkit）"),
    Build("ed25519-gpu-vanity", "https://github.com/vikulin/ed25519-gpu-vanity",
          solanity("https://github.com/vikulin/ed25519-gpu-vanity", "ed25519-gpu-vanity"),
          has_cuda, "GPU と nvcc（CUDA Toolkit）"),
    Build("ed25519-gpu-vanity（フォーク）", "https://github.com/84634E1A607A/ed25519-gpu-vanity",
          solanity("https://github.com/84634E1A607A/ed25519-gpu-vanity", "ed25519-gpu-vanity-fork"),
          has_cuda, "GPU と nvcc（CUDA Toolkit）"),
]


def build_all(only: Optional[list[str]] = None) -> dict[str, bool]:
    """ツールを全部ビルドする（ログは LOGS/<名前>-build.log）。名前ごとにできたかを返す"""
    for d in (SRC, BIN, LOGS):
        os.makedirs(d, exist_ok=True)
    built = {}
    for b in BUILDS:
        if only and b.name not in only:
            continue
        if not b.needs():
            print(f"✗ {b.name}: {b.needs_text} が無いので飛ばす")
            built[b.name] = False
            continue
        commands = b.commands.replace("{cc}", gpu_cc()) if "{cc}" in b.commands else b.commands
        code, out, sec = run(commands, cwd=SRC, timeout=3600)
        with open(os.path.join(LOGS, f"{b.name}-build.log"), "w") as f:
            f.write(out)
        built[b.name] = code == 0
        mark = "✓" if code == 0 else "✗"
        tail = "" if code == 0 else f"（ログ: {LOGS}/{b.name}-build.log）\n" + "\n".join(out.strip().splitlines()[-5:])
        print(f"{mark} {b.name}: {sec:.0f} 秒{tail}")
    return built


# ---------------------------------------------------------------------------
# 速度の測り方
# ---------------------------------------------------------------------------

SUFFIX = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}


def number(s: str) -> float:
    return float(s.replace(",", ""))


def parse_rate(out: str, pattern: str, scale: float = 1) -> Optional[float]:
    """出力の中で最後に pattern に合った速度（鍵/秒）。グループ 2 があれば K・M・B の接尾辞"""
    found = re.findall(pattern, plain(out))
    if not found:
        return None
    last = found[-1]
    if isinstance(last, tuple):
        return number(last[0]) * SUFFIX.get(last[1], 1) * scale
    return number(last) * scale


# 文字列が当たる確率（公開鍵の行の base64 68 文字のうち、鍵で決まるのは後ろの 42 文字と、その前の 1 文字の 4 ビット。
# LETTERS は A〜P を含まないので、その 1 文字には当たらず、後ろの 42 文字の中の 43 - k か所に入る）
def anywhere_case_sensitive(k: int) -> float:
    return (43 - k) / 64**k


def anywhere_ignore_case(k: int) -> float:
    return (43 - k) / 32**k


def suffix_case_sensitive(k: int) -> float:
    return 1 / 64**k



@dataclass
class Row:
    """表の1行（1つのツールの1つの使い方）"""
    name: str
    build: str
    lang: str
    device: str  # CPU・GPU・GPU + CPU
    target: str  # 文字列を探す場所（公開鍵・フィンガープリント）
    note: str
    # 速度を表示するツール: cmd(秒数) と、速度を読む正規表現
    cmd: Optional[Callable[[int], str]] = None
    rate: str = ""
    scale: float = 1
    tty: bool = False
    # 自分で止まる（meikey の --bench は決めた秒数で、VaniSSH は見つけたら止まって速度を出す）なら、待つ上限を延ばす
    self_timed: bool = False
    # 速度を表示しないツール: find(文字列) と、その文字列が当たる確率
    find: Optional[Callable[[str], str]] = None
    chance: Optional[Callable[[int], float]] = None
    cwd: Optional[str] = None
    needs: Callable[[], bool] = lambda: True
    # needs を満たさないとき、表に出さない（どちらか一方だけ測る組み合わせ用）
    hide: bool = False


def measure_rate(row: Row, seconds: int) -> tuple[Optional[float], str]:
    cwd = row.cwd or tempfile.mkdtemp(dir=WORK)  # 鍵のファイルを書くツールもあるので、空のディレクトリで動かす
    limit = max(seconds * 3, seconds + 120) if row.self_timed else seconds
    code, out, _ = run(row.cmd(seconds), cwd=cwd, timeout=limit, tty=row.tty)
    cleanup()
    if not row.cwd:
        shutil.rmtree(cwd, ignore_errors=True)
    rate = parse_rate(out, row.rate, row.scale)
    with open(os.path.join(LOGS, f"{slug(row.name)}.log"), "w") as f:
        f.write(out)
    return rate, "ツールが表示した速度"


def measure_hits(row: Row, seconds: int) -> tuple[Optional[float], str]:
    """短い文字列で何度も探させ、速度 = 見つかった数 / (当たる確率 × (かかった時間 - 起動の時間 × 起動の回数))"""
    log = open(os.path.join(LOGS, f"{slug(row.name)}.log"), "w")

    def once(k: int, limit: float) -> tuple[bool, float]:
        cwd = tempfile.mkdtemp(dir=WORK)
        code, out, sec = run(row.find(LETTERS[:k]), cwd=cwd, timeout=max(limit, 0.1))
        shutil.rmtree(cwd, ignore_errors=True)
        log.write(f"--- k={k} code={code} {sec:.3f}s\n{out[-2000:]}\n")
        return code == 0, sec

    # 起動の時間（1 文字なら、ほぼ最初の数本で見つかる）
    overhead = statistics.median(once(1, 60)[1] for _ in range(3))
    # だいたいの速度を k = 3 で見積もり、1回見つかるまでが起動の時間の 5 倍以上になる、いちばん短い k を選ぶ
    # （短いほど見つかる回数が増えて、速度のぶれが小さくなる）
    k, spent, hits, runs = 3, 0.0, 0, 0
    budget = max(3.0, seconds * 0.2)
    while spent < budget and hits < 20:
        ok, sec = once(k, budget - spent)
        spent += sec
        runs += 1
        hits += ok
    guess = hits / (row.chance(k) * max(spent - runs * overhead, 1e-3)) if hits else 0
    while guess and k < len(LETTERS) and 1 / (row.chance(k) * guess) < max(5 * overhead, 0.05):
        k += 1
    # 本番
    spent, hits, runs = 0.0, 0, 0
    while spent < seconds:
        ok, sec = once(k, seconds - spent)
        spent += sec
        runs += 1
        hits += ok
    cleanup()
    log.write(f"=== overhead={overhead:.3f}s k={k} hits={hits} runs={runs} spent={spent:.1f}s\n")
    log.close()
    if not hits:
        return None, f"見つかった数から計算（{seconds} 秒で見つからず）"
    work = max(spent - runs * overhead, 1e-3)
    return hits / (row.chance(k) * work), f"見つかった数から計算（{k} 文字・{hits} 回）"


def slug(name: str) -> str:
    """ログのファイル名（日本語も残す。「（フォーク）」を消すと元のツールと同じ名前になるため）"""
    return re.sub(r"[^\w.+-]+", "_", name).strip("_")


def src(*p: str) -> str:
    return os.path.join(SRC, *p)


MEIKEY = os.path.join(REPO, "target", "release", "meikey")

ROWS = [
    Row("meikey", "meikey", "Rust", "CPU", "公開鍵", "既定（定数時間の倍算・鍵ごとに ChaCha20 のシード）",
        cmd=lambda s: f"{MEIKEY} {NEVER} --bench {s}", rate=r"Speed\s+([\d.]+)\s*([KMB]?) keys/s", self_timed=True),
    # 速さを優先した組み合わせ: GPU があれば GPU と CPU の両方、無ければ CPU だけの低セキュリティモード
    Row("meikey --low-security --both", "meikey", "Rust・CUDA", "GPU + CPU", "公開鍵",
        "低セキュリティモードで GPU と CPU を同時に使う（GPU の設定は起動時に測って選ぶ）",
        cmd=lambda s: f"{MEIKEY} {NEVER} --low-security --both --bench {s}",
        rate=r"Speed\s+([\d.]+)\s*([KMB]?) keys/s", self_timed=True, needs=has_gpu, hide=True),
    Row("meikey --low-security", "meikey", "Rust", "CPU", "公開鍵",
        "低セキュリティモード（GPU が無いときだけ測る）",
        cmd=lambda s: f"{MEIKEY} {NEVER} --low-security --bench {s}",
        rate=r"Speed\s+([\d.]+)\s*([KMB]?) keys/s", self_timed=True, needs=lambda: not has_gpu(), hide=True),
    Row("vanityssh-go", "vanityssh-go", "Go", "CPU", "公開鍵（正規表現）", "",
        cmd=lambda s: f"vanityssh-go {NEVER}", rate=r"Rate: ([\d,]+)/s", tty=True),
    Row("vanity-ssh-keygen", "vanity-ssh-keygen", "Go", "CPU", "公開鍵", "",
        cmd=lambda s: f"{src('vanity-ssh-keygen', 'vanity-ssh-keygen')} {NEVER}", rate=r"kKeys/s=([\d.]+)", scale=1e3),
    Row("vanity-ssh-keygen --voi", "vanity-ssh-keygen", "Go", "CPU", "公開鍵", "curve25519-voi で鍵を作る",
        cmd=lambda s: f"{src('vanity-ssh-keygen', 'vanity-ssh-keygen')} --voi {NEVER}",
        rate=r"kKeys/s=([\d.]+)", scale=1e3),
    Row("vanity-ssh-rs", "vanity-ssh-rs", "Rust", "CPU", "公開鍵（末尾・正規表現）", "",
        cmd=lambda s: f"vanity-ssh-rs {NEVER}", tty=True,
        # crates.io の 1.1.1 は「Attempts: N (X keys/sec)」、GitHub の新しい版は「… | X keys/sec (avg)」
        rate=r"([\d,]+) keys/sec(?: \(avg\)|\))"),
    Row("vanissh", "vanissh", "Rust", "CPU", "公開鍵", "大小文字の区別なし（既定）",
        find=lambda p: f"vanissh {p}", chance=anywhere_ignore_case),
    Row("ssh-keygen-ed25519-vanity（Rust）", "ssh-keygen-ed25519-vanity", "Rust", "CPU", "公開鍵（正規表現）",
        f"-j {NPROC}",
        find=lambda p: f"{src('ssh-keygen-ed25519-vanity', 'target', 'release', 'vanity')} {p} -j {min(NPROC, 255)}",
        chance=anywhere_case_sensitive),
    Row("ssh-keygen-ed25519-vanity（C）", "ssh-keygen-ed25519-vanity", "C", "CPU", "公開鍵",
        f"1 スレッドのプログラムを {NPROC} 個同時に動かす",
        find=lambda p: (
            "trap 'kill $(jobs -p) 2>/dev/null' EXIT; "
            f"for i in $(seq {NPROC}); do {src('ssh-keygen-ed25519-vanity', 'vanity')} {p} > /dev/null & done; "
            "wait -n"),
        chance=anywhere_case_sensitive),
    Row("ed25519brute", "ed25519brute", "Go", "CPU", "公開鍵（末尾）・フィンガープリント", "公開鍵の末尾で測る",
        find=lambda p: f"ed25519brute -authorized-key-suffix {p}", chance=suffix_case_sensitive),
    Row("SSH-Ed25519-VanityGen", "SSH-Ed25519-VanityGen", "Python", "CPU", "公開鍵", "",
        # 5 秒ごとの進み具合の行（止めたときの最後の行は、止めた時点の数しか数えていないので読まない）
        cmd=lambda s: f"python3 ssh_ed25519_vanity_multicpu.py {NEVER}",
        rate=r"Progress: [\d,]+ keys \([^)]*\) \(avg ~([\d,]+) keys/sec\)",
        cwd=src("SSH-Ed25519-VanityGen")),
    # 速度は見つけたときにだけ表示するので、4 文字で探させる（コメントの -e を付けないと落ちる）
    Row("VaniSSH", "VaniSSH", "Python", "CPU", "公開鍵（正規表現）", "大小文字の区別なし（既定）。4 文字を見つけるまでの速度",
        cmd=lambda s: f"python3 src/vanissh.py -e bench@example.com -ap {LETTERS[:4]} -n {NPROC}", rate=r"Keys per second: ([\d.]+)",
        cwd=src("scottvr-vanissh"), self_timed=True),
    Row("vanity-key-ssh（CPU）", "vanity-key-ssh", "C++", "CPU", "公開鍵（末尾）", "--benchmark",
        cmd=lambda s: f"bin/cpu_vanity --suffix {NEVER} --table table.bin --benchmark",
        rate=r"= ([\d.]+) keys/s", cwd=src("vanity-key-ssh")),
    Row("vanity-key-ssh（GPU）", "vanity-key-ssh", "CUDA", "GPU", "公開鍵（末尾）", "--benchmark",
        cmd=lambda s: f"bin/gpu_vanity --suffix {NEVER} --table table16.bin --benchmark",
        rate=r"= ([\d.]+) Mkeys/s", scale=1e6, cwd=src("vanity-key-ssh"),
        needs=lambda: os.path.exists(src("vanity-key-ssh", "bin", "gpu_vanity"))),
    Row("vanity-keygen", "vanity-keygen", "Rust・OpenCL", "GPU", "公開鍵（正規表現の一部）", "",
        cmd=lambda s: f"{src('vanity-keygen', 'target', 'release', 'vanity-keygen')} {NEVER}",
        rate=r"\(([\d,]+) Hz\)"),
    Row("ed25519brute-cuda", "ed25519brute-cuda", "CUDA", "GPU", "フィンガープリント", "",
        cmd=lambda s: f"{src('ed25519brute-cuda', 'ed25519brute_cuda')} --fingerprint-prefix {NEVER}",
        rate=r"Speed: ([\d.]+) MKeys/s", scale=1e6),
    Row("SSH-Ed25519-VanityGen（GPU）", "SSH-Ed25519-VanityGen", "Python・OpenCL", "GPU", "公開鍵", "",
        # 5 秒ごとの行の「GPU0 47.7M/s」（GPU ごとの速度）を読む。「(~…/s)」の平均はカーネルのコンパイルの
        # 時間も含むので読まない
        cmd=lambda s: f"python3 ssh_ed25519_vanity_gpu_opencl.py {NEVER}", rate=r"\| GPU0 ([\d.]+)M/s", scale=1e6,
        cwd=src("SSH-Ed25519-VanityGen", "OpenCL-GPU"), needs=has_opencl_gpu),
    # 探す条件はソースに書いてあり、引数では渡せない（公開鍵の先頭 4 バイトが 0）。端末でないと出力が
    # まとめて書かれ、止めたときに失われるので、行ごとに書かせる。LD_LIBRARY_PATH は足す（Colab では
    # CUDA のドライバ（libcuda）が LD_LIBRARY_PATH の /usr/lib64-nvidia にあり、置き換えると GPU が見つからない）
    Row("ed25519-gpu-vanity", "ed25519-gpu-vanity", "CUDA", "GPU", "公開鍵（先頭のバイト）",
        "条件はソースの既定（公開鍵の先頭 4 バイトが 0）",
        cmd=lambda s: "LD_LIBRARY_PATH=src/release:$LD_LIBRARY_PATH stdbuf -oL src/release/cuda_ed25519_vanity",
        rate=r"at ([\d.]+)cps", cwd=src("ed25519-gpu-vanity")),
    # 条件はソースの既定（base64 の末尾が +Ajax、30 ビット）。1 つ見つけると止まるので、止まったらまた動かす
    Row("ed25519-gpu-vanity（フォーク）", "ed25519-gpu-vanity（フォーク）", "CUDA", "GPU", "公開鍵（末尾）",
        "条件はソースの既定（base64 の末尾が +Ajax）。見つけて止まったら起動し直す",
        cmd=lambda s: ("while :; do LD_LIBRARY_PATH=src/release:$LD_LIBRARY_PATH stdbuf -oL src/release/cuda_ed25519_vanity"
                       " || break; done"),
        rate=r"at ([\d.]+)cps", cwd=src("ed25519-gpu-vanity-fork")),
]


def count(x: Optional[float]) -> str:
    """meikey の表示と同じ「2.1億」「6250.1万」の形"""
    if x is None:
        return "—"
    if x >= 1e8:
        return f"{x / 1e8:.1f}億"
    if x >= 1e4:
        return f"{x / 1e4:.1f}万"
    return f"{x:.0f}"


def bench_all(seconds: int, built: dict[str, bool], only: Optional[list[str]] = None) -> list[dict]:
    """ビルドできたツールを1つずつ測る。結果は1行1つの dict（表にする）"""
    results = []
    for row in ROWS:
        if only and row.name not in only and row.build not in only:
            continue
        if not built.get(row.build):
            rate, how = None, "ビルドできなかった"
        elif not row.needs():
            if row.hide:
                continue
            rate, how = None, "このマシンでは動かない"
        else:
            print(f"---- {row.name}", flush=True)
            try:
                rate, how = (measure_hits if row.find else measure_rate)(row, seconds)
            except Exception as e:  # 1つのツールの失敗で全体を止めない
                rate, how = None, f"失敗: {e}"
            if rate is None and how == "ツールが表示した速度":
                how = f"速度を読めなかった（ログ: {LOGS}/{slug(row.name)}.log）"
            print(f"{count(rate)} 鍵/秒（{how}）", flush=True)
        results.append({"ツール": row.name, "言語": row.lang, "使うもの": row.device, "探す場所": row.target,
                        "鍵/秒": rate, "速度": count(rate), "測り方": how, "メモ": row.note})
    return results


def markdown(results: list[dict]) -> str:
    """結果の Markdown の表（速い順）"""
    rows = sorted(results, key=lambda r: -(r["鍵/秒"] or 0))
    cols = ["ツール", "言語", "使うもの", "探す場所", "速度", "測り方", "メモ"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)
