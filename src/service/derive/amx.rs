//! AMX（Advanced Matrix Extensions）のタイル演算で体の乗算をする、実験的な基点倍算
//!
//! 倍算の流れは [`Table::mul_base_batch`] と同じ（表を1行ずつ全部の鍵に足す）で、体の乗算だけを
//! AMX の `tdpbuud`（u8 × u8 を4つずつ足して i32 に積和する行列積）で計算する。
//!
//! ## 体の元
//!
//! 2^64 進 4 語（[`F`]）。2^256 未満で、p = 2^255 - 19 を法として等しい値（p 以上のこともある）。
//! 足し算・引き算は普通のレジスタで繰り上がりを伝え、2^256 ≡ 38 (mod p) で 256 ビットに収める。
//! 4 語をリトルエンディアンのバイト列として見ると、そのまま 256 進 32 桁（各桁 0〜255）になる。
//!
//! ## 乗算 x·y
//!
//! バイト列の畳み込み `c_n = Σ x_i·y_{n-i}`（n = 0..=62、各 c_n < 32·255² < 2^21）を、`tdpbuud` 1回で
//! `C[m][r] = c_{m+4r}`（m = 0..4, r = 0..16）として計算する。`C[m][r] = Σ_k A[m][k]·B[k][r]` に
//! なるよう、A と B を次のように取る（k = 0..36）:
//!
//! - `A[m][k] = x_{m+k-3}`: x の前に 0 を3バイト置いたバッファを、1バイトずつずらして読む（行の間隔 +1）
//! - `B[k][r] = y_{4r+3-k}`: B は VNNI 形式（K の4つを1行にまとめる）で、`B 行 q の 4r+i バイト目 =
//!   ỹ[4(r-q)+i]`（ỹ は y を4バイトごとに並べ逆にしたもの）。ỹ の前後を 0 で埋めたバッファを、
//!   -4 バイトずつずらして読む（行の間隔 -4）
//!
//! どちらもタイルの読み込み（`tileloadd`）の行の間隔だけで作れるので、行列を組み立てる手間はない。
//! C の m 行目は c_m, c_{m+4}, …, c_{m+60} を 32 ビットずつ並べたものなので、そのまま 512 ビットの
//! 整数 W_m として読め、`x·y = W_0 + W_1·2^8 + W_2·2^16 + W_3·2^24`。これを 2^256 ≡ 38 で畳む。
//!
//! ## 速さ
//!
//! 1回の `tdpbuud` で使うのはタイルの 16 行のうち 4 行で、行列の片方を使い回せない（鍵ごとに掛ける表の
//! エントリが違う）。さらに普通のレジスタとタイルの間の受け渡しがメモリ経由で遅いので、[`GROUP`] 鍵ぶんの
//! 乗算を「入力を書く → タイル演算 → 結果を畳む」の3段にまとめて流しても、乗算1回に約 35ns かかる。
//! Emerald Rapids（2.1GHz）の1スレッドで1鍵あたり約 5.8µs と、AVX-512 IFMA（約 0.77µs）はもちろん、
//! SIMD を使わない 51 ビット × 5 リムの体演算（約 3.2µs）より遅い。AMX でこの計算がどこまで速くなるかを
//! 試すための経路で、`--simd amx` を指定したときだけ使う。
//!
//! ## CPU と OS
//!
//! Rust 1.99 では AMX の intrinsics・`target_feature`・`is_x86_feature_detected!` が nightly 限定なので、
//! 命令は `asm!` で書き、対応は CPUID と XGETBV で調べる。Linux ではプロセスごとにタイルを使う許可が
//! 要るので arch_prctl で取る。Linux 以外では試していないので使わない。
//! SHA-512 は AVX-512 で8つずつ計算するので、AVX-512 にも対応している CPU でだけ使う
//! （AMX のある CPU はすべて AVX-512 にも対応している）。

use crate::service::derive::avx512::Avx512;
use crate::service::derive::comb::{COLS, Extended, ROWS, Table, signed_digits};
use crate::service::derive::field::Fe;
use std::arch::asm;
use std::arch::x86_64::__cpuid_count;
use std::marker::PhantomData;
use std::sync::{LazyLock, OnceLock};

/// CPU が AMX（AMX-TILE・AMX-INT8）と AVX-512 に対応し、OS がタイルを使わせてくれることの証明
/// （[`Amx::detect`] でしか作れない）
#[derive(Clone, Copy, Debug)]
pub struct Amx(Avx512);

impl Amx {
    pub fn detect() -> Option<Amx> {
        // 許可の要求はプロセスで1回
        static ENABLED: OnceLock<bool> = OnceLock::new();
        let avx512 = Avx512::detect()?;
        ENABLED.get_or_init(enable).then_some(Amx(avx512))
    }

    /// SHA-512 を計算する AVX-512
    pub fn avx512(self) -> Avx512 {
        self.0
    }
}

/// XCR0 の、OS がタイルの設定（17 ビット目）とデータ（18 ビット目）を保存するビット
const XCR0_TILE: u64 = 0b11 << 17;

fn enable() -> bool {
    let leaf1 = __cpuid_count(1, 0);
    let leaf7 = __cpuid_count(7, 0);
    let osxsave = leaf1.ecx >> 27 & 1 == 1;
    let amx_tile = leaf7.edx >> 24 & 1 == 1;
    let amx_int8 = leaf7.edx >> 25 & 1 == 1;
    // XGETBV は OSXSAVE が立っているときだけ使える
    amx_tile && amx_int8 && osxsave && xcr0() & XCR0_TILE == XCR0_TILE && request_permission()
}

/// XCR0（OS が保存するレジスタの種類）。OSXSAVE が立っているときだけ呼ぶこと
fn xcr0() -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: 呼び出し元が OSXSAVE を確かめている。XGETBV はレジスタを読むだけ
    unsafe {
        asm!("xgetbv", in("ecx") 0, out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags));
    }
    u64::from(hi) << 32 | u64::from(lo)
}

cfg_select! {
    target_os = "linux" => {
        /// タイルのデータを使う許可を取る（arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)）。
        /// 許可はプロセス全体（全スレッド）に効く
        fn request_permission() -> bool {
            const SYS_ARCH_PRCTL: i64 = 158;
            const ARCH_REQ_XCOMP_PERM: i64 = 0x1023;
            const XFEATURE_XTILEDATA: i64 = 18;
            let ret: i64;
            // SAFETY: このシステムコールはユーザーのメモリに触れず、失敗すると負の値を返すだけ
            unsafe {
                asm!(
                    "syscall",
                    inlateout("rax") SYS_ARCH_PRCTL => ret,
                    in("rdi") ARCH_REQ_XCOMP_PERM,
                    in("rsi") XFEATURE_XTILEDATA,
                    lateout("rcx") _,
                    lateout("r11") _,
                    options(nostack),
                );
            }
            ret == 0
        }
    }
    _ => {
        /// Linux 以外では試していないので使わない
        fn request_permission() -> bool {
            false
        }
    }
}

/// 2^255 - 19 を法とする体の元。2^64 進 4 語（リトルエンディアン）で 2^256 未満（p 以上のこともある）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct F([u64; 4]);

impl F {
    const ZERO: F = F([0; 4]);
    const ONE: F = F([1, 0, 0, 0]);

    fn from_fe(x: Fe) -> F {
        let b = x.to_bytes();
        F(std::array::from_fn(|i| {
            u64::from_le_bytes(*b[8 * i..].first_chunk().expect("32 バイト"))
        }))
    }

    fn to_fe(self) -> Fe {
        // 2^255 ≡ 19 で 255 ビット未満にしてから（Fe::from_bytes は最上位ビットを無視する）
        let mut v = self;
        for _ in 0..2 {
            let top = v.0[3] >> 63;
            v.0[3] &= u64::MAX >> 1;
            v = v.add_small(19 * top);
        }
        let mut b = [0u8; 32];
        for (d, w) in b.as_chunks_mut::<8>().0.iter_mut().zip(v.0) {
            *d = w.to_le_bytes();
        }
        Fe::from_bytes(&b)
    }

    /// self + k（k < 2^63）。2^256 を超えたら 38 を足して戻す
    #[inline(always)]
    fn add_small(self, k: u64) -> F {
        let mut r = self.0;
        let mut carry;
        (r[0], carry) = r[0].overflowing_add(k);
        for w in &mut r[1..] {
            (*w, carry) = w.carrying_add(0, carry);
        }
        if carry {
            // 2^256 を超えた分は 2^256 ≡ 38。超えたなら r は小さいので、もう一度は超えない
            r[0] += 38;
        }
        F(r)
    }

    #[inline(always)]
    fn add(self, b: F) -> F {
        let mut r = [0u64; 4];
        let mut carry = false;
        for ((r, &x), &y) in r.iter_mut().zip(&self.0).zip(&b.0) {
            (*r, carry) = x.carrying_add(y, carry);
        }
        F(r).add_small(38 * u64::from(carry))
    }

    #[inline(always)]
    fn sub(self, b: F) -> F {
        let mut r = [0u64; 4];
        let mut borrow = false;
        for ((r, &x), &y) in r.iter_mut().zip(&self.0).zip(&b.0) {
            (*r, borrow) = x.borrowing_sub(y, borrow);
        }
        // 借りた 2^256 は 38 なので、その分 38 を引く（引けなければもう一度借りる）
        for _ in 0..2 {
            if !borrow {
                break;
            }
            (r[0], borrow) = r[0].overflowing_sub(38);
            for w in &mut r[1..] {
                (*w, borrow) = w.borrowing_sub(0, borrow);
            }
        }
        F(r)
    }

    #[inline(always)]
    fn neg(self) -> F {
        F::ZERO.sub(self)
    }

    /// 512 ビットの値（8 語）を 2^256 ≡ 38 で畳む
    #[inline(always)]
    fn reduce_wide(s: [u64; 8]) -> F {
        let mut r = [0u64; 4];
        let mut c = 0;
        for i in 0..4 {
            (r[i], c) = s[i + 4].carrying_mul_add(38, s[i], c);
        }
        // c < 39 なので c·2^256 ≡ 38c
        F(r).add_small(38 * c)
    }
}

/// 乗算1回ぶんのタイルの入出力
#[repr(C, align(64))]
struct Slot {
    /// 0 を3バイト、x（32 バイト）、0。A の m 行目は m バイト目からの 36 バイト
    a: [u8; 64],
    /// 0 を 32 バイト、ỹ（32 バイト）、0 を 64 バイト。B の q 行目は 32 - 4q バイト目からの 64 バイト
    b: [u8; 128],
    /// C の m 行目 = W_m（512 ビットの整数を 8 語で）
    c: [[u64; 8]; 4],
}

impl Slot {
    const ZERO: Slot = Slot {
        a: [0; 64],
        b: [0; 128],
        c: [[0; 8]; 4],
    };

    /// x·y を計算するよう入力を書く（a・b の 0 の部分はそのまま）
    #[inline(always)]
    fn set(&mut self, x: &F, y: &F) {
        for (d, w) in self.a[3..35].as_chunks_mut::<8>().0.iter_mut().zip(x.0) {
            *d = w.to_le_bytes();
        }
        // 1語（8 バイト b0..b7）を b3 b2 b1 b0 b7 b6 b5 b4 の順に
        for (d, w) in self.b[32..64].as_chunks_mut::<8>().0.iter_mut().zip(y.0) {
            *d = w.swap_bytes().rotate_left(32).to_le_bytes();
        }
    }

    /// タイル演算の結果（C）から x·y
    #[inline(always)]
    fn product(&self) -> F {
        let mut s = [0u64; 8];
        for (m, w) in self.c.iter().enumerate() {
            // W_m·2^(8m) を足す（積は 2^512 未満なので、どの W_m·2^(8m) も 8 語に収まる）
            let sh = 8 * m as u32;
            let mut carry = false;
            for i in 0..8 {
                let v = if sh == 0 {
                    w[i]
                } else {
                    w[i] << sh | if i > 0 { w[i - 1] >> (64 - sh) } else { 0 }
                };
                (s[i], carry) = s[i].carrying_add(v, carry);
            }
            debug_assert!(!carry);
        }
        F::reduce_wide(s)
    }
}

/// タイルの設定（palette 1）。乗算を2組交互に流すので、C・A・B を2組（tmm0〜2 と tmm3〜5）
#[repr(C, align(64))]
struct TileConfig([u8; 64]);

static CONFIG: TileConfig = {
    let mut c = [0u8; 64];
    c[0] = 1;
    // (タイル, 行数, 1行のバイト数): C は 4 行 × 16 個の i32、A は 4 行 × 36 バイト、B は 9 行 × 64 バイト
    let tiles = [
        (0, 4, 64),
        (1, 4, 36),
        (2, 9, 64),
        (3, 4, 64),
        (4, 4, 36),
        (5, 9, 64),
    ];
    let mut i = 0;
    while i < tiles.len() {
        let (t, rows, colsb) = tiles[i];
        c[16 + 2 * t] = colsb;
        c[48 + t] = rows;
        i += 1;
    }
    TileConfig(c)
};

/// このスレッドにタイルの設定を読み込んだ状態。なくなるとタイルを解放する
/// （設定はスレッドごとなので、別のスレッドに渡せないよう Send にしない）
struct Tiles(PhantomData<*const ()>);

impl Tiles {
    fn load(amx: Amx) -> Tiles {
        let _ = amx;
        // SAFETY: Amx があるので、CPU は AMX に対応し、OS はタイルを使わせてくれる。CONFIG は正しい設定
        unsafe {
            asm!("ldtilecfg [{}]", in(reg) &raw const CONFIG, options(nostack, readonly));
        }
        Tiles(PhantomData)
    }

    /// 各 Slot の x·y を計算して c に書く（2組のタイルを交互に使う）
    #[inline(always)]
    fn run(&self, slots: &mut [Slot]) {
        let (pairs, rest) = slots.as_chunks_mut::<2>();
        for [s0, s1] in pairs {
            self.mul::<0>(s0);
            self.mul::<1>(s1);
        }
        if let [s] = rest {
            self.mul::<0>(s);
        }
    }

    #[inline(always)]
    fn mul<const SET: usize>(&self, s: &mut Slot) {
        let a = s.a.as_ptr();
        // SAFETY: B の行は b[32 - 4q..96 - 4q]（q = 0..9）で、b の中に収まる
        let b = unsafe { s.b.as_ptr().add(32) };
        let c = s.c.as_mut_ptr();
        // SAFETY: このスレッドに CONFIG を読み込んである（self があるので）。読み書きは a・b・c の中だけ
        unsafe {
            if SET == 0 {
                asm!(
                    "tileloadd tmm1, [{a} + {one}*1]",
                    "tileloadd tmm2, [{b} + {m4}*1]",
                    "tilezero tmm0",
                    "tdpbuud tmm0, tmm1, tmm2",
                    "tilestored [{c} + {row}*1], tmm0",
                    a = in(reg) a, b = in(reg) b, c = in(reg) c,
                    one = in(reg) 1_i64, m4 = in(reg) -4_i64, row = in(reg) 64_i64,
                    options(nostack),
                );
            } else {
                asm!(
                    "tileloadd tmm4, [{a} + {one}*1]",
                    "tileloadd tmm5, [{b} + {m4}*1]",
                    "tilezero tmm3",
                    "tdpbuud tmm3, tmm4, tmm5",
                    "tilestored [{c} + {row}*1], tmm3",
                    a = in(reg) a, b = in(reg) b, c = in(reg) c,
                    one = in(reg) 1_i64, m4 = in(reg) -4_i64, row = in(reg) 64_i64,
                    options(nostack),
                );
            }
        }
    }
}

impl Drop for Tiles {
    fn drop(&mut self) {
        // SAFETY: タイルの状態を初期化するだけ
        unsafe {
            asm!("tilerelease", options(nostack, nomem));
        }
    }
}

/// 拡張座標 (X:Y:Z:T)
#[derive(Clone, Copy)]
struct Point {
    x: F,
    y: F,
    z: F,
    t: F,
}

impl Point {
    const IDENTITY: Point = Point {
        x: F::ZERO,
        y: F::ONE,
        z: F::ONE,
        t: F::ZERO,
    };
}

/// 加算用に前計算したアフィン点 (y+x, y-x, 2dxy)
#[derive(Clone, Copy)]
struct Niels {
    ypx: F,
    ymx: F,
    xy2d: F,
}

/// [`Table`] と同じ値を [`F`] で持つテーブル（W = 12 で約 4.3MB）
struct TableF(Box<[Niels]>);

impl TableF {
    fn get() -> &'static TableF {
        static TABLE: LazyLock<TableF> = LazyLock::new(|| {
            let t = Table::get();
            TableF(
                (0..ROWS * COLS)
                    .map(|j| {
                        let [ypx, ymx, xy2d] = t.entry(j / COLS, j % COLS).map(F::from_fe);
                        Niels { ypx, ymx, xy2d }
                    })
                    .collect(),
            )
        });
        &TABLE
    }

    /// `i` 行目の、桁 `d` の点（負なら符号を反転）
    #[inline(always)]
    fn entry(&self, i: usize, d: i16) -> Niels {
        let n = self.0[i * COLS + usize::from(d.unsigned_abs())];
        if d < 0 {
            Niels {
                ypx: n.ymx,
                ymx: n.ypx,
                xy2d: n.xy2d.neg(),
            }
        } else {
            n
        }
    }
}

/// 一度に乗算をまとめて流す鍵の数
const GROUP: usize = 8;

/// 表を作っておく（探索の時間に含めないため）
pub fn prepare(amx: Amx) {
    let _ = amx;
    TableF::get();
}

/// [`Table::mul_base_batch`] と同じ計算を、体の乗算を AMX で
pub fn mul_base_batch<const N: usize>(amx: Amx, scalars: &[[u8; 32]; N], out: &mut [Extended; N]) {
    let table = TableF::get();
    let mut digits = [[0i16; N]; ROWS];
    for (k, s) in scalars.iter().enumerate() {
        for (i, d) in signed_digits(s).into_iter().enumerate() {
            digits[i][k] = d;
        }
    }
    let mut acc = vec![Point::IDENTITY; N];
    let mut slots: Box<[Slot; 4 * GROUP]> = Box::new([const { Slot::ZERO }; 4 * GROUP]);
    let tiles = Tiles::load(amx);

    // 単位元から始めて、全部の行を足す（桁 0 は表の0列目 = 単位元を足す）
    for (i, ds) in digits.iter().enumerate() {
        let with_t = i + 1 < ROWS; // 最後の加算では T を計算しない
        for (accs, ds) in acc.chunks_mut(GROUP).zip(ds.chunks(GROUP)) {
            let n = accs.len();
            // 1段目: (Y+X)·(y+x), (Y-X)·(y-x), T·2dxy
            for (j, (p, &d)) in accs.iter().zip(ds).enumerate() {
                let q = table.entry(i, d);
                slots[3 * j].set(&p.y.add(p.x), &q.ypx);
                slots[3 * j + 1].set(&p.y.sub(p.x), &q.ymx);
                slots[3 * j + 2].set(&p.t, &q.xy2d);
            }
            tiles.run(&mut slots[..3 * n]);
            let mut xyzt = [[F::ZERO; 4]; GROUP];
            for (j, p) in accs.iter().enumerate() {
                let (pp, mm, txy) = (
                    slots[3 * j].product(),
                    slots[3 * j + 1].product(),
                    slots[3 * j + 2].product(),
                );
                let z2 = p.z.add(p.z);
                xyzt[j] = [pp.sub(mm), pp.add(mm), z2.add(txy), z2.sub(txy)];
            }
            // 2段目: X3 = x·t, Y3 = y·z, Z3 = z·t, T3 = x·y
            let per = if with_t { 4 } else { 3 };
            for (j, [x, y, z, t]) in xyzt[..n].iter().enumerate() {
                slots[per * j].set(x, t);
                slots[per * j + 1].set(y, z);
                slots[per * j + 2].set(z, t);
                if with_t {
                    slots[per * j + 3].set(x, y);
                }
            }
            tiles.run(&mut slots[..per * n]);
            for (j, p) in accs.iter_mut().enumerate() {
                *p = Point {
                    x: slots[per * j].product(),
                    y: slots[per * j + 1].product(),
                    z: slots[per * j + 2].product(),
                    t: if with_t {
                        slots[per * j + 3].product()
                    } else {
                        F::ZERO
                    },
                };
            }
        }
    }
    drop(tiles);

    for (o, p) in out.iter_mut().zip(&acc) {
        *o = Extended {
            x: p.x.to_fe(),
            y: p.y.to_fe(),
            z: p.z.to_fe(),
            t: Fe::ZERO,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::check;
    use crate::simd_support::available;
    use curve25519_dalek::edwards::EdwardsPoint;
    use curve25519_dalek::scalar::clamp_integer;
    use num_bigint::BigUint;
    use proptest::prelude::*;
    use rstest::rstest;

    /// p = 2^255 - 19
    fn p() -> BigUint {
        (BigUint::from(1u8) << 255u32) - 19u32
    }

    fn big(x: &F) -> BigUint {
        BigUint::from_slice(
            &x.0.iter()
                .flat_map(|w| [*w as u32, (w >> 32) as u32])
                .collect::<Vec<_>>(),
        )
    }

    /// 2^256 未満の任意の値。端（0・p 付近・2^256 - 1 付近・全部のバイトが 0xff）を多めに混ぜる
    fn any_f() -> impl Strategy<Value = F> {
        let edge = prop_oneof![
            Just([0, 0, 0, 0]),
            Just([u64::MAX - 18, u64::MAX, u64::MAX, u64::MAX >> 1]), // p
            Just([u64::MAX - 19, u64::MAX, u64::MAX, u64::MAX >> 1]), // p - 1
            Just([u64::MAX; 4]),
            Just([u64::MAX - 37, u64::MAX, u64::MAX, u64::MAX]), // 2^256 - 38 = 2p
            Just([u64::MAX - 38, u64::MAX, u64::MAX, u64::MAX]),
        ];
        prop_oneof![3 => any::<[u64; 4]>(), 1 => edge].prop_map(F)
    }

    /// AMX が使えなければテストを飛ばす（MEIKEY_REQUIRE_SIMD に amx があれば失敗させる）
    fn amx() -> Option<Amx> {
        available("amx", Amx::detect())
    }

    /// 足し算・引き算・符号反転は、p を法として多倍長整数と一致する（結果はいつも 2^256 未満）
    #[rstest]
    fn add_sub_neg_match_bigint() {
        check(4096, (any_f(), any_f()), |(a, b)| {
            let (x, y, p) = (big(&a), big(&b), p());
            prop_assert_eq!(big(&a.add(b)) % &p, (&x + &y) % &p);
            prop_assert_eq!(big(&a.sub(b)) % &p, (&x + &p * 2u32 - &y % &p) % &p);
            prop_assert_eq!(big(&a.neg()) % &p, (&p - &x % &p) % &p);
            Ok(())
        });
    }

    /// 512 ビットの畳み込み結果を畳んでも、p を法として同じ値
    #[rstest]
    fn reduce_wide_matches_bigint() {
        check(4096, any::<[u64; 8]>(), |s| {
            let want = BigUint::from_slice(
                &s.iter()
                    .flat_map(|w| [*w as u32, (w >> 32) as u32])
                    .collect::<Vec<_>>(),
            ) % p();
            prop_assert_eq!(big(&F::reduce_wide(s)) % p(), want);
            Ok(())
        });
    }

    /// Fe との行き来で値が変わらない
    #[rstest]
    fn fe_round_trip() {
        check(4096, any_f(), |a| {
            let back = F::from_fe(a.to_fe());
            prop_assert_eq!(big(&back), big(&a) % p());
            Ok(())
        });
    }

    /// AMX の乗算（入力を書く → タイル演算 → 畳む）は、p を法として多倍長整数の積と一致する。
    /// 2組のタイルを交互に使うので、奇数個・偶数個の両方を流す
    #[rstest]
    fn mul_matches_bigint(#[values(1, 2, 7)] n: usize) {
        let Some(amx) = amx() else {
            return;
        };
        check(
            1024,
            prop::collection::vec((any_f(), any_f()), n),
            |pairs| {
                let tiles = Tiles::load(amx);
                let mut slots: Vec<Slot> = (0..n).map(|_| Slot::ZERO).collect();
                for (s, (x, y)) in slots.iter_mut().zip(&pairs) {
                    s.set(x, y);
                }
                tiles.run(&mut slots);
                for (s, (x, y)) in slots.iter().zip(&pairs) {
                    prop_assert_eq!(big(&s.product()) % p(), big(x) * big(y) % p());
                }
                Ok(())
            },
        );
    }

    /// 1・8・9・256 鍵（[`GROUP`] ちょうど・端数あり・バッチ全体）の倍算が curve25519-dalek と一致する
    #[rstest]
    fn mul_base_batch_matches_dalek(#[values(1, 8, 9, 256)] n: usize) {
        fn run<const N: usize>(amx: Amx, cases: u32) {
            check(
                cases,
                prop::collection::vec(any::<[u8; 32]>(), N),
                |random| {
                    let scalars: [[u8; 32]; N] = std::array::from_fn(|k| match k % 4 {
                        // 2^254（途中の桁がすべて 0）と、桁が正負に入り混じる値も混ぜる
                        1 => {
                            let mut s = [0u8; 32];
                            s[31] = 0x40;
                            s
                        }
                        2 => [0x88; 32].map(|b| b & 0x7f),
                        _ => clamp_integer(random[k]),
                    });
                    let mut out = [Extended::IDENTITY; N];
                    mul_base_batch(amx, &scalars, &mut out);
                    for (k, (p, s)) in out.iter().zip(&scalars).enumerate() {
                        let want = EdwardsPoint::mul_base(
                            &curve25519_dalek::Scalar::from_bytes_mod_order(*s),
                        )
                        .compress()
                        .to_bytes();
                        let got = crate::service::derive::comb::compress_with_zinv(p, p.z.invert());
                        prop_assert_eq!(got, want, "#{}", k);
                    }
                    Ok(())
                },
            );
        }
        let Some(amx) = amx() else {
            return;
        };
        match n {
            1 => run::<1>(amx, 32),
            8 => run::<8>(amx, 16),
            9 => run::<9>(amx, 16),
            _ => run::<256>(amx, 2),
        }
    }

    /// AMX が使えるなら AVX-512 も使える
    #[rstest]
    fn detection_is_consistent() {
        if let Some(a) = Amx::detect() {
            let _ = a.avx512();
            assert!(Avx512::detect().is_some());
        }
        let _ = amx();
    }
}
