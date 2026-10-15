// meikey の GPU カーネル（CUDA）。シードの生成から文字列の判定までを、1スレッドで鍵 MEIKEY_BATCH 本ずつ行う。
//
// 実行時に NVRTC でその GPU 向けにコンパイルする（service/gpu.rs）。テストでは同じソースを
// C++ として CPU 向けにコンパイルし（host.cpp）、Rust の実装と突き合わせる。そのため CUDA 固有の書き方は
// 下のマクロにまとめ、本体は C の整数演算だけで書く。
//
// 1鍵の流れ（CPU 版と同じ計算）:
//   1. シード = ChaCha20(鍵 = ホストが作った乱数, ブロック番号 = 鍵の通し番号 / 2) の前半か後半の 32 バイト
//      （通し番号が偶数なら前半、奇数なら後半。1 ブロックで 2 本ぶん）。低セキュリティモード（counter）では、
//      ホストが作った乱数 32 バイトの最後の 8 バイトに通し番号を足したもの（SHA-512 の最初の 3 ラウンドは
//      どの鍵でも同じなので、起動ごとに1回だけ計算しておく）
//   2. 秘密スカラー = clamp(SHA-512(シード)[..32])
//   3. 公開鍵の点 = 櫛形テーブル（幅 W、ホストが作って転送）を1行ずつ足す（倍加なし。W = 16 で加算 15 回）。
//      スカラー s = 8·(2^251 + m) の m を桁に分け、表は 8B を基点にして最上位の行に 2^254·B を足してある。
//      W は起動の引数で、桁は足しながら1つずつ取り出す。次の行の表の値は先に読み込みを始めておく
//   4. 最後の足し算は、y = Y/Z の Z が約分できるので、y の分子と分母（掛け算 3 回）だけにする
//   5. MEIKEY_BATCH 本の分母の積を、ブロックの中の 4 スレッド（ワープの同じレーン）でさらにまとめ、
//      逆元 1 回で全部の 1/分母 を求めて、1本ずつ y を圧縮する
//   6. 圧縮したらすぐ、公開鍵 blob の base64 の後半 44 文字に文字列があるか調べる。x の符号で変わる1文字は
//      どちらの値とも比べ（GPU では x を計算しない）、含まれそうなら通し番号を書いて知らせる。
//      本当に含まれるか（符号も合っているか）はホストが確かめる
//
// 体演算は MEIKEY_FIELD で選ぶ（どれが速いかは GPU で違うので、測って選ぶ）:
//   10: 25.5 ビット × 10 リム（偶数番目 26 ビット、奇数番目 25 ビット。AVX2 版 avx2.rs と同じ）。
//       32 × 32 → 64 ビットの積を 64 ビットで足し込む
//   8:  32 ビット × 8 リムを、桁上がりつきの命令で
//   26: 26 ビット × 10 リム（一様）。積は Karatsuba 法で 5 リムの積 3 回にする（掛け算は減り、足し算が増える）
//   51: 51 ビット × 5 リム。リムの積を倍精度浮動小数点数の FMA で求める（倍精度の速い A100・H100 向け）
// MEIKEY_UNPACKED が 1 なら、表をリムの形のまま持つ（10・26 だけ。読むときの変換が要らず、表は 1/3 大きい）。
//
// 10 リムのリムの上限（2^26・2^25 の何倍か）は AVX2 版と同じ:
//   fe_mul(a, b): a は 5 倍未満、b は 3.25 倍未満。結果は 1.01 倍未満
//   fe_sq(a): a は 1.5 倍未満（38a を 32 ビットに収めるため）。結果は fe_mul(a, a) とビット単位で同じ
//   fe_sub(a, b): b は 1.5 倍未満（2p を足してから引く）

#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
#define MEIKEY_CUDA 1
typedef unsigned char u8;
typedef int i32;
typedef unsigned int u32;
typedef unsigned long long u64;
#define DEV __device__ __forceinline__
#define NOINLINE __device__ __noinline__
#define CONSTANT __constant__
#define UNROLL _Pragma("unroll")
/// 展開しない（展開しても速くならず、コードだけ大きくなるループ）
#define NO_UNROLL _Pragma("unroll 1")
/// 16 バイト境界の4語を1回で読む（読み出し専用のキャッシュを通す）
#define LOAD4(dst, p)                                   \
    {                                                   \
        uint4 q_ = __ldg((const uint4*)(p));            \
        (dst)[0] = q_.x; (dst)[1] = q_.y; (dst)[2] = q_.z; (dst)[3] = q_.w; \
    }
/// 次に読むアドレスを L1 に読み込み始める（待たない）
#define PREFETCH(p) asm volatile("prefetch.global.L1 [%0];" ::"l"(p))
#else
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
typedef uint8_t u8;
typedef int32_t i32;
typedef uint32_t u32;
typedef unsigned long long u64;
#define DEV static inline
#define NOINLINE static
#define CONSTANT static const
#define UNROLL
#define NO_UNROLL
#define LOAD4(dst, p)                                                   \
    {                                                                   \
        (dst)[0] = (p)[0]; (dst)[1] = (p)[1]; (dst)[2] = (p)[2]; (dst)[3] = (p)[3]; \
    }
#define PREFETCH(p) ((void)(p))
#endif

// 体演算の種類（ファイル先頭を参照）
#ifndef MEIKEY_FIELD
#define MEIKEY_FIELD 10
#endif
static_assert(MEIKEY_FIELD == 10 || MEIKEY_FIELD == 8 || MEIKEY_FIELD == 26 || MEIKEY_FIELD == 51,
              "MEIKEY_FIELD は 10・8・26・51");
#ifndef MEIKEY_UNPACKED
#define MEIKEY_UNPACKED 0
#endif
static_assert(MEIKEY_UNPACKED == 0 || MEIKEY_FIELD == 10 || MEIKEY_FIELD == 26,
              "表をリムの形で持てるのは MEIKEY_FIELD が 10 か 26 のときだけ");

// 櫛形テーブル（service/derive/comb.rs と同じ作り方。幅 W は起動の引数）。桁は -2^(W-1)〜2^(W-1)。
// m は 251 ビットで、行数は ceil(251 / W)、列数は 2^(W-1) + 1（comb.rs の rows_cols と同じ）
#define M_BITS 251
#define MIN_W 2
#define MAX_W 20
// 表の1エントリ: y+x, y-x, 2dxy（1 行目は 2xy）を 0 以上 p 未満にして、32 ビット × 8 語ずつ並べたもの（96 バイト）か、
// リムの形（10 リム × 3 と余白 2 語。128 バイト）
#if MEIKEY_UNPACKED
#define ENTRY_WORDS 32
#define FE_WORDS 10
#else
#define ENTRY_WORDS 24
#define FE_WORDS 8
#endif

// 1ブロックのスレッド数（service/gpu.rs の BLOCK と同じ）
#define MEIKEY_BLOCK 128
// 逆元をまとめるスレッドの数（ブロックの中の、ワープの同じレーン）
#define WARPS (MEIKEY_BLOCK / 32)

// 1スレッドが1回にまとめて計算する鍵の数（逆元をまとめる数。当たりを 64 ビットに集めるので 64 まで）
#ifndef MEIKEY_BATCH
#define MEIKEY_BATCH 32
#endif
static_assert(MEIKEY_BATCH >= 1 && MEIKEY_BATCH <= 64, "MEIKEY_BATCH は 1〜64");

// 公開鍵の base64（68 文字）のうち、鍵で変わる後半 44 文字（blob の 18 バイト目から）と、
// そのうち判定に使う範囲の先頭（FIXED_PREFIX の直後 = 全体の 25 文字目 = 後半の 1 文字目）
#define TAIL_B64 44
#define SEARCH_FROM 1
#define MAX_NEEDLE 43

// ---------------------------------------------------------------------------
// ChaCha20（RFC 8439 のブロック関数。カウンタは 64 ビットで、rand の ChaCha20Rng と同じ並び）
// ---------------------------------------------------------------------------

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define QR(a, b, c, d)            \
    a += b; d ^= a; d = ROTL32(d, 16); \
    c += d; b ^= c; b = ROTL32(b, 12); \
    a += b; d ^= a; d = ROTL32(d, 8);  \
    c += d; b ^= c; b = ROTL32(b, 7);

/// 鍵 `key`、ブロック番号 `counter`（ストリーム 0）の出力 64 バイト（16 語）
DEV void chacha20_block(const u32 key[8], u64 counter, u32 out[16]) {
    u32 s[16] = {
        0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u,
        key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
        (u32)counter, (u32)(counter >> 32), 0u, 0u,
    };
    u32 x[16];
    UNROLL
    for (int i = 0; i < 16; i++) x[i] = s[i];
    UNROLL
    for (int r = 0; r < 10; r++) {
        QR(x[0], x[4], x[8], x[12]);
        QR(x[1], x[5], x[9], x[13]);
        QR(x[2], x[6], x[10], x[14]);
        QR(x[3], x[7], x[11], x[15]);
        QR(x[0], x[5], x[10], x[15]);
        QR(x[1], x[6], x[11], x[12]);
        QR(x[2], x[7], x[8], x[13]);
        QR(x[3], x[4], x[9], x[14]);
    }
    UNROLL
    for (int i = 0; i < 16; i++) out[i] = x[i] + s[i];
}

/// 32 ビットの語 w をリトルエンディアンの 4 バイトに
DEV void put_le32(u8* out, u32 w) {
    out[0] = (u8)w;
    out[1] = (u8)(w >> 8);
    out[2] = (u8)(w >> 16);
    out[3] = (u8)(w >> 24);
}

/// ブロック `block` の前半（`upper` なら後半）の 32 バイトをシードに
DEV void seed_from_block(const u32 block[16], bool upper, u8 seed[32]) {
    UNROLL
    for (int i = 0; i < 8; i++) put_le32(seed + 4 * i, upper ? block[8 + i] : block[i]);
}

/// 通し番号 `index` のシード（ブロック index / 2 の、index が偶数なら前半・奇数なら後半）
DEV void chacha20_seed(const u32 key[8], u64 index, u8 seed[32]) {
    u32 block[16];
    chacha20_block(key, index >> 1, block);
    seed_from_block(block, (index & 1) != 0, seed);
}

/// 低セキュリティモードのシードの最後の 8 バイト（リトルエンディアン）: `key` の最後の 8 バイトに通し番号を足したもの
DEV u64 counter_tail(const u32 key[8], u64 index) { return ((u64)key[7] << 32 | key[6]) + index; }

/// 低セキュリティモードのシード: `key`（ホストが作った乱数 32 バイト）の最後の 8 バイト（リトルエンディアン）に
/// 通し番号を足したもの（model::search の counter_seed と同じ）
DEV void counter_seed(const u32 key[8], u64 index, u8 seed[32]) {
    u64 tail = counter_tail(key, index);
    UNROLL
    for (int i = 0; i < 6; i++) put_le32(seed + 4 * i, key[i]);
    put_le32(seed + 24, (u32)tail);
    put_le32(seed + 28, (u32)(tail >> 32));
}

// ---------------------------------------------------------------------------
// SHA-512（32 バイトのメッセージ1ブロック専用。出力は前半 32 バイトだけ）
// ---------------------------------------------------------------------------

CONSTANT u64 SHA512_K[80] = {
    0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull,
    0x3956c25bf348b538ull, 0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull,
    0xd807aa98a3030242ull, 0x12835b0145706fbeull, 0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull,
    0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull, 0xc19bf174cf692694ull,
    0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,
    0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull,
    0x983e5152ee66dfabull, 0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull,
    0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull, 0x06ca6351e003826full, 0x142929670a0e6e70ull,
    0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull, 0x53380d139d95b3dfull,
    0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,
    0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull,
    0xd192e819d6ef5218ull, 0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull,
    0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull, 0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull,
    0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull, 0x682e6ff3d6b2b8a3ull,
    0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
    0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull,
    0xca273eceea26619cull, 0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull,
    0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull, 0x113f9804bef90daeull, 0x1b710b35131c471bull,
    0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull, 0x431d67c49c100d4cull,
    0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull,
};

CONSTANT u64 SHA512_IV[8] = {
    0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
    0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull,
};

#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
#define SIG0(x) (ROTR64(x, 1) ^ ROTR64(x, 8) ^ ((x) >> 7))
#define SIG1(x) (ROTR64(x, 19) ^ ROTR64(x, 61) ^ ((x) >> 6))

/// SHA-512 の1ラウンド（状態 s = a..h、メッセージの語 wt、定数 kt）
DEV void sha512_round(u64 s[8], u64 wt, u64 kt) {
    u64 S1 = ROTR64(s[4], 14) ^ ROTR64(s[4], 18) ^ ROTR64(s[4], 41);
    u64 ch = (s[4] & s[5]) ^ (~s[4] & s[6]);
    u64 t1 = s[7] + S1 + ch + kt + wt;
    u64 S0 = ROTR64(s[0], 28) ^ ROTR64(s[0], 34) ^ ROTR64(s[0], 39);
    u64 maj = (s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]);
    u64 t2 = S0 + maj;
    s[7] = s[6]; s[6] = s[5]; s[5] = s[4]; s[4] = s[3] + t1;
    s[3] = s[2]; s[2] = s[1]; s[1] = s[0]; s[0] = t1 + t2;
}

/// 圧縮関数の出力の前半 32 バイト（状態 + IV）をクランプして秘密スカラーに
DEV void sha512_finish(const u64 s[8], u8 out[32]) {
    UNROLL
    for (int i = 0; i < 4; i++) {
        u64 h = s[i] + SHA512_IV[i];
        UNROLL
        for (int k = 0; k < 8; k++) out[8 * i + k] = (u8)(h >> (56 - 8 * k));
    }
    out[0] &= 248;
    out[31] &= 127;
    out[31] |= 64;
}

/// メッセージの語 w[0..4) から、ラウンド `from` 以降（状態 s はラウンド from の前のもの）を計算する。
/// w16・w17 は 16・17 番目の語（前計算したもの。from が 0 なら使わずに計算する）
DEV void sha512_rounds(u64 s[8], const u64 w4[4], int from, u64 w16, u64 w17, u8 out[32]) {
    u64 w[16];
    UNROLL
    for (int i = 0; i < 4; i++) w[i] = w4[i];
    w[4] = 0x8000000000000000ull;
    UNROLL
    for (int i = 5; i < 15; i++) w[i] = 0;
    w[15] = 256;
    // 展開すると w の添字が定数になり、w をレジスタに置ける（展開しないとローカルメモリに置かれ、
    // ラウンドごとに読み書きと分岐が入る）。w[4..16] の定数も畳み込まれる
    UNROLL
    for (int t = 0; t < 80; t++) {
        if (t < from) continue;
        u64 wt;
        if (t < 16) {
            wt = w[t];
        } else if (t == 16 && from > 0) {
            wt = w16;
            w[0] = wt;
        } else if (t == 17 && from > 0) {
            wt = w17;
            w[1] = wt;
        } else {
            u64 w15 = w[(t + 1) & 15], w2 = w[(t + 14) & 15];
            wt = w[t & 15] + SIG0(w15) + w[(t + 9) & 15] + SIG1(w2);
            w[t & 15] = wt;
        }
        sha512_round(s, wt, SHA512_K[t]);
    }
    sha512_finish(s, out);
}

/// 8 バイトをビッグエンディアンの 64 ビットに
DEV u64 load_be64(const u8* p) {
    u64 v = 0;
    UNROLL
    for (int b = 0; b < 8; b++) v = (v << 8) | p[b];
    return v;
}

/// `clamp(SHA-512(seed)[..32])`（RFC 8032 5.1.5 の手順 1・2）
DEV void secret_scalar(const u8 seed[32], u8 out[32]) {
    u64 w4[4];
    UNROLL
    for (int i = 0; i < 4; i++) w4[i] = load_be64(seed + 8 * i);
    u64 s[8];
    UNROLL
    for (int i = 0; i < 8; i++) s[i] = SHA512_IV[i];
    sha512_rounds(s, w4, 0, 0, 0, out);
}

/// 低セキュリティモードで、どの鍵でも同じ部分の SHA-512（シードの前半 24 バイトはどの鍵でも同じで、
/// 通し番号は最後の 8 バイト = メッセージの 4 語目にだけ入る）:
/// ラウンド 0〜2 の後の状態、ラウンド 3 の t1 から w3 を除いたものと t2、メッセージの 16・17 番目の語
typedef struct {
    u64 s[8];
    u64 t1, t2;
    u64 w[3];
    u64 w16, w17;
} ShaMid;

DEV ShaMid sha512_mid(const u32 key[8]) {
    ShaMid m;
    UNROLL
    for (int i = 0; i < 3; i++) {
        // シードの 8i〜8i+7 バイト目（key のリトルエンディアンの語 2i・2i+1）をビッグエンディアンで読む
        u8 b[8];
        put_le32(b, key[2 * i]);
        put_le32(b + 4, key[2 * i + 1]);
        m.w[i] = load_be64(b);
    }
    UNROLL
    for (int i = 0; i < 8; i++) m.s[i] = SHA512_IV[i];
    UNROLL
    for (int t = 0; t < 3; t++) sha512_round(m.s, m.w[t], SHA512_K[t]);
    // ラウンド 3: t1 = h + Σ1(e) + Ch(e, f, g) + K[3] + w3、t2 = Σ0(a) + Maj(a, b, c)
    u64 S1 = ROTR64(m.s[4], 14) ^ ROTR64(m.s[4], 18) ^ ROTR64(m.s[4], 41);
    u64 ch = (m.s[4] & m.s[5]) ^ (~m.s[4] & m.s[6]);
    m.t1 = m.s[7] + S1 + ch + SHA512_K[3];
    u64 S0 = ROTR64(m.s[0], 28) ^ ROTR64(m.s[0], 34) ^ ROTR64(m.s[0], 39);
    m.t2 = S0 + ((m.s[0] & m.s[1]) ^ (m.s[0] & m.s[2]) ^ (m.s[1] & m.s[2]));
    // w16 = σ1(w14) + w9 + σ0(w1) + w0、w17 = σ1(w15) + w10 + σ0(w2) + w1（w9・w10・w14 は 0、w15 は 256）
    m.w16 = SIG0(m.w[1]) + m.w[0];
    m.w17 = SIG1(256ull) + SIG0(m.w[2]) + m.w[1];
    return m;
}

/// 低セキュリティモードの通し番号 `index` の秘密スカラー（`m` は sha512_mid(key)。counter_seed のシードの
/// secret_scalar と同じ）
DEV void secret_scalar_mid(const ShaMid& m, const u32 key[8], u64 index, u8 out[32]) {
    u64 tail = counter_tail(key, index);
    u8 b[8];
    put_le32(b, (u32)tail);
    put_le32(b + 4, (u32)(tail >> 32));
    u64 w3 = load_be64(b);
    u64 s[8];
    // ラウンド 3 は前計算の残りだけ
    u64 t1 = m.t1 + w3;
    s[0] = t1 + m.t2;
    s[1] = m.s[0];
    s[2] = m.s[1];
    s[3] = m.s[2];
    s[4] = m.s[3] + t1;
    s[5] = m.s[4];
    s[6] = m.s[5];
    s[7] = m.s[6];
    u64 w4[4] = {m.w[0], m.w[1], m.w[2], w3};
    sha512_rounds(s, w4, 4, m.w16, m.w17, out);
}

/// 0 以上 2^256 未満の値（32 ビット × 8 語、リトルエンディアン）を、0 以上 p 未満に正規化したリトルエンディアン 32 バイトに
DEV void words_to_bytes(const u32 a[8], u8 out[32]) {
    // 2^255 以上の分を 19 倍して下に足す（2^255 ≡ 19）。これで 2^255 + 19 未満
    u32 w[8], t[8];
    u64 c = (u64)(a[7] >> 31) * 19;
    UNROLL
    for (int i = 0; i < 8; i++) {
        c += i == 7 ? (a[7] & 0x7fffffffu) : a[i];
        w[i] = (u32)c;
        c >>= 32;
    }
    // p 以上なら p を引く（w + 19 が 2^255 以上かで判定し、そのときは w + 19 - 2^255）
    c = 19;
    UNROLL
    for (int i = 0; i < 8; i++) {
        c += w[i];
        t[i] = (u32)c;
        c >>= 32;
    }
    u32 mask = 0u - (t[7] >> 31);
    t[7] &= 0x7fffffffu;
    UNROLL
    for (int i = 0; i < 8; i++) {
        u32 v = (w[i] & ~mask) | (t[i] & mask);
        UNROLL
        for (int k = 0; k < 4; k++) out[4 * i + k] = (u8)(v >> (8 * k));
    }
}

// ---------------------------------------------------------------------------
// 体 GF(2^255 - 19)（MEIKEY_FIELD で選ぶ。ファイル先頭を参照）。どれも同じ関数をそろえる:
// fe_add・fe_sub・fe_mul・fe_sq・fe_carry（整える）・fe_unpack（表の 32 ビット × 8 語から）・fe_load（表のリムの形から。
// 10・26 だけ）・fe_to_bytes（正規化したバイト列に）・fe_small（小さい整数）
// ---------------------------------------------------------------------------

#if MEIKEY_FIELD == 10

// 25.5 ビット × 10 リム（偶数番目 26 ビット、奇数番目 25 ビット）。リムの上限はファイル先頭のとおり

typedef struct { u32 v[10]; } fe;

#define MASK26 0x3ffffffu
#define MASK25 0x1ffffffu

/// 2p のリム（引き算で負にならないように足す）
CONSTANT u32 FE_P2[10] = {
    2 * (MASK26 - 18), 2 * MASK25, 2 * MASK26, 2 * MASK25, 2 * MASK26,
    2 * MASK25, 2 * MASK26, 2 * MASK25, 2 * MASK26, 2 * MASK25,
};

DEV fe fe_add(const fe a, const fe b) {
    fe r;
    UNROLL
    for (int i = 0; i < 10; i++) r.v[i] = a.v[i] + b.v[i];
    return r;
}

/// a + 2p - b
DEV fe fe_sub(const fe a, const fe b) {
    fe r;
    UNROLL
    for (int i = 0; i < 10; i++) r.v[i] = a.v[i] + FE_P2[i] - b.v[i];
    return r;
}

/// 64 ビットのリムを 26/25 ビット + α に収める（ref10・avx2.rs と同じ順に、2本の列を交互に進める）
#define CARRY_STEP(c, k)                                              \
    {                                                                 \
        u64 hi = (c)[k] >> (((k) & 1) ? 25 : 26);                     \
        (c)[k] &= ((k) & 1) ? MASK25 : MASK26;                        \
        if ((k) == 9) (c)[0] += hi * 19; else (c)[((k) + 1) % 10] += hi; \
    }

DEV fe fe_carry64(u64 c[10]) {
    CARRY_STEP(c, 0); CARRY_STEP(c, 4); CARRY_STEP(c, 1); CARRY_STEP(c, 5);
    CARRY_STEP(c, 2); CARRY_STEP(c, 6); CARRY_STEP(c, 3); CARRY_STEP(c, 7);
    CARRY_STEP(c, 4); CARRY_STEP(c, 8); CARRY_STEP(c, 9); CARRY_STEP(c, 0);
    fe r;
    UNROLL
    for (int i = 0; i < 10; i++) r.v[i] = (u32)c[i];
    return r;
}

DEV fe fe_carry(const fe a) {
    u64 c[10];
    UNROLL
    for (int i = 0; i < 10; i++) c[i] = a.v[i];
    return fe_carry64(c);
}

/// a·b。c[k] = Σ a[i]·b[k-i]（k-i < 0 なら 19·b[k-i+10]）。奇数番目どうしの積は半ビットずれるので 2 倍する
DEV fe fe_mul(const fe a, const fe b) {
    u32 b19[10];
    UNROLL
    for (int j = 1; j < 10; j++) b19[j] = b.v[j] * 19;
    b19[0] = 0;
    u64 c[10];
    UNROLL
    for (int k = 0; k < 10; k++) c[k] = 0;
    UNROLL
    for (int i = 0; i < 10; i++) {
        UNROLL
        for (int k = 0; k < 10; k++) {
            int j = k - i;
            u32 bb = j >= 0 ? b.v[j] : b19[j + 10];
            u32 aa = ((i & 1) && !(k & 1)) ? 2 * a.v[i] : a.v[i];
            c[k] += (u64)aa * bb;
        }
    }
    return fe_carry64(c);
}

/// a²。fe_mul(a, a) の c[k] の項を i ≤ j の組にまとめる（積は 100 回から 55 回に減る）。
/// 係数（i ≠ j で 2 倍、i・j とも奇数で 2 倍、i + j ≥ 10 で 19 倍）は 2a・19a・38a を先に作って掛ける
/// （a が 1.5 倍未満なら 38a は 32 ビットに収まる）。c[k] は fe_mul(a, a) と同じ値なので、結果もビット単位で同じ
DEV fe fe_sq(const fe a) {
    u32 a2[10], a19[10], a38[10];
    UNROLL
    for (int i = 0; i < 10; i++) {
        a2[i] = 2 * a.v[i];
        a19[i] = 19 * a.v[i];
        a38[i] = 38 * a.v[i];
    }
    u64 c[10];
    UNROLL
    for (int k = 0; k < 10; k++) c[k] = 0;
    UNROLL
    for (int i = 0; i < 10; i++) {
        UNROLL
        for (int j = i; j < 10; j++) {
            bool wrap = i + j >= 10;
            // 1・2・4 倍（19 倍は下で掛ける）
            int coef = (i != j ? 2 : 1) * ((i & 1) && (j & 1) ? 2 : 1);
            u32 l = coef >= 2 ? a2[i] : a.v[i];
            u32 r = wrap ? (coef == 4 ? a38[j] : a19[j]) : (coef == 4 ? a2[j] : a.v[j]);
            c[(i + j) % 10] += (u64)l * r;
        }
    }
    return fe_carry64(c);
}

#define MASK51 0x7ffffffffffffull

/// 32 ビット × 8 語（リトルエンディアン、0 以上 2^255 未満）を 25.5 ビット × 10 リムに
/// （i 番目のリムは floor((51i + 1) / 2) ビット目から 26・25 ビット）
DEV fe fe_unpack(const u32 w[8]) {
    fe r;
    UNROLL
    for (int i = 0; i < 10; i++) {
        const int o = (51 * i + 1) / 2, q = o / 32, b = o % 32;
        u64 v = (u64)w[q] | (q + 1 < 8 ? (u64)w[q + 1] << 32 : 0ull);
        r.v[i] = (u32)(v >> b) & ((i & 1) ? MASK25 : MASK26);
    }
    return r;
}

/// 0 以上 p 未満に正規化したリトルエンディアン 32 バイト（field.rs の to_bytes と同じ。a は fe_mul の結果）
DEV void fe_to_bytes(const fe a, u8 out[32]) {
    u64 l[5];
    UNROLL
    for (int i = 0; i < 5; i++) l[i] = (u64)a.v[2 * i] + ((u64)a.v[2 * i + 1] << 26);
    // 2^51 + α に収める
    u64 c;
    c = l[0] >> 51; l[0] &= MASK51; l[1] += c;
    c = l[1] >> 51; l[1] &= MASK51; l[2] += c;
    c = l[2] >> 51; l[2] &= MASK51; l[3] += c;
    c = l[3] >> 51; l[3] &= MASK51; l[4] += c;
    c = l[4] >> 51; l[4] &= MASK51; l[0] += c * 19;
    // l < 2p なので、p 以上なら p を引く（l + 19 が 2^255 を超えるかで判定）
    u64 q = (l[0] + 19) >> 51;
    q = (l[1] + q) >> 51;
    q = (l[2] + q) >> 51;
    q = (l[3] + q) >> 51;
    q = (l[4] + q) >> 51;
    l[0] += 19 * q;
    l[1] += l[0] >> 51; l[0] &= MASK51;
    l[2] += l[1] >> 51; l[1] &= MASK51;
    l[3] += l[2] >> 51; l[2] &= MASK51;
    l[4] += l[3] >> 51; l[3] &= MASK51;
    l[4] &= MASK51;
    u64 w[4] = {
        l[0] | l[1] << 51,
        l[1] >> 13 | l[2] << 38,
        l[2] >> 26 | l[3] << 25,
        l[3] >> 39 | l[4] << 12,
    };
    UNROLL
    for (int i = 0; i < 4; i++) {
        UNROLL
        for (int k = 0; k < 8; k++) out[8 * i + k] = (u8)(w[i] >> (8 * k));
    }
}

/// 表のリムの形のエントリ（10 語）
DEV fe fe_load(const u32 w[10]) {
    fe r;
    UNROLL
    for (int i = 0; i < 10; i++) r.v[i] = w[i];
    return r;
}

DEV fe fe_small(u32 v) {
    fe r;
    UNROLL
    for (int i = 0; i < 10; i++) r.v[i] = 0;
    r.v[0] = v;
    return r;
}

#elif MEIKEY_FIELD == 8

/// 0 以上 2^256 未満の値（p 以上のこともある）を 32 ビット × 8 リム（リトルエンディアン）で持つ。
/// 結果も 2^256 未満に収め、正規化（p 未満にする）は fe_to_bytes だけでする
typedef struct { u32 v[8]; } fe;

#ifdef MEIKEY_CUDA
// GPU では桁上がりつきの命令（add.cc・addc・mad.lo.cc・madc.hi.cc など）で計算する。
// 桁上がりのフラグは asm の中だけで使うので、1つの演算を1つの asm にまとめる

/// a + b（2^256 を超えたら 38 を足す。2^256 ≡ 38）
DEV fe fe_add(const fe a, const fe b) {
    fe r;
    asm(
        "{\n\t"
        ".reg .u32 c;\n\t"
        "add.cc.u32 %0, %8, %16;\n\t"
        "addc.cc.u32 %1, %9, %17;\n\t"
        "addc.cc.u32 %2, %10, %18;\n\t"
        "addc.cc.u32 %3, %11, %19;\n\t"
        "addc.cc.u32 %4, %12, %20;\n\t"
        "addc.cc.u32 %5, %13, %21;\n\t"
        "addc.cc.u32 %6, %14, %22;\n\t"
        "addc.cc.u32 %7, %15, %23;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mul.lo.u32 c, c, 38;\n\t"
        "add.cc.u32 %0, %0, c;\n\t"
        "addc.cc.u32 %1, %1, 0;\n\t"
        "addc.cc.u32 %2, %2, 0;\n\t"
        "addc.cc.u32 %3, %3, 0;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\t"
        "addc.cc.u32 %5, %5, 0;\n\t"
        "addc.cc.u32 %6, %6, 0;\n\t"
        "addc.cc.u32 %7, %7, 0;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mad.lo.u32 %0, c, 38, %0;\n\t"
        "}"
        : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
        : "r"(a.v[0]), "r"(a.v[1]), "r"(a.v[2]), "r"(a.v[3]), "r"(a.v[4]), "r"(a.v[5]), "r"(a.v[6]), "r"(a.v[7]), "r"(b.v[0]), "r"(b.v[1]), "r"(b.v[2]), "r"(b.v[3]), "r"(b.v[4]), "r"(b.v[5]), "r"(b.v[6]), "r"(b.v[7]));
    return r;
}

/// a - b（負になったら 38 を引く。もう一度負になったら、さらに 38 を引く）
DEV fe fe_sub(const fe a, const fe b) {
    fe r;
    asm(
        "{\n\t"
        ".reg .u32 c;\n\t"
        "sub.cc.u32 %0, %8, %16;\n\t"
        "subc.cc.u32 %1, %9, %17;\n\t"
        "subc.cc.u32 %2, %10, %18;\n\t"
        "subc.cc.u32 %3, %11, %19;\n\t"
        "subc.cc.u32 %4, %12, %20;\n\t"
        "subc.cc.u32 %5, %13, %21;\n\t"
        "subc.cc.u32 %6, %14, %22;\n\t"
        "subc.cc.u32 %7, %15, %23;\n\t"
        "subc.u32 c, 0, 0;\n\t"
        "and.b32 c, c, 38;\n\t"
        "sub.cc.u32 %0, %0, c;\n\t"
        "subc.cc.u32 %1, %1, 0;\n\t"
        "subc.cc.u32 %2, %2, 0;\n\t"
        "subc.cc.u32 %3, %3, 0;\n\t"
        "subc.cc.u32 %4, %4, 0;\n\t"
        "subc.cc.u32 %5, %5, 0;\n\t"
        "subc.cc.u32 %6, %6, 0;\n\t"
        "subc.cc.u32 %7, %7, 0;\n\t"
        "subc.u32 c, 0, 0;\n\t"
        "and.b32 c, c, 38;\n\t"
        "sub.u32 %0, %0, c;\n\t"
        "}"
        : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
        : "r"(a.v[0]), "r"(a.v[1]), "r"(a.v[2]), "r"(a.v[3]), "r"(a.v[4]), "r"(a.v[5]), "r"(a.v[6]), "r"(a.v[7]), "r"(b.v[0]), "r"(b.v[1]), "r"(b.v[2]), "r"(b.v[3]), "r"(b.v[4]), "r"(b.v[5]), "r"(b.v[6]), "r"(b.v[7]));
    return r;
}

/// a·b。8 × 8 語の積（512 ビット、t0〜t15）を、行ごとに下位・上位の桁上がりの鎖で足し、
/// 上の 256 ビットを 38 倍して下に足す
DEV fe fe_mul(const fe a, const fe b) {
    fe r;
    asm(
        "{\n\t"
        ".reg .u32 c, t<17>;\n\t"
        "mov.u32 t0, 0;\n\t"
        "mov.u32 t1, 0;\n\t"
        "mov.u32 t2, 0;\n\t"
        "mov.u32 t3, 0;\n\t"
        "mov.u32 t4, 0;\n\t"
        "mov.u32 t5, 0;\n\t"
        "mov.u32 t6, 0;\n\t"
        "mov.u32 t7, 0;\n\t"
        "mov.u32 t8, 0;\n\t"
        "mov.u32 t9, 0;\n\t"
        "mov.u32 t10, 0;\n\t"
        "mov.u32 t11, 0;\n\t"
        "mov.u32 t12, 0;\n\t"
        "mov.u32 t13, 0;\n\t"
        "mov.u32 t14, 0;\n\t"
        "mov.u32 t15, 0;\n\t"
        "mov.u32 t16, 0;\n\t"
        "mad.lo.cc.u32 t0, %8, %16, t0;\n\t"
        "madc.lo.cc.u32 t1, %8, %17, t1;\n\t"
        "madc.lo.cc.u32 t2, %8, %18, t2;\n\t"
        "madc.lo.cc.u32 t3, %8, %19, t3;\n\t"
        "madc.lo.cc.u32 t4, %8, %20, t4;\n\t"
        "madc.lo.cc.u32 t5, %8, %21, t5;\n\t"
        "madc.lo.cc.u32 t6, %8, %22, t6;\n\t"
        "madc.lo.cc.u32 t7, %8, %23, t7;\n\t"
        "addc.u32 t8, t8, 0;\n\t"
        "mad.hi.cc.u32 t1, %8, %16, t1;\n\t"
        "madc.hi.cc.u32 t2, %8, %17, t2;\n\t"
        "madc.hi.cc.u32 t3, %8, %18, t3;\n\t"
        "madc.hi.cc.u32 t4, %8, %19, t4;\n\t"
        "madc.hi.cc.u32 t5, %8, %20, t5;\n\t"
        "madc.hi.cc.u32 t6, %8, %21, t6;\n\t"
        "madc.hi.cc.u32 t7, %8, %22, t7;\n\t"
        "madc.hi.cc.u32 t8, %8, %23, t8;\n\t"
        "addc.u32 t9, 0, 0;\n\t"
        "mad.lo.cc.u32 t1, %9, %16, t1;\n\t"
        "madc.lo.cc.u32 t2, %9, %17, t2;\n\t"
        "madc.lo.cc.u32 t3, %9, %18, t3;\n\t"
        "madc.lo.cc.u32 t4, %9, %19, t4;\n\t"
        "madc.lo.cc.u32 t5, %9, %20, t5;\n\t"
        "madc.lo.cc.u32 t6, %9, %21, t6;\n\t"
        "madc.lo.cc.u32 t7, %9, %22, t7;\n\t"
        "madc.lo.cc.u32 t8, %9, %23, t8;\n\t"
        "addc.u32 t9, t9, 0;\n\t"
        "mad.hi.cc.u32 t2, %9, %16, t2;\n\t"
        "madc.hi.cc.u32 t3, %9, %17, t3;\n\t"
        "madc.hi.cc.u32 t4, %9, %18, t4;\n\t"
        "madc.hi.cc.u32 t5, %9, %19, t5;\n\t"
        "madc.hi.cc.u32 t6, %9, %20, t6;\n\t"
        "madc.hi.cc.u32 t7, %9, %21, t7;\n\t"
        "madc.hi.cc.u32 t8, %9, %22, t8;\n\t"
        "madc.hi.cc.u32 t9, %9, %23, t9;\n\t"
        "addc.u32 t10, 0, 0;\n\t"
        "mad.lo.cc.u32 t2, %10, %16, t2;\n\t"
        "madc.lo.cc.u32 t3, %10, %17, t3;\n\t"
        "madc.lo.cc.u32 t4, %10, %18, t4;\n\t"
        "madc.lo.cc.u32 t5, %10, %19, t5;\n\t"
        "madc.lo.cc.u32 t6, %10, %20, t6;\n\t"
        "madc.lo.cc.u32 t7, %10, %21, t7;\n\t"
        "madc.lo.cc.u32 t8, %10, %22, t8;\n\t"
        "madc.lo.cc.u32 t9, %10, %23, t9;\n\t"
        "addc.u32 t10, t10, 0;\n\t"
        "mad.hi.cc.u32 t3, %10, %16, t3;\n\t"
        "madc.hi.cc.u32 t4, %10, %17, t4;\n\t"
        "madc.hi.cc.u32 t5, %10, %18, t5;\n\t"
        "madc.hi.cc.u32 t6, %10, %19, t6;\n\t"
        "madc.hi.cc.u32 t7, %10, %20, t7;\n\t"
        "madc.hi.cc.u32 t8, %10, %21, t8;\n\t"
        "madc.hi.cc.u32 t9, %10, %22, t9;\n\t"
        "madc.hi.cc.u32 t10, %10, %23, t10;\n\t"
        "addc.u32 t11, 0, 0;\n\t"
        "mad.lo.cc.u32 t3, %11, %16, t3;\n\t"
        "madc.lo.cc.u32 t4, %11, %17, t4;\n\t"
        "madc.lo.cc.u32 t5, %11, %18, t5;\n\t"
        "madc.lo.cc.u32 t6, %11, %19, t6;\n\t"
        "madc.lo.cc.u32 t7, %11, %20, t7;\n\t"
        "madc.lo.cc.u32 t8, %11, %21, t8;\n\t"
        "madc.lo.cc.u32 t9, %11, %22, t9;\n\t"
        "madc.lo.cc.u32 t10, %11, %23, t10;\n\t"
        "addc.u32 t11, t11, 0;\n\t"
        "mad.hi.cc.u32 t4, %11, %16, t4;\n\t"
        "madc.hi.cc.u32 t5, %11, %17, t5;\n\t"
        "madc.hi.cc.u32 t6, %11, %18, t6;\n\t"
        "madc.hi.cc.u32 t7, %11, %19, t7;\n\t"
        "madc.hi.cc.u32 t8, %11, %20, t8;\n\t"
        "madc.hi.cc.u32 t9, %11, %21, t9;\n\t"
        "madc.hi.cc.u32 t10, %11, %22, t10;\n\t"
        "madc.hi.cc.u32 t11, %11, %23, t11;\n\t"
        "addc.u32 t12, 0, 0;\n\t"
        "mad.lo.cc.u32 t4, %12, %16, t4;\n\t"
        "madc.lo.cc.u32 t5, %12, %17, t5;\n\t"
        "madc.lo.cc.u32 t6, %12, %18, t6;\n\t"
        "madc.lo.cc.u32 t7, %12, %19, t7;\n\t"
        "madc.lo.cc.u32 t8, %12, %20, t8;\n\t"
        "madc.lo.cc.u32 t9, %12, %21, t9;\n\t"
        "madc.lo.cc.u32 t10, %12, %22, t10;\n\t"
        "madc.lo.cc.u32 t11, %12, %23, t11;\n\t"
        "addc.u32 t12, t12, 0;\n\t"
        "mad.hi.cc.u32 t5, %12, %16, t5;\n\t"
        "madc.hi.cc.u32 t6, %12, %17, t6;\n\t"
        "madc.hi.cc.u32 t7, %12, %18, t7;\n\t"
        "madc.hi.cc.u32 t8, %12, %19, t8;\n\t"
        "madc.hi.cc.u32 t9, %12, %20, t9;\n\t"
        "madc.hi.cc.u32 t10, %12, %21, t10;\n\t"
        "madc.hi.cc.u32 t11, %12, %22, t11;\n\t"
        "madc.hi.cc.u32 t12, %12, %23, t12;\n\t"
        "addc.u32 t13, 0, 0;\n\t"
        "mad.lo.cc.u32 t5, %13, %16, t5;\n\t"
        "madc.lo.cc.u32 t6, %13, %17, t6;\n\t"
        "madc.lo.cc.u32 t7, %13, %18, t7;\n\t"
        "madc.lo.cc.u32 t8, %13, %19, t8;\n\t"
        "madc.lo.cc.u32 t9, %13, %20, t9;\n\t"
        "madc.lo.cc.u32 t10, %13, %21, t10;\n\t"
        "madc.lo.cc.u32 t11, %13, %22, t11;\n\t"
        "madc.lo.cc.u32 t12, %13, %23, t12;\n\t"
        "addc.u32 t13, t13, 0;\n\t"
        "mad.hi.cc.u32 t6, %13, %16, t6;\n\t"
        "madc.hi.cc.u32 t7, %13, %17, t7;\n\t"
        "madc.hi.cc.u32 t8, %13, %18, t8;\n\t"
        "madc.hi.cc.u32 t9, %13, %19, t9;\n\t"
        "madc.hi.cc.u32 t10, %13, %20, t10;\n\t"
        "madc.hi.cc.u32 t11, %13, %21, t11;\n\t"
        "madc.hi.cc.u32 t12, %13, %22, t12;\n\t"
        "madc.hi.cc.u32 t13, %13, %23, t13;\n\t"
        "addc.u32 t14, 0, 0;\n\t"
        "mad.lo.cc.u32 t6, %14, %16, t6;\n\t"
        "madc.lo.cc.u32 t7, %14, %17, t7;\n\t"
        "madc.lo.cc.u32 t8, %14, %18, t8;\n\t"
        "madc.lo.cc.u32 t9, %14, %19, t9;\n\t"
        "madc.lo.cc.u32 t10, %14, %20, t10;\n\t"
        "madc.lo.cc.u32 t11, %14, %21, t11;\n\t"
        "madc.lo.cc.u32 t12, %14, %22, t12;\n\t"
        "madc.lo.cc.u32 t13, %14, %23, t13;\n\t"
        "addc.u32 t14, t14, 0;\n\t"
        "mad.hi.cc.u32 t7, %14, %16, t7;\n\t"
        "madc.hi.cc.u32 t8, %14, %17, t8;\n\t"
        "madc.hi.cc.u32 t9, %14, %18, t9;\n\t"
        "madc.hi.cc.u32 t10, %14, %19, t10;\n\t"
        "madc.hi.cc.u32 t11, %14, %20, t11;\n\t"
        "madc.hi.cc.u32 t12, %14, %21, t12;\n\t"
        "madc.hi.cc.u32 t13, %14, %22, t13;\n\t"
        "madc.hi.cc.u32 t14, %14, %23, t14;\n\t"
        "addc.u32 t15, 0, 0;\n\t"
        "mad.lo.cc.u32 t7, %15, %16, t7;\n\t"
        "madc.lo.cc.u32 t8, %15, %17, t8;\n\t"
        "madc.lo.cc.u32 t9, %15, %18, t9;\n\t"
        "madc.lo.cc.u32 t10, %15, %19, t10;\n\t"
        "madc.lo.cc.u32 t11, %15, %20, t11;\n\t"
        "madc.lo.cc.u32 t12, %15, %21, t12;\n\t"
        "madc.lo.cc.u32 t13, %15, %22, t13;\n\t"
        "madc.lo.cc.u32 t14, %15, %23, t14;\n\t"
        "addc.u32 t15, t15, 0;\n\t"
        "mad.hi.cc.u32 t8, %15, %16, t8;\n\t"
        "madc.hi.cc.u32 t9, %15, %17, t9;\n\t"
        "madc.hi.cc.u32 t10, %15, %18, t10;\n\t"
        "madc.hi.cc.u32 t11, %15, %19, t11;\n\t"
        "madc.hi.cc.u32 t12, %15, %20, t12;\n\t"
        "madc.hi.cc.u32 t13, %15, %21, t13;\n\t"
        "madc.hi.cc.u32 t14, %15, %22, t14;\n\t"
        "madc.hi.cc.u32 t15, %15, %23, t15;\n\t"
        "addc.u32 t16, 0, 0;\n\t"
        "mad.lo.cc.u32 %0, t8, 38, t0;\n\t"
        "madc.lo.cc.u32 %1, t9, 38, t1;\n\t"
        "madc.lo.cc.u32 %2, t10, 38, t2;\n\t"
        "madc.lo.cc.u32 %3, t11, 38, t3;\n\t"
        "madc.lo.cc.u32 %4, t12, 38, t4;\n\t"
        "madc.lo.cc.u32 %5, t13, 38, t5;\n\t"
        "madc.lo.cc.u32 %6, t14, 38, t6;\n\t"
        "madc.lo.cc.u32 %7, t15, 38, t7;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mad.hi.cc.u32 %1, t8, 38, %1;\n\t"
        "madc.hi.cc.u32 %2, t9, 38, %2;\n\t"
        "madc.hi.cc.u32 %3, t10, 38, %3;\n\t"
        "madc.hi.cc.u32 %4, t11, 38, %4;\n\t"
        "madc.hi.cc.u32 %5, t12, 38, %5;\n\t"
        "madc.hi.cc.u32 %6, t13, 38, %6;\n\t"
        "madc.hi.cc.u32 %7, t14, 38, %7;\n\t"
        "madc.hi.u32 c, t15, 38, c;\n\t"
        "mul.lo.u32 c, c, 38;\n\t"
        "add.cc.u32 %0, %0, c;\n\t"
        "addc.cc.u32 %1, %1, 0;\n\t"
        "addc.cc.u32 %2, %2, 0;\n\t"
        "addc.cc.u32 %3, %3, 0;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\t"
        "addc.cc.u32 %5, %5, 0;\n\t"
        "addc.cc.u32 %6, %6, 0;\n\t"
        "addc.cc.u32 %7, %7, 0;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mad.lo.u32 %0, c, 38, %0;\n\t"
        "}"
        : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
        : "r"(a.v[0]), "r"(a.v[1]), "r"(a.v[2]), "r"(a.v[3]), "r"(a.v[4]), "r"(a.v[5]), "r"(a.v[6]), "r"(a.v[7]), "r"(b.v[0]), "r"(b.v[1]), "r"(b.v[2]), "r"(b.v[3]), "r"(b.v[4]), "r"(b.v[5]), "r"(b.v[6]), "r"(b.v[7]));
    return r;
}

/// a²。対角でない積 a_i·a_j（i < j）を足して 2 倍し、対角の積 a_i² を足す（積は 64 回から 36 回に減る）。
/// 上の 256 ビットを 38 倍して下に足すのは fe_mul と同じ
DEV fe fe_sq(const fe a) {
    fe r;
    asm(
        "{\n\t"
        ".reg .u32 c, t<16>;\n\t"
        "mov.u32 t0, 0;\n\t"
        "mov.u32 t1, 0;\n\t"
        "mov.u32 t2, 0;\n\t"
        "mov.u32 t3, 0;\n\t"
        "mov.u32 t4, 0;\n\t"
        "mov.u32 t5, 0;\n\t"
        "mov.u32 t6, 0;\n\t"
        "mov.u32 t7, 0;\n\t"
        "mov.u32 t8, 0;\n\t"
        "mov.u32 t9, 0;\n\t"
        "mov.u32 t10, 0;\n\t"
        "mov.u32 t11, 0;\n\t"
        "mov.u32 t12, 0;\n\t"
        "mov.u32 t13, 0;\n\t"
        "mov.u32 t14, 0;\n\t"
        "mov.u32 t15, 0;\n\t"
        "mad.lo.cc.u32 t1, %8, %9, t1;\n\t"
        "madc.lo.cc.u32 t2, %8, %10, t2;\n\t"
        "madc.lo.cc.u32 t3, %8, %11, t3;\n\t"
        "madc.lo.cc.u32 t4, %8, %12, t4;\n\t"
        "madc.lo.cc.u32 t5, %8, %13, t5;\n\t"
        "madc.lo.cc.u32 t6, %8, %14, t6;\n\t"
        "madc.lo.cc.u32 t7, %8, %15, t7;\n\t"
        "addc.u32 t8, t8, 0;\n\t"
        "mad.hi.cc.u32 t2, %8, %9, t2;\n\t"
        "madc.hi.cc.u32 t3, %8, %10, t3;\n\t"
        "madc.hi.cc.u32 t4, %8, %11, t4;\n\t"
        "madc.hi.cc.u32 t5, %8, %12, t5;\n\t"
        "madc.hi.cc.u32 t6, %8, %13, t6;\n\t"
        "madc.hi.cc.u32 t7, %8, %14, t7;\n\t"
        "madc.hi.cc.u32 t8, %8, %15, t8;\n\t"
        "addc.u32 t9, 0, 0;\n\t"
        "mad.lo.cc.u32 t3, %9, %10, t3;\n\t"
        "madc.lo.cc.u32 t4, %9, %11, t4;\n\t"
        "madc.lo.cc.u32 t5, %9, %12, t5;\n\t"
        "madc.lo.cc.u32 t6, %9, %13, t6;\n\t"
        "madc.lo.cc.u32 t7, %9, %14, t7;\n\t"
        "madc.lo.cc.u32 t8, %9, %15, t8;\n\t"
        "addc.u32 t9, t9, 0;\n\t"
        "mad.hi.cc.u32 t4, %9, %10, t4;\n\t"
        "madc.hi.cc.u32 t5, %9, %11, t5;\n\t"
        "madc.hi.cc.u32 t6, %9, %12, t6;\n\t"
        "madc.hi.cc.u32 t7, %9, %13, t7;\n\t"
        "madc.hi.cc.u32 t8, %9, %14, t8;\n\t"
        "madc.hi.cc.u32 t9, %9, %15, t9;\n\t"
        "addc.u32 t10, 0, 0;\n\t"
        "mad.lo.cc.u32 t5, %10, %11, t5;\n\t"
        "madc.lo.cc.u32 t6, %10, %12, t6;\n\t"
        "madc.lo.cc.u32 t7, %10, %13, t7;\n\t"
        "madc.lo.cc.u32 t8, %10, %14, t8;\n\t"
        "madc.lo.cc.u32 t9, %10, %15, t9;\n\t"
        "addc.u32 t10, t10, 0;\n\t"
        "mad.hi.cc.u32 t6, %10, %11, t6;\n\t"
        "madc.hi.cc.u32 t7, %10, %12, t7;\n\t"
        "madc.hi.cc.u32 t8, %10, %13, t8;\n\t"
        "madc.hi.cc.u32 t9, %10, %14, t9;\n\t"
        "madc.hi.cc.u32 t10, %10, %15, t10;\n\t"
        "addc.u32 t11, 0, 0;\n\t"
        "mad.lo.cc.u32 t7, %11, %12, t7;\n\t"
        "madc.lo.cc.u32 t8, %11, %13, t8;\n\t"
        "madc.lo.cc.u32 t9, %11, %14, t9;\n\t"
        "madc.lo.cc.u32 t10, %11, %15, t10;\n\t"
        "addc.u32 t11, t11, 0;\n\t"
        "mad.hi.cc.u32 t8, %11, %12, t8;\n\t"
        "madc.hi.cc.u32 t9, %11, %13, t9;\n\t"
        "madc.hi.cc.u32 t10, %11, %14, t10;\n\t"
        "madc.hi.cc.u32 t11, %11, %15, t11;\n\t"
        "addc.u32 t12, 0, 0;\n\t"
        "mad.lo.cc.u32 t9, %12, %13, t9;\n\t"
        "madc.lo.cc.u32 t10, %12, %14, t10;\n\t"
        "madc.lo.cc.u32 t11, %12, %15, t11;\n\t"
        "addc.u32 t12, t12, 0;\n\t"
        "mad.hi.cc.u32 t10, %12, %13, t10;\n\t"
        "madc.hi.cc.u32 t11, %12, %14, t11;\n\t"
        "madc.hi.cc.u32 t12, %12, %15, t12;\n\t"
        "addc.u32 t13, 0, 0;\n\t"
        "mad.lo.cc.u32 t11, %13, %14, t11;\n\t"
        "madc.lo.cc.u32 t12, %13, %15, t12;\n\t"
        "addc.u32 t13, t13, 0;\n\t"
        "mad.hi.cc.u32 t12, %13, %14, t12;\n\t"
        "madc.hi.cc.u32 t13, %13, %15, t13;\n\t"
        "addc.u32 t14, 0, 0;\n\t"
        "mad.lo.cc.u32 t13, %14, %15, t13;\n\t"
        "addc.u32 t14, t14, 0;\n\t"
        "mad.hi.cc.u32 t14, %14, %15, t14;\n\t"
        "addc.u32 t15, 0, 0;\n\t"
        "add.cc.u32 t1, t1, t1;\n\t"
        "addc.cc.u32 t2, t2, t2;\n\t"
        "addc.cc.u32 t3, t3, t3;\n\t"
        "addc.cc.u32 t4, t4, t4;\n\t"
        "addc.cc.u32 t5, t5, t5;\n\t"
        "addc.cc.u32 t6, t6, t6;\n\t"
        "addc.cc.u32 t7, t7, t7;\n\t"
        "addc.cc.u32 t8, t8, t8;\n\t"
        "addc.cc.u32 t9, t9, t9;\n\t"
        "addc.cc.u32 t10, t10, t10;\n\t"
        "addc.cc.u32 t11, t11, t11;\n\t"
        "addc.cc.u32 t12, t12, t12;\n\t"
        "addc.cc.u32 t13, t13, t13;\n\t"
        "addc.cc.u32 t14, t14, t14;\n\t"
        "addc.u32 t15, t15, t15;\n\t"
        "mad.lo.cc.u32 t0, %8, %8, t0;\n\t"
        "madc.hi.cc.u32 t1, %8, %8, t1;\n\t"
        "madc.lo.cc.u32 t2, %9, %9, t2;\n\t"
        "madc.hi.cc.u32 t3, %9, %9, t3;\n\t"
        "madc.lo.cc.u32 t4, %10, %10, t4;\n\t"
        "madc.hi.cc.u32 t5, %10, %10, t5;\n\t"
        "madc.lo.cc.u32 t6, %11, %11, t6;\n\t"
        "madc.hi.cc.u32 t7, %11, %11, t7;\n\t"
        "madc.lo.cc.u32 t8, %12, %12, t8;\n\t"
        "madc.hi.cc.u32 t9, %12, %12, t9;\n\t"
        "madc.lo.cc.u32 t10, %13, %13, t10;\n\t"
        "madc.hi.cc.u32 t11, %13, %13, t11;\n\t"
        "madc.lo.cc.u32 t12, %14, %14, t12;\n\t"
        "madc.hi.cc.u32 t13, %14, %14, t13;\n\t"
        "madc.lo.cc.u32 t14, %15, %15, t14;\n\t"
        "madc.hi.u32 t15, %15, %15, t15;\n\t"
        "mad.lo.cc.u32 %0, t8, 38, t0;\n\t"
        "madc.lo.cc.u32 %1, t9, 38, t1;\n\t"
        "madc.lo.cc.u32 %2, t10, 38, t2;\n\t"
        "madc.lo.cc.u32 %3, t11, 38, t3;\n\t"
        "madc.lo.cc.u32 %4, t12, 38, t4;\n\t"
        "madc.lo.cc.u32 %5, t13, 38, t5;\n\t"
        "madc.lo.cc.u32 %6, t14, 38, t6;\n\t"
        "madc.lo.cc.u32 %7, t15, 38, t7;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mad.hi.cc.u32 %1, t8, 38, %1;\n\t"
        "madc.hi.cc.u32 %2, t9, 38, %2;\n\t"
        "madc.hi.cc.u32 %3, t10, 38, %3;\n\t"
        "madc.hi.cc.u32 %4, t11, 38, %4;\n\t"
        "madc.hi.cc.u32 %5, t12, 38, %5;\n\t"
        "madc.hi.cc.u32 %6, t13, 38, %6;\n\t"
        "madc.hi.cc.u32 %7, t14, 38, %7;\n\t"
        "madc.hi.u32 c, t15, 38, c;\n\t"
        "mul.lo.u32 c, c, 38;\n\t"
        "add.cc.u32 %0, %0, c;\n\t"
        "addc.cc.u32 %1, %1, 0;\n\t"
        "addc.cc.u32 %2, %2, 0;\n\t"
        "addc.cc.u32 %3, %3, 0;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\t"
        "addc.cc.u32 %5, %5, 0;\n\t"
        "addc.cc.u32 %6, %6, 0;\n\t"
        "addc.cc.u32 %7, %7, 0;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mad.lo.u32 %0, c, 38, %0;\n\t"
        "}"
        : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
        : "r"(a.v[0]), "r"(a.v[1]), "r"(a.v[2]), "r"(a.v[3]), "r"(a.v[4]), "r"(a.v[5]), "r"(a.v[6]), "r"(a.v[7]));
    return r;
}

#else
// CPU（テスト）では同じ計算を 64 ビットの整数でする

/// r + c·2^256（c < 2^32 / 38）を 2^256 未満に（c·38 を足し、もう一度あふれたら 38 を足す）
DEV fe fe_fold(fe r, u64 c) {
    c *= 38;
    for (int i = 0; i < 8; i++) {
        c += r.v[i];
        r.v[i] = (u32)c;
        c >>= 32;
    }
    r.v[0] += (u32)c * 38;
    return r;
}

DEV fe fe_add(const fe a, const fe b) {
    fe r;
    u64 c = 0;
    for (int i = 0; i < 8; i++) {
        c += (u64)a.v[i] + b.v[i];
        r.v[i] = (u32)c;
        c >>= 32;
    }
    return fe_fold(r, c);
}

DEV fe fe_sub(const fe a, const fe b) {
    fe r;
    u64 borrow = 0;
    for (int i = 0; i < 8; i++) {
        u64 d = (u64)a.v[i] - b.v[i] - borrow;
        r.v[i] = (u32)d;
        borrow = d >> 63;
    }
    // 負になったら 2^256 足した値になっているので 38 を引く。もう一度負になったら、さらに 38 を引く
    for (int round = 0; round < 2; round++) {
        u64 m = borrow * 38;
        borrow = 0;
        for (int i = 0; i < 8; i++) {
            u64 d = (u64)r.v[i] - (i == 0 ? m : 0) - borrow;
            r.v[i] = (u32)d;
            borrow = d >> 63;
        }
    }
    return r;
}

DEV fe fe_mul(const fe a, const fe b) {
    u64 t[16] = {0};
    for (int i = 0; i < 8; i++) {
        u64 c = 0;
        for (int j = 0; j < 8; j++) {
            c += (u64)a.v[i] * b.v[j] + t[i + j];
            t[i + j] = (u32)c;
            c >>= 32;
        }
        t[i + 8] = c;
    }
    fe r;
    u64 c = 0;
    for (int j = 0; j < 8; j++) {
        c += t[j] + t[8 + j] * 38;
        r.v[j] = (u32)c;
        c >>= 32;
    }
    return fe_fold(r, c);
}

/// CPU では fe_mul(a, a)（GPU の 2 乗と同じ積から同じ手順で畳むので、結果もビット単位で同じ）
DEV fe fe_sq(const fe a) { return fe_mul(a, a); }
#endif

/// 8 リムでは桁を整えなくてよい（どの演算の結果も 2^256 未満）
DEV fe fe_carry(const fe a) { return a; }

/// 表の 32 ビット × 8 語は、そのまま 8 リム
DEV fe fe_unpack(const u32 w[8]) {
    fe r;
    UNROLL
    for (int i = 0; i < 8; i++) r.v[i] = w[i];
    return r;
}

/// 0 以上 p 未満に正規化したリトルエンディアン 32 バイト
DEV void fe_to_bytes(const fe a, u8 out[32]) { words_to_bytes(a.v, out); }

DEV fe fe_small(u32 v) {
    fe r;
    UNROLL
    for (int i = 0; i < 8; i++) r.v[i] = 0;
    r.v[0] = v;
    return r;
}

#elif MEIKEY_FIELD == 26

// 26 ビット × 10 リム（一様。値は Σ v[i]·2^(26i)、260 ビットまで持てる）。積は 5 リムどうしの積 3 回
// （Karatsuba 法）にし、2^260 ≡ 608（= 19·2^5）で下に畳む。
// リムの上限: 「整えた」値（fe_mul・fe_sq・fe_carry の結果、表の値）は各リム 2^26 + 2^16 未満。
// fe_add・fe_sub の結果（2^27.6 未満）もそのまま fe_mul・fe_sq に渡せる（列の和は 2^64 未満）。
// fe_sub(a, b) の b は整えた値（64p の各リムを超えない）

typedef struct { u32 v[10]; } fe;

#define MASK26 0x3ffffffu

/// 64p = 2·(2^260 - 608) のリム（引き算で負にならないように足す）
CONSTANT u32 FE_P64[10] = {
    (1u << 27) - 1216, (1u << 27) - 2, (1u << 27) - 2, (1u << 27) - 2, (1u << 27) - 2,
    (1u << 27) - 2,    (1u << 27) - 2, (1u << 27) - 2, (1u << 27) - 2, (1u << 27) - 2,
};

DEV fe fe_add(const fe a, const fe b) {
    fe r;
    UNROLL
    for (int i = 0; i < 10; i++) r.v[i] = a.v[i] + b.v[i];
    return r;
}

/// a + 64p - b
DEV fe fe_sub(const fe a, const fe b) {
    fe r;
    UNROLL
    for (int i = 0; i < 10; i++) r.v[i] = a.v[i] + FE_P64[i] - b.v[i];
    return r;
}

/// 5 リムどうしの積の k 番目の列（Σ a[i]·b[k-i]）
DEV u64 col5(const u32* a, const u32* b, int k) {
    u64 v = 0;
    UNROLL
    for (int i = 0; i < 5; i++) {
        if (k - i >= 0 && k - i < 5) v += (u64)a[i] * b[k - i];
    }
    return v;
}

/// 5 リムの 2 乗の k 番目の列（a2 は 2a。a は 2^31 未満）
DEV u64 sqcol5(const u32* a, const u32* a2, int k) {
    u64 v = 0;
    UNROLL
    for (int i = 0; i < 5; i++) {
        const int j = k - i;
        if (j > i && j < 5) v += (u64)a2[i] * a[j];
        if (j == i) v += (u64)a[i] * a[i];
    }
    return v;
}

/// 列 c[0..19)（値は Σ c[k]·2^(26k)）を整えた 10 リムに
DEV fe fe_fold(u64 c[19]) {
    // 上の列を 26 ビットずつに（はみ出しは c19）
    UNROLL
    for (int k = 10; k < 18; k++) {
        c[k + 1] += c[k] >> 26;
        c[k] &= MASK26;
    }
    u64 c19 = c[18] >> 26;
    c[18] &= MASK26;
    // 2^260 ≡ 608 で下に畳む
    UNROLL
    for (int k = 0; k < 9; k++) c[k] += (u64)(u32)c[k + 10] * 608u;
    c[9] += c19 * 608u;
    UNROLL
    for (int k = 0; k < 9; k++) {
        c[k + 1] += c[k] >> 26;
        c[k] &= MASK26;
    }
    u64 top = c[9] >> 26;
    c[9] &= MASK26;
    c[0] += top * 608u;
    c[1] += c[0] >> 26;
    c[0] &= MASK26;
    fe r;
    UNROLL
    for (int i = 0; i < 10; i++) r.v[i] = (u32)c[i];
    return r;
}

/// Karatsuba: a·b = P0 + (Pm - P0 - P2)·2^130 + P2·2^260（P0 = 下半分どうし、P2 = 上半分どうし、
/// Pm = (下 + 上)·(下 + 上)）。P0・P2 の列は計算したらすぐ足して引く（列を全部持つとレジスタが足りない）。
/// 途中は 2^64 で回るが、最後の各列は Pm - P0 - P2 ≥ 0 なので正しい
DEV fe fe_mul(const fe a, const fe b) {
    u32 sa[5], sb[5];
    UNROLL
    for (int i = 0; i < 5; i++) {
        sa[i] = a.v[i] + a.v[i + 5];
        sb[i] = b.v[i] + b.v[i + 5];
    }
    u64 c[19];
    UNROLL
    for (int k = 0; k < 19; k++) c[k] = 0;
    UNROLL
    for (int k = 0; k < 9; k++) c[k + 5] = col5(sa, sb, k);
    UNROLL
    for (int k = 0; k < 9; k++) {
        u64 lo = col5(a.v, b.v, k), hi = col5(a.v + 5, b.v + 5, k);
        c[k] += lo;
        c[k + 10] += hi;
        c[k + 5] -= lo + hi;
    }
    return fe_fold(c);
}

DEV fe fe_sq(const fe a) {
    u32 sa[5], a2[10], sa2[5];
    UNROLL
    for (int i = 0; i < 5; i++) {
        sa[i] = a.v[i] + a.v[i + 5];
        sa2[i] = 2 * sa[i];
    }
    UNROLL
    for (int i = 0; i < 10; i++) a2[i] = 2 * a.v[i];
    u64 c[19];
    UNROLL
    for (int k = 0; k < 19; k++) c[k] = 0;
    UNROLL
    for (int k = 0; k < 9; k++) c[k + 5] = sqcol5(sa, sa2, k);
    UNROLL
    for (int k = 0; k < 9; k++) {
        u64 lo = sqcol5(a.v, a2, k), hi = sqcol5(a.v + 5, a2 + 5, k);
        c[k] += lo;
        c[k + 10] += hi;
        c[k + 5] -= lo + hi;
    }
    return fe_fold(c);
}

/// 整える（fe_add・fe_sub の結果を、fe_sub の b や逆元に渡せる形に）
DEV fe fe_carry(const fe a) {
    u64 c[10];
    UNROLL
    for (int i = 0; i < 10; i++) c[i] = a.v[i];
    UNROLL
    for (int k = 0; k < 9; k++) {
        c[k + 1] += c[k] >> 26;
        c[k] &= MASK26;
    }
    u64 top = c[9] >> 26;
    c[9] &= MASK26;
    c[0] += top * 608u;
    c[1] += c[0] >> 26;
    c[0] &= MASK26;
    fe r;
    UNROLL
    for (int i = 0; i < 10; i++) r.v[i] = (u32)c[i];
    return r;
}

/// 32 ビット × 8 語（リトルエンディアン、2^255 未満）を 26 ビット × 10 リムに
DEV fe fe_unpack(const u32 w[8]) {
    fe r;
    UNROLL
    for (int i = 0; i < 10; i++) {
        const int o = 26 * i, q = o / 32, b = o % 32;
        u32 v = w[q] >> b;
        if (b > 6 && q + 1 < 8) v |= w[q + 1] << (32 - b);
        r.v[i] = v & MASK26;
    }
    return r;
}

/// 表のリムの形のエントリ（10 語）
DEV fe fe_load(const u32 w[10]) {
    fe r;
    UNROLL
    for (int i = 0; i < 10; i++) r.v[i] = w[i];
    return r;
}

/// 0 以上 p 未満に正規化したリトルエンディアン 32 バイト（a は整えた値）
DEV void fe_to_bytes(const fe a, u8 out[32]) {
    // 2^255 以上の分（リム 9 の 21 ビット目から）を 19 倍して下に足し、26 ビットずつにする。これで 2^256 未満
    u64 c[10];
    UNROLL
    for (int i = 0; i < 10; i++) c[i] = a.v[i];
    UNROLL
    for (int k = 0; k < 9; k++) {
        c[k + 1] += c[k] >> 26;
        c[k] &= MASK26;
    }
    u64 top = c[9] >> 21;
    c[9] &= (1u << 21) - 1;
    c[0] += top * 19;
    UNROLL
    for (int k = 0; k < 9; k++) {
        c[k + 1] += c[k] >> 26;
        c[k] &= MASK26;
    }
    // 32 ビットの語に並べる（リム i は 26i ビット目から）
    u32 w[8];
    UNROLL
    for (int q = 0; q < 8; q++) w[q] = 0;
    UNROLL
    for (int i = 0; i < 10; i++) {
        const int o = 26 * i, q = o / 32, b = o % 32;
        w[q] |= (u32)(c[i] << b);
        if (b > 6 && q + 1 < 8) w[q + 1] |= (u32)(c[i] >> (32 - b));
    }
    words_to_bytes(w, out);
}

DEV fe fe_small(u32 v) {
    fe r;
    UNROLL
    for (int i = 0; i < 10; i++) r.v[i] = 0;
    r.v[0] = v;
    return r;
}

#else

// 51 ビット × 5 リム（値は Σ v[i]·2^(51i)）。リムどうしの積（104 ビット）を、倍精度浮動小数点数の FMA 2 回で
// 上と下に分ける: t = fma_rz(a, b, 2^104) は 2^104 + floor(ab / 2^52)·2^52、l = fma(a, b, 2^104 + 2^52 - t) は
// 2^52 + (ab mod 2^52)。どちらもビット列を整数として読むと「定数 + 上の桁・下の桁」になるので、そのまま
// 64 ビットの整数で列ごとに足し、最後に定数を引く（Emmart ら, ARITH 2018 の方法）。
// リムの上限: 整えた値（どの演算の結果も）は各リム 2^51 + 2^14 未満。積に渡すリムは 2^52 未満なら正確
// （fe_add・fe_sub は結果を整える）

typedef struct { u64 v[5]; } fe;

#define MASK51 0x7ffffffffffffull

/// 2^104 と 2^104 + 2^52
#define F51_C104 20282409603651670423947251286016.0
#define F51_CS 20282409603651674927546878656512.0
/// 2^52 と、[2^52, 2^53) の数のビット列の上位（指数）と、[2^104, 2^105) の数のビット列の上位
#define F51_TWO52 4503599627370496.0
#define F51_BIAS_LO 0x4330000000000000ull
#define F51_BIAS_HI 0x4670000000000000ull

#ifdef MEIKEY_CUDA
DEV double f51_from_bits(u64 b) { return __longlong_as_double((long long)b); }
DEV u64 f51_bits(double d) { return (u64)__double_as_longlong(d); }
/// fma(a, b, 2^104) を 0 方向に丸めたもの（a・b は 2^52 未満の整数）
DEV double f51_top(double a, double b) { return __fma_rz(a, b, F51_C104); }
DEV double f51_fma(double a, double b, double c) { return __fma_rn(a, b, c); }
DEV double f51_sub(double a, double b) { return __dsub_rn(a, b); }
#else
DEV double f51_from_bits(u64 b) {
    double d;
    memcpy(&d, &b, 8);
    return d;
}
DEV u64 f51_bits(double d) {
    u64 b;
    memcpy(&b, &d, 8);
    return b;
}
/// CPU では丸めの向きを変えずに、同じ値（2^104 + floor(ab / 2^52)·2^52）を整数で計算する
/// （a = ah·2^26 + al、b = bh·2^26 + bl に分けると floor(ab / 2^52) = ah·bh + floor((ah·bl + al·bh + floor(al·bl / 2^26)) / 2^26)）
DEV double f51_top(double a, double b) {
    u64 x = (u64)a, y = (u64)b;
    u64 ah = x >> 26, al = x & ((1ull << 26) - 1), bh = y >> 26, bl = y & ((1ull << 26) - 1);
    u64 hi = ah * bh + ((ah * bl + al * bh + ((al * bl) >> 26)) >> 26);
    return F51_C104 + (double)hi * F51_TWO52;
}
DEV double f51_fma(double a, double b, double c) { return fma(a, b, c); }
DEV double f51_sub(double a, double b) { return a - b; }
#endif

/// 2^52 未満の整数を倍精度に（ビット列に 2^52 の指数を付けて 2^52 を引く）
DEV double f51_to_double(u64 x) { return f51_sub(f51_from_bits(F51_BIAS_LO | x), F51_TWO52); }

/// 整える（各リムを 51 ビットに。2^255 以上は 19 倍して下へ）
DEV fe fe_carry(fe a) {
    UNROLL
    for (int i = 0; i < 4; i++) {
        a.v[i + 1] += a.v[i] >> 51;
        a.v[i] &= MASK51;
    }
    u64 t = a.v[4] >> 51;
    a.v[4] &= MASK51;
    a.v[0] += 19 * t;
    a.v[1] += a.v[0] >> 51;
    a.v[0] &= MASK51;
    return a;
}

DEV fe fe_add(const fe a, const fe b) {
    fe r;
    UNROLL
    for (int i = 0; i < 5; i++) r.v[i] = a.v[i] + b.v[i];
    return fe_carry(r);
}

/// a + 4p - b（b は整えた値）
DEV fe fe_sub(const fe a, const fe b) {
    fe r;
    UNROLL
    for (int i = 0; i < 5; i++) r.v[i] = a.v[i] + (i == 0 ? 4 * (MASK51 - 18) : 4 * MASK51) - b.v[i];
    return fe_carry(r);
}

DEV fe fe_mul(const fe a, const fe b) {
    double da[5], db[5];
    UNROLL
    for (int i = 0; i < 5; i++) {
        da[i] = f51_to_double(a.v[i]);
        db[i] = f51_to_double(b.v[i]);
    }
    // lo[k]: 列 k の下の桁のビット列の和、hi[k]: 列 k - 1 の上の桁（2^52 単位 = 列 k の 2 倍）のビット列の和
    u64 lo[9], hi[10];
    UNROLL
    for (int k = 0; k < 9; k++) lo[k] = 0;
    UNROLL
    for (int k = 0; k < 10; k++) hi[k] = 0;
    UNROLL
    for (int i = 0; i < 5; i++) {
        UNROLL
        for (int j = 0; j < 5; j++) {
            double t = f51_top(da[i], db[j]);
            double l = f51_fma(da[i], db[j], f51_sub(F51_CS, t));
            lo[i + j] += f51_bits(l);
            hi[i + j + 1] += f51_bits(t);
        }
    }
    // 定数を引いて列の値に（列 k の積の数は 5 - |k - 4|）。2^255 ≡ 19 で下に畳む
    u64 c[10];
    UNROLL
    for (int k = 0; k < 10; k++) {
        const u64 nl = k < 9 ? (u64)(5 - (k > 4 ? k - 4 : 4 - k)) : 0;
        const u64 nh = k >= 1 ? (u64)(5 - (k - 1 > 4 ? k - 5 : 5 - k)) : 0;
        u64 l = k < 9 ? lo[k] - nl * F51_BIAS_LO : 0;
        c[k] = l + 2 * (hi[k] - nh * F51_BIAS_HI);
    }
    fe r;
    UNROLL
    for (int k = 0; k < 5; k++) r.v[k] = c[k] + 19 * c[k + 5];
    return fe_carry(r);
}

/// 2 乗（同じ積は同じ計算なので、コンパイラが 15 個にまとめる）
DEV fe fe_sq(const fe a) { return fe_mul(a, a); }

/// 32 ビット × 8 語（リトルエンディアン、2^255 未満）を 51 ビット × 5 リムに
DEV fe fe_unpack(const u32 w[8]) {
    u64 q[4];
    UNROLL
    for (int i = 0; i < 4; i++) q[i] = (u64)w[2 * i] | (u64)w[2 * i + 1] << 32;
    fe r;
    r.v[0] = q[0] & MASK51;
    r.v[1] = (q[0] >> 51 | q[1] << 13) & MASK51;
    r.v[2] = (q[1] >> 38 | q[2] << 26) & MASK51;
    r.v[3] = (q[2] >> 25 | q[3] << 39) & MASK51;
    r.v[4] = (q[3] >> 12) & MASK51;
    return r;
}

/// 0 以上 p 未満に正規化したリトルエンディアン 32 バイト（a は整えた値なので 2^256 未満）
DEV void fe_to_bytes(fe a, u8 out[32]) {
    // 下のリムを 51 ビットちょうどにしてから並べる（整えた値でも、リム 1〜4 は 2^51 になることがある）
    UNROLL
    for (int i = 0; i < 4; i++) {
        a.v[i + 1] += a.v[i] >> 51;
        a.v[i] &= MASK51;
    }
    u64 q[4] = {
        a.v[0] | a.v[1] << 51,
        a.v[1] >> 13 | a.v[2] << 38,
        a.v[2] >> 26 | a.v[3] << 25,
        a.v[3] >> 39 | a.v[4] << 12,
    };
    u32 w[8];
    UNROLL
    for (int i = 0; i < 4; i++) {
        w[2 * i] = (u32)q[i];
        w[2 * i + 1] = (u32)(q[i] >> 32);
    }
    words_to_bytes(w, out);
}

DEV fe fe_small(u32 v) {
    fe r;
    UNROLL
    for (int i = 0; i < 5; i++) r.v[i] = 0;
    r.v[0] = v;
    return r;
}

#endif

/// a^(2^k)。k は最大 100 なので、展開すると 2 乗が 100 個並んでコードが大きくなるだけ（8 リムでは、
/// インラインアセンブリの 2 乗を展開してしまい PTX が数倍になる）
DEV fe fe_pow2k(fe a, int k) {
    NO_UNROLL
    for (int i = 0; i < k; i++) a = fe_sq(a);
    return a;
}

/// 1/a = a^(p-2)（field.rs の invert と同じ加算連鎖。a は整えた値）
DEV fe fe_invert(const fe a) {
    fe a2 = fe_sq(a);
    fe a9 = fe_mul(fe_sq(fe_sq(a2)), a);
    fe a11 = fe_mul(a9, a2);
    fe a_5_0 = fe_mul(fe_sq(a11), a9);
    fe a_10_0 = fe_mul(fe_pow2k(a_5_0, 5), a_5_0);
    fe a_20_0 = fe_mul(fe_pow2k(a_10_0, 10), a_10_0);
    fe a_40_0 = fe_mul(fe_pow2k(a_20_0, 20), a_20_0);
    fe a_50_0 = fe_mul(fe_pow2k(a_40_0, 10), a_10_0);
    fe a_100_0 = fe_mul(fe_pow2k(a_50_0, 50), a_50_0);
    fe a_200_0 = fe_mul(fe_pow2k(a_100_0, 100), a_100_0);
    fe a_250_0 = fe_mul(fe_pow2k(a_200_0, 50), a_50_0);
    return fe_mul(fe_pow2k(a_250_0, 5), a11);
}

/// v[0..n) をそれぞれの逆元に（Montgomery の一括逆元。v は整えた値、scratch は n 個）
DEV void batch_invert(fe* v, fe* scratch, int n) {
    scratch[0] = v[0];
    for (int i = 1; i < n; i++) scratch[i] = fe_mul(scratch[i - 1], v[i]);
    fe inv = fe_invert(scratch[n - 1]);
    for (int i = n - 1; i > 0; i--) {
        fe t = fe_mul(inv, scratch[i - 1]);
        inv = fe_mul(inv, v[i]);
        v[i] = t;
    }
    v[0] = inv;
}

/// q[start + stride·k]（k < n、n は WARPS 以下）をそれぞれの逆元に（ブロックの中で逆元をまとめる。q は整えた値）。
/// 途中の積は scratch の同じ位置に置く（レジスタに持つとスピルする）
DEV void invert_strided(fe* q, fe* scratch, int start, int stride, int n) {
    scratch[start] = q[start];
    for (int k = 1; k < n; k++) {
        scratch[start + stride * k] = fe_mul(scratch[start + stride * (k - 1)], q[start + stride * k]);
    }
    fe inv = fe_invert(scratch[start + stride * (n - 1)]);
    for (int k = n - 1; k > 0; k--) {
        fe t = fe_mul(inv, scratch[start + stride * (k - 1)]);
        inv = fe_mul(inv, q[start + stride * k]);
        q[start + stride * k] = t;
    }
    q[start] = inv;
}

// ---------------------------------------------------------------------------
// 点の演算（comb.rs・avx2.rs と同じ式）
// ---------------------------------------------------------------------------

typedef struct { fe x, y, z, t; } ext;
typedef struct { fe ypx, ymx, xy2d; } niels;

/// 表の row 行目の桁 d のエントリの先頭（列数 cols）
DEV const u32* entry_at(const u32* table, u32 row, i32 d, u32 cols) {
    return table + ((u64)row * cols + (u32)(d < 0 ? -d : d)) * ENTRY_WORDS;
}

/// 表のエントリを読み込み始める（最初と最後の語。96 バイトのエントリは 128 バイトの境界をまたぐことがある）
DEV void prefetch_entry(const u32* e) {
    PREFETCH(e);
    PREFETCH(e + ENTRY_WORDS - 1);
}

/// 表の row 行目の桁 d の点（負の桁は y+x と y-x を入れ替え、2dxy を反転）。
/// エントリは 16 バイト境界なので、16 バイトずつ読む。1 行目の 3 つ目は 2xy（to_extended 用）
DEV niels table_entry(const u32* table, u32 row, i32 d, u32 cols) {
    const u32* e = entry_at(table, row, d, cols);
    u32 w[ENTRY_WORDS];
    UNROLL
    for (int q = 0; q < ENTRY_WORDS / 4; q++) LOAD4(w + 4 * q, e + 4 * q);
    niels n;
#if MEIKEY_UNPACKED
    n.ypx = fe_load(w);
    n.ymx = fe_load(w + 10);
    n.xy2d = fe_load(w + 20);
#else
    n.ypx = fe_unpack(w);
    n.ymx = fe_unpack(w + 8);
    n.xy2d = fe_unpack(w + 16);
#endif
    if (d < 0) {
        fe t = n.ypx;
        n.ypx = n.ymx;
        n.ymx = t;
        n.xy2d = fe_sub(fe_small(0), n.xy2d);
    }
    return n;
}

/// 表の 1 行目の点（3 つ目が 2xy）を拡張座標に（X = 2x, Y = 2y, Z = 2, T = 2xy。乗算なし）
DEV ext to_extended(const niels n) {
    ext p;
    p.x = fe_carry(fe_sub(n.ypx, n.ymx));
    p.y = fe_carry(fe_add(n.ypx, n.ymx));
    p.z = fe_small(2);
    p.t = n.xy2d;
    return p;
}

/// 拡張座標 + アフィン点（乗算 7 回。with_t が偽なら T を計算しない）
DEV ext add_niels(const ext p, const niels q, bool with_t) {
    fe pp = fe_mul(fe_add(p.y, p.x), q.ypx);
    fe mm = fe_mul(fe_sub(p.y, p.x), q.ymx);
    fe txy = fe_mul(p.t, q.xy2d);
    fe z2 = fe_add(p.z, p.z);
    fe x = fe_sub(pp, mm), y = fe_add(pp, mm);
    fe z = fe_add(z2, txy), t = fe_sub(z2, txy);
    ext r;
    r.x = fe_mul(t, x);
    r.y = fe_mul(y, z);
    r.z = fe_mul(t, z);
    r.t = with_t ? fe_mul(x, y) : p.t;
    return r;
}

/// 最後の足し算: add_niels の結果は X = (pp - mm)·t、Y = (pp + mm)·z、Z = t·z なので、
/// y = Y / Z = (pp + mm) / t、x = X / Z = (pp - mm) / z（t = 2Z - T·2dxy、z = 2Z + T·2dxy）。
/// 掛け算は 3 回だけ。y の分子・分母を yn・yd に、want_x なら x の分子・分母を xn・xd に（分母は整えた値）
DEV void add_niels_frac(const ext p, const niels q, fe* yn, fe* yd, fe* xn, fe* xd, bool want_x) {
    fe pp = fe_mul(fe_add(p.y, p.x), q.ypx);
    fe mm = fe_mul(fe_sub(p.y, p.x), q.ymx);
    fe txy = fe_mul(p.t, q.xy2d);
    fe z2 = fe_add(p.z, p.z);
    *yn = fe_add(pp, mm);
    *yd = fe_carry(fe_sub(z2, txy));
    if (want_x) {
        *xn = fe_sub(pp, mm);
        *xd = fe_carry(fe_add(z2, txy));
    }
}

// ---------------------------------------------------------------------------
// 符号付き 2^W 進の桁（comb.rs の signed_digits と同じ。W は起動の引数なので、足しながら1つずつ取り出す）
// ---------------------------------------------------------------------------

typedef struct {
    /// m = s の 3〜253 ビット目
    u64 m[4];
    u32 w, rows, cols, i;
    i32 half, carry;
} Digits;

/// クランプ済みのスカラー s = 8·(2^251 + m) の桁を、幅 w で取り出す準備
DEV Digits digits_of(const u8 s[32], u32 w) {
    Digits g;
    u64 words[5];
    UNROLL
    for (int i = 0; i < 4; i++) {
        u64 v = 0;
        UNROLL
        for (int b = 7; b >= 0; b--) v = (v << 8) | s[8 * i + b];
        words[i] = v;
    }
    words[4] = 0;
    UNROLL
    for (int i = 0; i < 4; i++) g.m[i] = words[i] >> 3 | words[i + 1] << 61;
    g.m[3] &= (1ull << (M_BITS - 192)) - 1;
    g.w = w;
    g.rows = (M_BITS + w - 1) / w;
    g.cols = (1u << (w - 1)) + 1;
    g.half = 1 << (w - 1);
    g.carry = 0;
    g.i = 0;
    return g;
}

DEV u64 digit_word(const Digits& g, u32 q) { return q == 0 ? g.m[0] : q == 1 ? g.m[1] : q == 2 ? g.m[2] : q == 3 ? g.m[3] : 0; }

/// 次の桁（-2^(W-1)〜2^(W-1) - 1。最上位の桁だけ 0〜2^(W-1)）
DEV i32 next_digit(Digits& g) {
    u32 off = g.i * g.w, q = off >> 6, r = off & 63;
    u64 bits = digit_word(g, q) >> r;
    if (r != 0) bits |= digit_word(g, q + 1) << (64 - r);
    i32 v = (i32)(bits & ((1ull << g.w) - 1)) + g.carry;
    i32 d = v;
    if (g.i + 1 < g.rows) {
        g.carry = (v + g.half) >> g.w;
        d = v - (g.carry << g.w);
    }
    g.i++;
    return d;
}

/// 行数の上限（幅が最小のとき）
#define MAX_ROWS ((M_BITS + MIN_W - 1) / MIN_W)

/// 秘密スカラー s の s·B を、表の幅 w で計算する（最後の足し算は add_niels_frac の分数のまま）。
/// 桁は先に全部取り出して（ローカルメモリに）置き、次の行の表の値を読み込み始めてから今の行を足す
DEV void mul_base_frac(const u32* table, u32 w, const u8 s[32], fe* yn, fe* yd, fe* xn, fe* xd, bool want_x) {
    Digits g = digits_of(s, w);
    const u32 rows = g.rows, cols = g.cols;
    i32 d[MAX_ROWS];
    NO_UNROLL
    for (u32 i = 0; i < rows; i++) d[i] = next_digit(g);
    prefetch_entry(entry_at(table, 1, d[1], cols));
    ext acc = to_extended(table_entry(table, 0, d[0], cols));
    NO_UNROLL
    for (u32 row = 1; row + 1 < rows; row++) {
        prefetch_entry(entry_at(table, row + 1, d[row + 1], cols));
        acc = add_niels(acc, table_entry(table, row, d[row], cols), true);
    }
    add_niels_frac(acc, table_entry(table, rows - 1, d[rows - 1], cols), yn, yd, xn, xd, want_x);
}

// ---------------------------------------------------------------------------
// base64 と文字列の判定
// ---------------------------------------------------------------------------

/// 6 ビットの値 v の base64 の文字（A-Z a-z 0-9 + /）。分岐を使わず、範囲ごとの差を足す。`fold` なら英大文字を小文字に
DEV u8 b64_char(u32 v, bool fold) {
    u32 c = v + 'A';
    c += 6u * (v >= 26);   // 26 → 'a'
    c -= 75u * (v >= 52);  // 52 → '0'
    c -= 15u * (v >= 62);  // 62 → '+'
    c += 3u * (v >= 63);   // 63 → '/'
    c += 32u * (fold && v < 26);
    return (u8)c;
}

/// 公開鍵 blob の 18 バイト目からの 33 バイト（32 と公開鍵）の base64（44 文字）。`fold` なら英字を小文字に
DEV void tail_b64(const u8 pk[32], bool fold, u8 out[TAIL_B64]) {
    UNROLL
    for (int g = 0; g < 11; g++) {
        u32 b0 = g == 0 ? 32u : pk[3 * g - 1];
        u32 b1 = pk[3 * g];
        u32 b2 = pk[3 * g + 1];
        u32 n = b0 << 16 | b1 << 8 | b2;
        UNROLL
        for (int k = 0; k < 4; k++) out[4 * g + k] = b64_char((n >> (18 - 6 * k)) & 63, fold);
    }
}

/// x の符号で変わる、後半 44 文字の中の位置（pk[31] の最上位ビットは、この文字の値のビット 1）
#define SIGN_CHAR 42

/// 候補の位置（`candidates` のビット）のどれかから needle 全体が一致するか。候補が見つかったときだけ呼ぶので、
/// base64 を作り直して配列で比べる。`sign_unknown` なら pk の x の符号のビットは 0 にしてあり、SIGN_CHAR の
/// 文字は符号が 1 のときの文字とも比べる（どちらかで一致すれば true）
NOINLINE bool b64_contains_at(const u8 pk[32], bool fold, const u8* needle, u32 len, u64 candidates,
                              bool sign_unknown) {
    u8 b64[TAIL_B64];
    tail_b64(pk, fold, b64);
    u32 n = (u32)pk[29] << 16 | (u32)pk[30] << 8 | pk[31];
    u8 alt = b64_char(((n >> 6) & 63) | 2, fold);
    while (candidates != 0) {
        u32 s = 0;
        while (((candidates >> s) & 1) == 0) s++;
        candidates &= candidates - 1;
        u32 k = 0;
        while (k < len && (b64[s + k] == needle[k] || (sign_unknown && s + k == SIGN_CHAR && alt == needle[k]))) k++;
        if (k == len) return true;
    }
    return false;
}

/// base64 の文字 c の 6 ビットの値（文字表に無ければ、どの値とも一致しない 64）
DEV u32 b64_value(u8 c) {
    return c >= 'A' && c <= 'Z'   ? c - 'A'
           : c >= 'a' && c <= 'z' ? c - 'a' + 26
           : c >= '0' && c <= '9' ? c - '0' + 52
           : c == '+'             ? 62
           : c == '/'             ? 63
                                  : 64;
}

/// needle の1文字に一致する 6 ビットの値の集合（ビット v が立っていれば値 v が一致）。大小文字を区別しない（fold）
/// なら needle は小文字なので、英字は小文字と大文字の2つ
DEV u64 value_mask(u8 c, bool fold) {
    u32 v = b64_value(c);
    if (v >= 64) return 0;
    u64 m = 1ull << v;
    if (fold && c >= 'a' && c <= 'z') m |= 1ull << (v - 26);
    return m;
}

/// 探し方（model::pattern::Mode）。needle は、MODE_REGEX なら DFA の表、ほかは文字列
#define MODE_CONTAINS 0u
#define MODE_SUFFIX 1u
#define MODE_REGEX 2u

/// 探す文字列と、その先頭 2 文字（1 文字なら同じ文字）に一致する値の集合（起動ごとに1回だけ作る）
typedef struct {
    const u8* needle;
    u32 len, mode;
    bool fold;
    u64 m0, m1;
} Matcher;

DEV Matcher make_matcher(const u8* needle, u32 len, bool fold, u32 mode) {
    Matcher mt;
    mt.needle = needle;
    mt.len = len;
    mt.mode = mode;
    mt.fold = fold;
    mt.m0 = mt.m1 = 0;
    if (mode != MODE_REGEX && len > 0) {
        mt.m0 = value_mask(needle[0], fold);
        mt.m1 = value_mask(len >= 2 ? needle[1] : needle[0], fold);
    }
    return mt;
}

/// pk の base64 の判定の範囲（後半の 1 文字目から）に needle があるか（`suffix` なら最後にあるか）。
/// 文字にはせず 6 ビットの値のまま、全部の位置で先頭 2 文字（1 文字なら 1 文字）に一致するかを値の集合のビットで
/// 調べて候補を絞り、候補があるときだけ残りを比べる。`sign_unknown` なら pk の x の符号のビットは
/// 0 にしてあり、符号が 0 でも 1 でも含まれる可能性があれば true（含まれる鍵はかならず true）
DEV bool pk_contains(const u8 pk[32], const Matcher& mt, bool suffix, bool sign_unknown) {
    u32 len = mt.len;
    if (len == 0 || len > TAIL_B64 - SEARCH_FROM) return false;
    // h0・h1 のビット s: s 文字目が needle の 1・2 文字目と一致する
    u64 h0 = 0, h1 = 0;
    UNROLL
    for (int g = 0; g < 11; g++) {
        u32 b0 = g == 0 ? 32u : pk[3 * g - 1];
        u32 n = b0 << 16 | (u32)pk[3 * g] << 8 | pk[3 * g + 1];
        UNROLL
        for (int k = 0; k < 4; k++) {
            const int pos = 4 * g + k;
            u32 v = (n >> (18 - 6 * k)) & 63;
            u64 a = (mt.m0 >> v) & 1, b = (mt.m1 >> v) & 1;
            // 符号が分からなければ、SIGN_CHAR は符号が 1 のときの値（ビット 1 を立てたもの）とも比べる
            if (sign_unknown && pos == SIGN_CHAR) {
                a |= (mt.m0 >> (v | 2)) & 1;
                b |= (mt.m1 >> (v | 2)) & 1;
            }
            h0 |= a << pos;
            h1 |= b << pos;
        }
    }
    u64 candidates = len >= 2 ? h0 & (h1 >> 1) : h0;
    // 判定の範囲（SEARCH_FROM から）で、needle が収まる位置（s + len ≤ 44）だけ
    candidates &= ((2ull << (TAIL_B64 - len)) - 1) & ~((1ull << SEARCH_FROM) - 1);
    // 末尾なら、最後に収まる位置だけ
    if (suffix) candidates &= 1ull << (TAIL_B64 - len);
    if (candidates == 0) return false;
    if (len <= 2) return true;
    return b64_contains_at(pk, mt.fold, mt.needle, len, candidates, sign_unknown);
}

/// 正規表現の DFA の表（model::pattern::dfa の gpu_table）で、状態 s から 6 ビットの値 v を読んだ後の状態。
/// 表は状態ごとに 65 個の u16（リトルエンディアン）で、64 番目は文字列の終わりで一致するか
DEV u32 dfa_at(const u8* table, u32 s, u32 i) {
    u32 at = (s * 65u + i) * 2u;
    return (u32)table[at] | (u32)table[at + 1] << 8;
}

/// 一致した状態（どの文字でもここに留まる）
#define DFA_ACCEPT 0u
/// 判定の範囲の最初の文字を読む前の状態
#define DFA_START 1u

/// DFA の状態 s が一致しているか（一致した状態か、ここで文字列が終わると一致するか）
DEV bool dfa_done(const u8* table, u32 s) { return s == DFA_ACCEPT || dfa_at(table, s, 64) != 0; }

/// pk の base64 の判定の範囲（後半の 1 文字目から最後まで）が、正規表現の DFA `table` に一致するか。
/// `sign_unknown` なら pk の x の符号のビットは 0 にしてあり、SIGN_CHAR の文字は符号が 1 のときの値でも
/// 読んで（そこから先は 2 つの状態を進める）、どちらかで一致すれば true
DEV bool pk_dfa(const u8 pk[32], const u8* table, bool sign_unknown) {
    u32 s = DFA_START, alt = DFA_START;
    UNROLL
    for (int g = 0; g < 11; g++) {
        u32 b0 = g == 0 ? 32u : pk[3 * g - 1];
        u32 n = b0 << 16 | (u32)pk[3 * g] << 8 | pk[3 * g + 1];
        UNROLL
        for (int k = 0; k < 4; k++) {
            int pos = 4 * g + k;
            if (pos < SEARCH_FROM) continue;
            u32 v = (n >> (18 - 6 * k)) & 63;
            if (pos == SIGN_CHAR) {
                alt = dfa_at(table, s, sign_unknown ? v | 2 : v);
            } else if (pos > SIGN_CHAR) {
                alt = dfa_at(table, alt, v);
            }
            s = dfa_at(table, s, v);
        }
    }
    return dfa_done(table, s) || dfa_done(table, alt);
}

/// pk の base64 が、探し方 mt.mode で needle に一致するか（`sign_unknown` は pk_contains・pk_dfa と同じ）
DEV bool pk_match(const u8 pk[32], const Matcher& mt, bool sign_unknown) {
    if (mt.mode == MODE_REGEX) return pk_dfa(pk, mt.needle, sign_unknown);
    return pk_contains(pk, mt, mt.mode == MODE_SUFFIX, sign_unknown);
}

// ---------------------------------------------------------------------------
// 1スレッドの仕事
// ---------------------------------------------------------------------------

#ifdef MEIKEY_CUDA
#define ATOMIC_CAS64(p, cmp, val) atomicCAS((p), (cmp), (val))
#define VOLATILE_LOAD64(p) (*(volatile unsigned long long*)(p))
#else
DEV u64 host_cas64(u64* p, u64 cmp, u64 val) { u64 old = *p; if (old == cmp) *p = val; return old; }
#define ATOMIC_CAS64(p, cmp, val) host_cas64((p), (cmp), (val))
#define VOLATILE_LOAD64(p) (*(volatile u64*)(p))
#endif

/// 通し番号 `index` の秘密スカラー（`counter` ならシードは counter_seed（低セキュリティモード。mid は
/// sha512_mid(key)）、そうでなければ ChaCha20。block は ChaCha20 のブロックの使い回し）
DEV void scalar_at(const u32 key[8], u64 index, bool counter, const ShaMid& mid, u32 block[16], bool fresh,
                   u8 s[32]) {
    if (counter) {
        secret_scalar_mid(mid, key, index, s);
    } else {
        // 偶数の通し番号でブロックを作り、次の奇数の通し番号ではその後半を使う
        if (fresh || (index & 1) == 0) chacha20_block(key, index >> 1, block);
        u8 seed[32];
        seed_from_block(block, (index & 1) != 0, seed);
        secret_scalar(seed, s);
    }
}

/// 1スレッドの MEIKEY_BATCH 本の途中の値: y の分子・分母と、分母の積（pre[j] = 0〜j 本目の分母の積）
typedef struct {
    fe ys[MEIKEY_BATCH], zs[MEIKEY_BATCH], pre[MEIKEY_BATCH];
} Batch;

/// 通し番号 `first` から MEIKEY_BATCH 本の公開鍵の点を、y の分子・分母まで計算する（表の幅 w）
DEV void batch_derive(Batch& b, const u32* table, u32 w, const u32 key[8], u64 first, bool counter,
                      const ShaMid& mid) {
    u32 block[16];
    NO_UNROLL
    for (int j = 0; j < MEIKEY_BATCH; j++) {
        u8 s[32];
        scalar_at(key, first + (u64)j, counter, mid, block, j == 0, s);
        fe yn, yd, xn, xd;
        mul_base_frac(table, w, s, &yn, &yd, &xn, &xd, false);
        b.ys[j] = yn;
        b.zs[j] = yd;
        b.pre[j] = j == 0 ? yd : fe_mul(b.pre[j - 1], yd);
    }
}

/// 全部の分母の積の逆元 `inv` から、後ろの鍵から順に 1/分母 を取り出して y を圧縮し（x の符号のビットは 0）、
/// 1本できるたびに f(j, pk) を呼ぶ
template <class F>
DEV void batch_each(const Batch& b, fe inv, F f) {
    NO_UNROLL
    for (int j = MEIKEY_BATCH - 1; j >= 0; j--) {
        fe zinv = j == 0 ? inv : fe_mul(inv, b.pre[j - 1]);
        if (j > 0) inv = fe_mul(inv, b.zs[j]);
        u8 pk[32];
        fe_to_bytes(fe_mul(b.ys[j], zinv), pk);
        f(j, pk);
    }
}

/// 全部の分母の積の逆元 `inv` から、後ろの鍵から順に 1/分母 を取り出して y を圧縮し、判定する。
/// 文字列が含まれそうな鍵（x の符号はどちらでもよい）の番号 j をビットで返す
DEV u64 batch_finish(const Batch& b, fe inv, const Matcher& mt, u32* sink) {
    u64 hits = 0;
    batch_each(b, inv, [&](int j, const u8 pk[32]) {
        *sink ^= pk[31];
        hits |= (u64)pk_match(pk, mt, true) << j;
    });
    return hits;
}

/// 当たりがあれば、result[0] を 0 → 1 にした最初のスレッドが result[1] にいちばん前の当たりの通し番号を書く
DEV void report(u64 hits, u64 first, u64* result) {
    if (hits != 0 && ATOMIC_CAS64(result, 0ull, 1ull) == 0ull) {
        u32 j = 0;
        while (((hits >> j) & 1) == 0) j++;
        result[1] = first + j;
    }
}

/// 通し番号 `first` から MEIKEY_BATCH 本の公開鍵（圧縮形式。x の符号も計算する）を pks に書く
/// （確かめる・テスト用。y と x の分母をまとめて逆元にする）
DEV void derive_batch(const u32* table, u32 w, const u32 key[8], u64 first, bool counter, u8* pks) {
    ShaMid mid = sha512_mid(key);
    fe num[2 * MEIKEY_BATCH], den[2 * MEIKEY_BATCH], scratch[2 * MEIKEY_BATCH];
    u32 block[16];
    for (int j = 0; j < MEIKEY_BATCH; j++) {
        u8 s[32];
        scalar_at(key, first + (u64)j, counter, mid, block, j == 0, s);
        mul_base_frac(table, w, s, &num[j], &den[j], &num[MEIKEY_BATCH + j], &den[MEIKEY_BATCH + j], true);
    }
    batch_invert(den, scratch, 2 * MEIKEY_BATCH);
    for (int j = 0; j < MEIKEY_BATCH; j++) {
        u8* pk = pks + 32 * j;
        fe_to_bytes(fe_mul(num[j], den[j]), pk);
        u8 x[32];
        fe_to_bytes(fe_mul(num[MEIKEY_BATCH + j], den[MEIKEY_BATCH + j]), x);
        pk[31] |= (u8)((x[0] & 1) << 7);
    }
}

#ifdef MEIKEY_CUDA

/// 探索のカーネル。スレッド gid（全 nthreads 本）は、1回に MEIKEY_BATCH 本を `iters` 回。
/// 鍵の通し番号は base + ((it·nthreads + gid)·MEIKEY_BATCH + j)。分母の積はブロックの中の、ワープの同じ
/// レーンの WARPS スレッドでまとめて逆元にする（ワープ 0 が計算し、その間ほかのワープは待つ。SM には
/// ほかのブロックも載っているので、その間も計算は進む）。sink は、判定しないとき（ベンチマーク）に計算が
/// 省かれないよう、各スレッドの結果の一部を書き出す先
// 1 SM に 4 ブロック（512 スレッド）載るよう、レジスタを 128 本以内にする（足りない分は少しスピルする。
// 制限しないと 160〜180 本になり、載るブロックが 2〜3 に減る）
extern "C" __global__ void __launch_bounds__(MEIKEY_BLOCK, 4)
    meikey_search(const u32* __restrict__ table, u32 width, const u32* __restrict__ key, u32 counter,
                  const u8* __restrict__ needle, u32 needle_len, u32 fold, u32 mode, u64 base, u32 iters,
                  u32 stop_on_hit, u64* result, u32* sink) {
    __shared__ fe group[2][MEIKEY_BLOCK];
    // どのスレッドでも同じ値（鍵・SHA-512 の前計算・探す文字列）は共有メモリに置く（レジスタを空ける）
    __shared__ u32 k[8];
    __shared__ ShaMid mid;
    __shared__ Matcher mt;
    u32 tid = threadIdx.x;
    u64 gid = (u64)blockIdx.x * blockDim.x + tid;
    u64 nthreads = (u64)gridDim.x * blockDim.x;
    if (tid < 8) k[tid] = key[tid];
    __syncthreads();
    if (tid == 0) {
        if (counter != 0) mid = sha512_mid(k);
        mt = make_matcher(needle, needle_len, fold != 0, mode);
    }
    __syncthreads();
    Batch b;
    u32 s = 0;
    for (u32 it = 0; it < iters; it++) {
        // 止めるかはブロックでそろえる（ブロックの中で待ち合わせるので）
        if (stop_on_hit && __syncthreads_or(VOLATILE_LOAD64(result) != 0)) break;
        u64 first = base + ((u64)it * nthreads + gid) * MEIKEY_BATCH;
        batch_derive(b, table, width, k, first, counter != 0, mid);
        // 2 組を交互に使う（次の回に書き込むのは、全員が前の回の値を読んだ後）。逆元を計算している間は、
        // もう 1 組は誰も読まないので、途中の積を置く場所にする
        fe* q = group[it & 1];
        q[tid] = b.pre[MEIKEY_BATCH - 1];
        __syncthreads();
        if (tid < 32) invert_strided(q, group[(it + 1) & 1], tid, 32, WARPS);
        __syncthreads();
        report(batch_finish(b, q[tid], mt, &s), first, result);
    }
    sink[gid] = s;
}

/// 確かめる用: 探索のカーネルと同じ手順（ブロックの中で分母の積をまとめて逆元にする）で、スレッド gid が
/// 通し番号 first + gid·MEIKEY_BATCH から MEIKEY_BATCH 本の公開鍵の y（x の符号のビットは 0）を out に書く
/// （ブロックは MEIKEY_BLOCK スレッドで起動する）
extern "C" __global__ void __launch_bounds__(MEIKEY_BLOCK, 4)
    meikey_pooled(const u32* __restrict__ table, u32 width, const u32* __restrict__ key, u32 counter, u64 first,
                  u8* out) {
    __shared__ fe group[2][MEIKEY_BLOCK];
    __shared__ u32 k[8];
    __shared__ ShaMid mid;
    u32 tid = threadIdx.x;
    u64 gid = (u64)blockIdx.x * blockDim.x + tid;
    if (tid < 8) k[tid] = key[tid];
    __syncthreads();
    if (tid == 0 && counter != 0) mid = sha512_mid(k);
    __syncthreads();
    Batch b;
    batch_derive(b, table, width, k, first + gid * MEIKEY_BATCH, counter != 0, mid);
    group[0][tid] = b.pre[MEIKEY_BATCH - 1];
    __syncthreads();
    if (tid < 32) invert_strided(group[0], group[1], tid, 32, WARPS);
    __syncthreads();
    u8* o = out + gid * MEIKEY_BATCH * 32;
    batch_each(b, group[0][tid], [&](int j, const u8 pk[32]) {
        for (int i = 0; i < 32; i++) o[32 * j + i] = pk[i];
    });
}

/// 確かめる・テスト用: スレッド gid が通し番号 first + gid·MEIKEY_BATCH から MEIKEY_BATCH 本の公開鍵を out に書く
extern "C" __global__ void meikey_derive(const u32* __restrict__ table, u32 width, const u32* __restrict__ key,
                                         u32 counter, u64 first, u8* out) {
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u32 k[8];
    UNROLL
    for (int i = 0; i < 8; i++) k[i] = key[i];
    derive_batch(table, width, k, first + gid * MEIKEY_BATCH, counter != 0, out + gid * MEIKEY_BATCH * 32);
}

#else

// ---- テスト用（CPU でコンパイルしたときだけ。service/gpu.rs のテストから呼ぶ） ----
// 名前は MEIKEY_TEST_PREFIX（既定 meikey_test_）+ 関数名。build.rs は埋め込む設定ごとに、MEIKEY_BATCH・
// MEIKEY_FIELD・MEIKEY_UNPACKED と接頭辞を変えて並べてコンパイルする
#ifndef MEIKEY_TEST_PREFIX
#define MEIKEY_TEST_PREFIX meikey_test_
#endif
#define MEIKEY_CAT2(a, b) a##b
#define MEIKEY_CAT(a, b) MEIKEY_CAT2(a, b)
#define TEST_FN(name) MEIKEY_CAT(MEIKEY_TEST_PREFIX, name)

extern "C" void TEST_FN(seed)(const u32 key[8], u64 index, u8 out[32]) { chacha20_seed(key, index, out); }

extern "C" void TEST_FN(counter_seed)(const u32 key[8], u64 index, u8 out[32]) { counter_seed(key, index, out); }

extern "C" void TEST_FN(secret)(const u8 seed[32], u8 out[32]) { secret_scalar(seed, out); }

/// 低セキュリティモードの通し番号 index の秘密スカラー（SHA-512 の前計算を使う）
extern "C" void TEST_FN(secret_mid)(const u32 key[8], u64 index, u8 out[32]) {
    ShaMid mid = sha512_mid(key);
    secret_scalar_mid(mid, key, index, out);
}

/// 幅 w の桁（out は ceil(251 / w) 個）
extern "C" void TEST_FN(digits)(const u8 s[32], u32 w, i32* out) {
    Digits g = digits_of(s, w);
    for (u32 i = 0; i < g.rows; i++) out[i] = next_digit(g);
}

#if MEIKEY_FIELD == 10
// 25.5 ビット × 10 リムのまま渡す（リムの上限の境界を確かめる）
extern "C" void TEST_FN(fe_mul)(const u32 a[10], const u32 b[10], u32 out[10]) {
    fe x, y;
    for (int i = 0; i < 10; i++) { x.v[i] = a[i]; y.v[i] = b[i]; }
    fe r = fe_mul(x, y);
    for (int i = 0; i < 10; i++) out[i] = r.v[i];
}

extern "C" void TEST_FN(fe_sq)(const u32 a[10], u32 out[10]) {
    fe x;
    for (int i = 0; i < 10; i++) x.v[i] = a[i];
    fe r = fe_sq(x);
    for (int i = 0; i < 10; i++) out[i] = r.v[i];
}

extern "C" void TEST_FN(fe_invert_bytes)(const u32 a[10], u8 out[32]) {
    fe x;
    for (int i = 0; i < 10; i++) x.v[i] = a[i];
    fe_to_bytes(fe_invert(x), out);
}
#endif

/// 体演算の値を、表の値と同じ形（32 ビット × 8 語の 2^255 未満か、リムの形）から作る
DEV fe fe_from_words(const u32 a[8]) { return fe_unpack(a); }

// 32 ビット × 8 語で渡し、結果は正規化したバイト列で返す（2^255 未満の値。8 リムでは 2^256 未満の任意の値）。
// op: 0 = 加算、1 = 減算、2 = 乗算、3 = 2乗、4 = 逆元、5 = 加算の結果どうしの乗算（上限の境界）、
// 6 = 減算の結果どうしの乗算（同じく）、7 = 整えてから逆元
extern "C" void TEST_FN(fe_op_words)(u32 op, const u32 a[8], const u32 b[8], u8 out[32]) {
    fe x = fe_from_words(a), y = fe_from_words(b), r;
    switch (op) {
        case 0: r = fe_carry(fe_add(x, y)); break;
        case 1: r = fe_carry(fe_sub(x, y)); break;
        case 2: r = fe_mul(x, y); break;
        case 3: r = fe_sq(x); break;
        case 4: r = fe_invert(x); break;
        case 5: r = fe_mul(fe_add(x, y), fe_add(y, x)); break;
        case 6: r = fe_mul(fe_sub(x, y), fe_sub(y, x)); break;
        default: r = fe_invert(fe_carry(fe_sub(x, y))); break;
    }
    fe_to_bytes(r, out);
}

extern "C" u32 TEST_FN(field)(void) { return MEIKEY_FIELD; }

extern "C" u32 TEST_FN(unpacked)(void) { return MEIKEY_UNPACKED; }

extern "C" u32 TEST_FN(entry_words)(void) { return ENTRY_WORDS; }

extern "C" u32 TEST_FN(block)(void) { return MEIKEY_BLOCK; }

extern "C" u32 TEST_FN(batch)(void) { return MEIKEY_BATCH; }

extern "C" void TEST_FN(derive_batch)(const u32* table, u32 width, const u32 key[8], u32 counter, u64 first,
                                      u8 out[MEIKEY_BATCH][32]) {
    derive_batch(table, width, key, first, counter != 0, &out[0][0]);
}

extern "C" int TEST_FN(contains)(const u8 pk[32], const u8* needle, u32 len, u32 fold, u32 mode) {
    return pk_match(pk, make_matcher(needle, len, fold != 0, mode), false);
}

extern "C" int TEST_FN(contains_unsigned)(const u8 pk[32], const u8* needle, u32 len, u32 fold, u32 mode) {
    return pk_match(pk, make_matcher(needle, len, fold != 0, mode), true);
}

extern "C" void TEST_FN(tail_b64)(const u8 pk[32], u8 out[TAIL_B64]) { tail_b64(pk, false, out); }

/// 32 ビット × 8 語の値 n 個（n ≤ MEIKEY_BLOCK）を、探索のカーネルと同じ組（ワープの同じレーンのスレッド）で
/// まとめて逆元にして、正規化したバイト列で返す
extern "C" void TEST_FN(invert_pooled)(const u32* words, u32 n, u8* out) {
    std::vector<fe> q(n), scratch(n);
    for (u32 i = 0; i < n; i++) q[i] = fe_carry(fe_unpack(words + 8 * i));
    for (u32 lane = 0; lane < 32 && lane < n; lane++) {
        invert_strided(q.data(), scratch.data(), (int)lane, 32, (int)((n - lane + 31) / 32));
    }
    for (u32 i = 0; i < n; i++) fe_to_bytes(q[i], out + 32 * i);
}

/// カーネル全体を CPU で真似る（MEIKEY_BLOCK スレッドずつのブロックを順に回し、ブロックの中では
/// 全スレッドの点を計算してから、分母の積を GPU と同じ組でまとめて逆元にする）
extern "C" void TEST_FN(search)(u64 nthreads, const u32* table, u32 width, const u32 key[8], u32 counter,
                                   const u8* needle, u32 needle_len, u32 fold, u32 mode, u64 base, u32 iters,
                                   u32 stop_on_hit, u64* result) {
    ShaMid mid = sha512_mid(key);
    Matcher mt = make_matcher(needle, needle_len, fold != 0, mode);
    u32 sink = 0;
    for (u64 start = 0; start < nthreads; start += MEIKEY_BLOCK) {
        u64 n = nthreads - start < MEIKEY_BLOCK ? nthreads - start : MEIKEY_BLOCK;
        std::vector<Batch> b(n);
        std::vector<fe> q(MEIKEY_BLOCK), scratch(MEIKEY_BLOCK);
        std::vector<u64> first(n);
        for (u32 it = 0; it < iters; it++) {
            if (stop_on_hit && VOLATILE_LOAD64(result) != 0) break;
            for (u64 t = 0; t < n; t++) {
                first[t] = base + ((u64)it * nthreads + start + t) * MEIKEY_BATCH;
                batch_derive(b[t], table, width, key, first[t], counter != 0, mid);
                q[t] = b[t].pre[MEIKEY_BATCH - 1];
            }
            for (u64 lane = 0; lane < 32 && lane < n; lane++) {
                invert_strided(q.data(), scratch.data(), (int)lane, 32, (int)((n - lane + 31) / 32));
            }
            for (u64 t = 0; t < n; t++) report(batch_finish(b[t], q[t], mt, &sink), first[t], result);
        }
    }
    (void)sink;
}

#endif
