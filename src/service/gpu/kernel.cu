// meikey の GPU カーネル（CUDA）。シードの生成から文字列の判定までを、1スレッドで鍵 MEIKEY_BATCH 本ずつ行う。
//
// 実行時に NVRTC でその GPU 向けにコンパイルする（service/gpu.rs）。テストでは同じソースを
// C++ として CPU 向けにコンパイルし（host.cpp）、Rust の実装と突き合わせる。そのため CUDA 固有の書き方は
// 下のマクロにまとめ、本体は C の整数演算だけで書く。
//
// 1鍵の流れ（CPU 版と同じ計算）:
//   1. シード = ChaCha20(鍵 = ホストが作った乱数, ブロック番号 = 鍵の通し番号) の先頭 32 バイト
//   2. 秘密スカラー = clamp(SHA-512(シード)[..32])
//   3. 公開鍵の点 = 櫛形テーブル（W = 12、ホストが作って転送）を1行ずつ足す（加算 22 回、倍加なし）
//   4. MEIKEY_BATCH 本の 1/Z を逆元1回でまとめて求め、1本ずつ圧縮する
//   5. 圧縮したらすぐ、公開鍵 blob の base64 の後半 44 文字に文字列があるか調べる。見つけたら通し番号を書いて知らせる
//
// 体演算は AVX2 版（avx2.rs）と同じ 25.5 ビット × 10 リム（偶数番目 26 ビット、奇数番目 25 ビット）で、
// 32 × 32 → 64 ビットの積を 64 ビットで足し込む。リムの上限（2^26・2^25 の何倍か）も AVX2 版と同じ:
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
/// 16 バイト境界の4語を1回で読む（読み出し専用のキャッシュを通す）
#define LOAD4(dst, p)                                   \
    {                                                   \
        uint4 q_ = __ldg((const uint4*)(p));            \
        (dst)[0] = q_.x; (dst)[1] = q_.y; (dst)[2] = q_.z; (dst)[3] = q_.w; \
    }
#else
#include <stdint.h>
typedef uint8_t u8;
typedef int32_t i32;
typedef uint32_t u32;
typedef unsigned long long u64;
#define DEV static inline
#define NOINLINE static
#define CONSTANT static const
#define UNROLL
#define LOAD4(dst, p)                                                   \
    {                                                                   \
        (dst)[0] = (p)[0]; (dst)[1] = (p)[1]; (dst)[2] = (p)[2]; (dst)[3] = (p)[3]; \
    }
#endif

// 櫛形テーブル（service/derive/comb.rs と同じ作り方。ホストが W を -D で渡す）
#ifndef MEIKEY_W
#define MEIKEY_W 12
#endif
// 桁は -2^(W-1)〜2^(W-1)。最上位の桁が収まる（255 - W·(行数 - 1) < W）のも W ≤ 16 まで確かめてある
static_assert(MEIKEY_W >= 2 && MEIKEY_W <= 16, "MEIKEY_W は 2〜16");
#define ROWS (255 / MEIKEY_W + 1)
#define COLS ((1 << (MEIKEY_W - 1)) + 1)
#define HALF (1 << (MEIKEY_W - 1))
// 表の1エントリ: y+x, y-x, 2dxy の 10 リムずつと未使用の 2 語（128 バイト）
#define ENTRY_WORDS 32

// 1スレッドが1回にまとめて計算する鍵の数（逆元をまとめる数。当たりを 64 ビットに集めるので 64 まで）
#ifndef MEIKEY_BATCH
#define MEIKEY_BATCH 16
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

/// 鍵 `key`、ブロック番号 `counter`（ストリーム 0）の出力の先頭 32 バイト
DEV void chacha20_seed(const u32 key[8], u64 counter, u8 seed[32]) {
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
    for (int i = 0; i < 8; i++) {
        u32 w = x[i] + s[i];
        seed[4 * i] = (u8)w;
        seed[4 * i + 1] = (u8)(w >> 8);
        seed[4 * i + 2] = (u8)(w >> 16);
        seed[4 * i + 3] = (u8)(w >> 24);
    }
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

/// `clamp(SHA-512(seed)[..32])`（RFC 8032 5.1.5 の手順 1・2）
DEV void secret_scalar(const u8 seed[32], u8 out[32]) {
    u64 w[16];
    UNROLL
    for (int i = 0; i < 4; i++) {
        u64 v = 0;
        UNROLL
        for (int b = 0; b < 8; b++) v = (v << 8) | seed[8 * i + b];
        w[i] = v;
    }
    w[4] = 0x8000000000000000ull;
    UNROLL
    for (int i = 5; i < 15; i++) w[i] = 0;
    w[15] = 256;

    u64 a = SHA512_IV[0], b = SHA512_IV[1], c = SHA512_IV[2], d = SHA512_IV[3];
    u64 e = SHA512_IV[4], f = SHA512_IV[5], g = SHA512_IV[6], h = SHA512_IV[7];
    for (int t = 0; t < 80; t++) {
        u64 wt;
        if (t < 16) {
            wt = w[t];
        } else {
            u64 w15 = w[(t + 1) & 15], w2 = w[(t + 14) & 15];
            u64 s0 = ROTR64(w15, 1) ^ ROTR64(w15, 8) ^ (w15 >> 7);
            u64 s1 = ROTR64(w2, 19) ^ ROTR64(w2, 61) ^ (w2 >> 6);
            wt = w[t & 15] + s0 + w[(t + 9) & 15] + s1;
            w[t & 15] = wt;
        }
        u64 S1 = ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41);
        u64 ch = (e & f) ^ (~e & g);
        u64 t1 = h + S1 + ch + SHA512_K[t] + wt;
        u64 S0 = ROTR64(a, 28) ^ ROTR64(a, 34) ^ ROTR64(a, 39);
        u64 maj = (a & b) ^ (a & c) ^ (b & c);
        u64 t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    u64 hs[4] = {a + SHA512_IV[0], b + SHA512_IV[1], c + SHA512_IV[2], d + SHA512_IV[3]};
    UNROLL
    for (int i = 0; i < 4; i++) {
        UNROLL
        for (int k = 0; k < 8; k++) out[8 * i + k] = (u8)(hs[i] >> (56 - 8 * k));
    }
    out[0] &= 248;
    out[31] &= 127;
    out[31] |= 64;
}

// ---------------------------------------------------------------------------
// 符号付き 2^W 進の桁（comb.rs の signed_digits と同じ）
// ---------------------------------------------------------------------------

DEV void signed_digits(const u8 s[32], i32 out[ROWS]) {
    u64 words[5];
    UNROLL
    for (int i = 0; i < 4; i++) {
        u64 v = 0;
        UNROLL
        for (int b = 7; b >= 0; b--) v = (v << 8) | s[8 * i + b];
        words[i] = v;
    }
    words[4] = 0;
    i32 carry = 0;
    UNROLL
    for (int i = 0; i < ROWS; i++) {
        int q = i * MEIKEY_W / 64, r = i * MEIKEY_W % 64;
        u64 bits = words[q] >> r;
        if (r + MEIKEY_W > 64) bits |= words[q + 1] << (64 - r);
        i32 v = (i32)(bits & ((1u << MEIKEY_W) - 1)) + carry;
        if (i == ROWS - 1) {
            out[i] = v;
        } else {
            carry = (v + HALF) >> MEIKEY_W;
            out[i] = v - (carry << MEIKEY_W);
        }
    }
}

// ---------------------------------------------------------------------------
// 体 GF(2^255 - 19): 25.5 ビット × 10 リム
// ---------------------------------------------------------------------------

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

DEV fe fe_pow2k(fe a, int k) {
    for (int i = 0; i < k; i++) a = fe_sq(a);
    return a;
}

/// 1/a = a^(p-2)（field.rs の invert と同じ加算連鎖）
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

#define MASK51 0x7ffffffffffffull

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

// ---------------------------------------------------------------------------
// 点の演算（comb.rs・avx2.rs と同じ式）
// ---------------------------------------------------------------------------

typedef struct { fe x, y, z, t; } ext;
typedef struct { fe ypx, ymx, xy2d; } niels;

/// 表の i 行目の桁 d の点（負の桁は y+x と y-x を入れ替え、2dxy を反転）。
/// エントリは 128 バイト境界にあるので、16 バイトずつ読む（未使用の 2 語も読む）
DEV niels table_entry(const u32* table, int i, int d) {
    const u32* e = table + ((u64)i * COLS + (u32)(d < 0 ? -d : d)) * ENTRY_WORDS;
    u32 w[ENTRY_WORDS];
    UNROLL
    for (int q = 0; q < ENTRY_WORDS / 4; q++) LOAD4(w + 4 * q, e + 4 * q);
    niels n;
    UNROLL
    for (int k = 0; k < 10; k++) {
        n.ypx.v[k] = w[k];
        n.ymx.v[k] = w[10 + k];
        n.xy2d.v[k] = w[20 + k];
    }
    if (d < 0) {
        fe t = n.ypx;
        n.ypx = n.ymx;
        n.ymx = t;
        fe zero;
        UNROLL
        for (int k = 0; k < 10; k++) zero.v[k] = 0;
        n.xy2d = fe_sub(zero, n.xy2d);
    }
    return n;
}

/// 拡張座標に（X = 2x, Y = 2y, Z = 2, T = 2xy）
DEV ext to_extended(const niels n) {
    // 1/2
    const fe inv2 = {{67108855u, 33554431u, 67108863u, 33554431u, 67108863u,
                      33554431u, 67108863u, 33554431u, 67108863u, 16777215u}};
    ext p;
    p.x = fe_carry(fe_sub(n.ypx, n.ymx));
    p.y = fe_carry(fe_add(n.ypx, n.ymx));
    UNROLL
    for (int k = 0; k < 10; k++) p.z.v[k] = 0;
    p.z.v[0] = 2;
    p.t = fe_mul(fe_mul(p.x, p.y), inv2);
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

/// 秘密スカラー s の s·B（T は計算しない）
DEV ext mul_base(const u32* table, const u8 s[32]) {
    i32 d[ROWS];
    signed_digits(s, d);
    ext acc = to_extended(table_entry(table, 0, d[0]));
    for (int i = 1; i < ROWS - 1; i++) acc = add_niels(acc, table_entry(table, i, d[i]), true);
    return add_niels(acc, table_entry(table, ROWS - 1, d[ROWS - 1]), false);
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

/// 候補の位置（`candidates` のビット）のどれかから needle 全体が一致するか。候補が見つかったときだけ呼ぶので、
/// base64 を作り直して配列で比べる
NOINLINE bool b64_contains_at(const u8 pk[32], bool fold, const u8* needle, u32 len, u64 candidates) {
    u8 b64[TAIL_B64];
    tail_b64(pk, fold, b64);
    while (candidates != 0) {
        u32 s = 0;
        while (((candidates >> s) & 1) == 0) s++;
        candidates &= candidates - 1;
        u32 k = 0;
        while (k < len && b64[s + k] == needle[k]) k++;
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

/// needle の1文字に一致する 6 ビットの値の組。大小文字を区別しない（fold）なら needle は小文字なので、
/// 英字は小文字と大文字の2つ。それ以外は同じ値を2つ
DEV void needle_values(u8 c, bool fold, u32 v[2]) {
    v[0] = b64_value(c);
    v[1] = fold && c >= 'a' && c <= 'z' ? v[0] - 26 : v[0];
}

/// pk の base64 の判定の範囲（後半の 1 文字目から）に needle があるか。needle は fold なら小文字にしてある。
/// 文字にはせず 6 ビットの値のまま、全部の位置で先頭 2 文字（1 文字なら 1 文字）を比べて候補を絞り、
/// 候補があるときだけ残りを比べる
DEV bool pk_contains(const u8 pk[32], const u8* needle, u32 len, bool fold) {
    if (len == 0 || len > TAIL_B64 - SEARCH_FROM) return false;
    u32 c0[2], c1[2];
    needle_values(needle[0], fold, c0);
    needle_values(len >= 2 ? needle[1] : needle[0], fold, c1);
    // m0・m1 のビット s: s 文字目が needle の 1・2 文字目と一致する
    u64 m0 = 0, m1 = 0;
    UNROLL
    for (int g = 0; g < 11; g++) {
        u32 b0 = g == 0 ? 32u : pk[3 * g - 1];
        u32 n = b0 << 16 | (u32)pk[3 * g] << 8 | pk[3 * g + 1];
        UNROLL
        for (int k = 0; k < 4; k++) {
            u32 v = (n >> (18 - 6 * k)) & 63;
            m0 |= (u64)(v == c0[0] || v == c0[1]) << (4 * g + k);
            m1 |= (u64)(v == c1[0] || v == c1[1]) << (4 * g + k);
        }
    }
    u64 candidates = len >= 2 ? m0 & (m1 >> 1) : m0;
    // 判定の範囲（SEARCH_FROM から）で、needle が収まる位置（s + len ≤ 44）だけ
    candidates &= ((2ull << (TAIL_B64 - len)) - 1) & ~((1ull << SEARCH_FROM) - 1);
    if (candidates == 0) return false;
    if (len <= 2) return true;
    return b64_contains_at(pk, fold, needle, len, candidates);
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

/// 通し番号 `first` から MEIKEY_BATCH 本の公開鍵（圧縮形式）を計算し、1本できるたびに emit(j, pk) を呼ぶ
/// （j は 0 始まりの番号。後ろの鍵から順に呼ぶ）
template <class Emit>
DEV void derive_batch_each(const u32* table, const u32 key[8], u64 first, Emit emit) {
    fe xs[MEIKEY_BATCH], ys[MEIKEY_BATCH], zs[MEIKEY_BATCH], prefix[MEIKEY_BATCH];
    for (int j = 0; j < MEIKEY_BATCH; j++) {
        u8 seed[32], s[32];
        chacha20_seed(key, first + (u64)j, seed);
        secret_scalar(seed, s);
        ext p = mul_base(table, s);
        xs[j] = p.x;
        ys[j] = p.y;
        zs[j] = p.z;
        prefix[j] = j == 0 ? p.z : fe_mul(prefix[j - 1], p.z);
    }
    // Montgomery の一括逆元: 全部の積の逆元から、後ろの鍵から順に 1/Z を取り出す
    fe inv = fe_invert(prefix[MEIKEY_BATCH - 1]);
    for (int j = MEIKEY_BATCH - 1; j >= 0; j--) {
        fe zinv = j == 0 ? inv : fe_mul(inv, prefix[j - 1]);
        if (j > 0) inv = fe_mul(inv, zs[j]);
        u8 pk[32], x[32];
        fe_to_bytes(fe_mul(ys[j], zinv), pk);
        fe_to_bytes(fe_mul(xs[j], zinv), x);
        pk[31] |= (u8)((x[0] & 1) << 7);
        emit(j, pk);
    }
}

/// 通し番号 `first` から MEIKEY_BATCH 本の公開鍵（圧縮形式）を pks に書く
DEV void derive_batch(const u32* table, const u32 key[8], u64 first, u8 pks[MEIKEY_BATCH][32]) {
    derive_batch_each(table, key, first, [&](int j, const u8 pk[32]) {
        for (int b = 0; b < 32; b++) pks[j][b] = pk[b];
    });
}

/// スレッド `gid`（全 `nthreads` 本）の仕事: 1回に MEIKEY_BATCH 本を `iters` 回。
/// 鍵の通し番号は base + ((it·nthreads + gid)·MEIKEY_BATCH + j)。
/// 見つけたら result[0] を 0 → 1 にした最初のスレッドが result[1] に通し番号を書く
DEV u32 thread_work(u64 gid, u64 nthreads, const u32* table, const u32 key[8], const u8* needle,
                    u32 needle_len, u32 fold, u64 base, u32 iters, u32 stop_on_hit, u64* result) {
    u32 sink = 0;
    for (u32 it = 0; it < iters; it++) {
        if (stop_on_hit && VOLATILE_LOAD64(result) != 0) break;
        u64 first = base + ((u64)it * nthreads + gid) * MEIKEY_BATCH;
        // 当たった鍵の番号 j をビットで集め、まとめて計算し終えてから、いちばん前の当たりを書く
        // （公開鍵は後ろの鍵から順にできる）
        u64 hits = 0;
        derive_batch_each(table, key, first, [&](int j, const u8 pk[32]) {
            sink ^= pk[31];
            hits |= (u64)pk_contains(pk, needle, needle_len, fold != 0) << j;
        });
        if (hits != 0 && ATOMIC_CAS64(result, 0ull, 1ull) == 0ull) {
            u32 j = 0;
            while (((hits >> j) & 1) == 0) j++;
            result[1] = first + j;
        }
    }
    return sink;
}

#ifdef MEIKEY_CUDA

/// 探索のカーネル。sink は、判定しないとき（ベンチマーク）に計算が省かれないよう、各スレッドの結果の
/// 一部を書き出す先
extern "C" __global__ void meikey_search(const u32* __restrict__ table, const u32* __restrict__ key,
                                         const u8* __restrict__ needle, u32 needle_len, u32 fold, u64 base,
                                         u32 iters, u32 stop_on_hit, u64* result, u32* sink) {
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 nthreads = (u64)gridDim.x * blockDim.x;
    u32 k[8];
    UNROLL
    for (int i = 0; i < 8; i++) k[i] = key[i];
    sink[gid] = thread_work(gid, nthreads, table, k, needle, needle_len, fold, base, iters, stop_on_hit, result);
}

/// テスト用: スレッド gid が通し番号 first + gid·MEIKEY_BATCH から MEIKEY_BATCH 本の公開鍵を out に書く
extern "C" __global__ void meikey_derive(const u32* __restrict__ table, const u32* __restrict__ key, u64 first,
                                         u8* out) {
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u32 k[8];
    UNROLL
    for (int i = 0; i < 8; i++) k[i] = key[i];
    u8 pks[MEIKEY_BATCH][32];
    derive_batch(table, k, first + gid * MEIKEY_BATCH, pks);
    for (int j = 0; j < MEIKEY_BATCH; j++) {
        for (int b = 0; b < 32; b++) out[(gid * MEIKEY_BATCH + j) * 32 + b] = pks[j][b];
    }
}

#else

// ---- テスト用（CPU でコンパイルしたときだけ。service/gpu.rs のテストから呼ぶ） ----
// 名前は MEIKEY_TEST_PREFIX（既定 meikey_test_）+ 関数名。build.rs は埋め込む設定ごとに、W・MEIKEY_BATCH と
// 接頭辞を変えて並べてコンパイルする
#ifndef MEIKEY_TEST_PREFIX
#define MEIKEY_TEST_PREFIX meikey_test_
#endif
#define MEIKEY_CAT2(a, b) a##b
#define MEIKEY_CAT(a, b) MEIKEY_CAT2(a, b)
#define TEST_FN(name) MEIKEY_CAT(MEIKEY_TEST_PREFIX, name)

extern "C" void TEST_FN(seed)(const u32 key[8], u64 counter, u8 out[32]) { chacha20_seed(key, counter, out); }

extern "C" void TEST_FN(secret)(const u8 seed[32], u8 out[32]) { secret_scalar(seed, out); }

extern "C" void TEST_FN(digits)(const u8 s[32], i32 out[ROWS]) { signed_digits(s, out); }

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

extern "C" void TEST_FN(derive_batch)(const u32* table, const u32 key[8], u64 first, u8 out[MEIKEY_BATCH][32]) {
    derive_batch(table, key, first, out);
}

extern "C" int TEST_FN(contains)(const u8 pk[32], const u8* needle, u32 len, u32 fold) {
    return pk_contains(pk, needle, len, fold != 0);
}

extern "C" void TEST_FN(tail_b64)(const u8 pk[32], u8 out[TAIL_B64]) { tail_b64(pk, false, out); }

/// カーネル全体を CPU で真似る（スレッドを順に回す）
extern "C" void TEST_FN(search)(u64 nthreads, const u32* table, const u32 key[8], const u8* needle, u32 needle_len,
                                   u32 fold, u64 base, u32 iters, u32 stop_on_hit, u64* result) {
    for (u64 gid = 0; gid < nthreads; gid++) {
        thread_work(gid, nthreads, table, key, needle, needle_len, fold, base, iters, stop_on_hit, result);
    }
}

extern "C" u32 TEST_FN(batch)(void) { return MEIKEY_BATCH; }

extern "C" u32 TEST_FN(width)(void) { return MEIKEY_W; }

#endif
