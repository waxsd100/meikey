// meikey の GPU カーネル（CUDA）。シードの生成から文字列の判定までを、1スレッドで鍵 MEIKEY_BATCH 本ずつ行う。
//
// 実行時に NVRTC でその GPU 向けにコンパイルする（service/gpu.rs）。テストでは同じソースを
// C++ として CPU 向けにコンパイルし（host.cpp）、Rust の実装と突き合わせる。そのため CUDA 固有の書き方は
// 下のマクロにまとめ、本体は C の整数演算だけで書く。
//
// 1鍵の流れ（CPU 版と同じ計算）:
//   1. シード = ChaCha20(鍵 = ホストが作った乱数, ブロック番号 = 鍵の通し番号 / 2) の前半か後半の 32 バイト
//      （通し番号が偶数なら前半、奇数なら後半。1 ブロックで 2 本ぶん）。低セキュリティモード（counter）では、
//      ホストが作った乱数 32 バイトの先頭 8 バイトに通し番号を足したもの
//   2. 秘密スカラー = clamp(SHA-512(シード)[..32])
//   3. 公開鍵の点 = 櫛形テーブル（幅 W、ホストが作って転送）を1行ずつ足す（倍加なし。W = 16 で加算 15 回）。
//      スカラー s = 8·(2^251 + m) の m を桁に分け、表は 8B を基点にして最上位の行に 2^254·B を足してある
//   4. MEIKEY_BATCH 本の 1/Z を逆元1回でまとめて求め、1本ずつ圧縮する（y だけ。x の符号は 5 で要るときだけ）
//   5. 圧縮したらすぐ、公開鍵 blob の base64 の後半 44 文字に文字列があるか調べる。x の符号で変わる1文字は
//      どちらの値とも比べ、含まれそうなときだけ符号を計算して確かめ直す。見つけたら通し番号を書いて知らせる
//
// 体演算は MEIKEY_LIMBS で選ぶ。10（既定）は AVX2 版（avx2.rs）と同じ 25.5 ビット × 10 リム（偶数番目 26 ビット、
// 奇数番目 25 ビット）で、32 × 32 → 64 ビットの積を 64 ビットで足し込む。8 は 32 ビット × 8 リムで、桁上がりつきの
// 命令で計算する（体演算の節を参照）。10 リムのリムの上限（2^26・2^25 の何倍か）は AVX2 版と同じ:
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
#define NO_UNROLL
#define LOAD4(dst, p)                                                   \
    {                                                                   \
        (dst)[0] = (p)[0]; (dst)[1] = (p)[1]; (dst)[2] = (p)[2]; (dst)[3] = (p)[3]; \
    }
#endif

// 櫛形テーブル（service/derive/comb.rs と同じ作り方。ホストが W を -D で渡す）
#ifndef MEIKEY_W
#define MEIKEY_W 12
#endif
// 桁は -2^(W-1)〜2^(W-1)。m は 251 ビットで、行数は ceil(251 / W)（comb.rs の rows_cols と同じ）
static_assert(MEIKEY_W >= 2 && MEIKEY_W <= 20, "MEIKEY_W は 2〜20");
#define M_BITS 251
#define ROWS ((M_BITS + MEIKEY_W - 1) / MEIKEY_W)
// 最上位の桁は W - 1 ビット以下（繰り上がりを足しても 2^(W-1) 以下）
static_assert(M_BITS - MEIKEY_W * (ROWS - 1) < MEIKEY_W, "最上位の桁が表に収まらない");
#define COLS ((1 << (MEIKEY_W - 1)) + 1)
#define HALF (1 << (MEIKEY_W - 1))
// 表の1エントリ: y+x, y-x, 2dxy（1 行目は 2xy）を 0 以上 p 未満にして、32 ビット × 8 語ずつ並べたもの（96 バイト）
#define ENTRY_WORDS 24

// 1ブロックのスレッド数（service/gpu.rs の BLOCK と同じ）
#define MEIKEY_BLOCK 128

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

/// ブロック `block` の前半（`upper` なら後半）の 32 バイトをシードに
DEV void seed_from_block(const u32 block[16], bool upper, u8 seed[32]) {
    UNROLL
    for (int i = 0; i < 8; i++) {
        u32 w = upper ? block[8 + i] : block[i];
        seed[4 * i] = (u8)w;
        seed[4 * i + 1] = (u8)(w >> 8);
        seed[4 * i + 2] = (u8)(w >> 16);
        seed[4 * i + 3] = (u8)(w >> 24);
    }
}

/// 通し番号 `index` のシード（ブロック index / 2 の、index が偶数なら前半・奇数なら後半）
DEV void chacha20_seed(const u32 key[8], u64 index, u8 seed[32]) {
    u32 block[16];
    chacha20_block(key, index >> 1, block);
    seed_from_block(block, (index & 1) != 0, seed);
}

/// 低セキュリティモードのシード: `key`（ホストが作った乱数 32 バイト）の先頭 8 バイト（リトルエンディアン）に
/// 通し番号を足したもの（model::search の counter_seed と同じ）
DEV void counter_seed(const u32 key[8], u64 index, u8 seed[32]) {
    u64 head = ((u64)key[1] << 32 | key[0]) + index;
    UNROLL
    for (int i = 0; i < 8; i++) {
        u32 w = i == 0 ? (u32)head : i == 1 ? (u32)(head >> 32) : key[i];
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
    // 展開すると w の添字が定数になり、w をレジスタに置ける（展開しないとローカルメモリに置かれ、
    // ラウンドごとに読み書きと分岐が入る）。w[4..16] の定数も畳み込まれる
    UNROLL
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

/// クランプ済みのスカラー s = 8·(2^251 + m) の m（s の 3〜253 ビット目）を桁に分ける
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
    UNROLL
    for (int i = 0; i < 4; i++) words[i] = words[i] >> 3 | words[i + 1] << 61;
    words[3] &= (1ull << (M_BITS - 192)) - 1;
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
// 体 GF(2^255 - 19)。MEIKEY_LIMBS が 10 なら 25.5 ビット × 10 リム、8 なら 32 ビット × 8 リム
// （どちらが速いかは GPU で違うので、両方を埋め込んで起動時に測る）
// ---------------------------------------------------------------------------

#ifndef MEIKEY_LIMBS
#define MEIKEY_LIMBS 10
#endif
static_assert(MEIKEY_LIMBS == 10 || MEIKEY_LIMBS == 8, "MEIKEY_LIMBS は 10 か 8");
#define FE_LIMBS MEIKEY_LIMBS

#if MEIKEY_LIMBS == 10

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

#else

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
#endif

DEV fe fe_sq(const fe a) { return fe_mul(a, a); }

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
DEV void fe_to_bytes(const fe a, u8 out[32]) {
    // 2^255 以上の分を 19 倍して下に足す（2^255 ≡ 19）。これで 2^255 + 19 未満
    u32 w[8], t[8];
    u64 c = (u64)(a.v[7] >> 31) * 19;
    UNROLL
    for (int i = 0; i < 8; i++) {
        c += i == 7 ? (a.v[7] & 0x7fffffffu) : a.v[i];
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

#endif

/// a^(2^k)。k は最大 100 なので、展開すると 2 乗が 100 個並んでコードが大きくなるだけ（8 リムでは、
/// インラインアセンブリの 2 乗を展開してしまい PTX が数倍になる）
DEV fe fe_pow2k(fe a, int k) {
    NO_UNROLL
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

// ---------------------------------------------------------------------------
// 点の演算（comb.rs・avx2.rs と同じ式）
// ---------------------------------------------------------------------------

typedef struct { fe x, y, z, t; } ext;
typedef struct { fe ypx, ymx, xy2d; } niels;

/// 表の i 行目の桁 d の点（負の桁は y+x と y-x を入れ替え、2dxy を反転）。
/// エントリは 96 バイト（16 バイト境界）なので、16 バイトずつ読む。1 行目の 3 つ目は 2xy（to_extended 用）
DEV niels table_entry(const u32* table, int i, int d) {
    const u32* e = table + ((u64)i * COLS + (u32)(d < 0 ? -d : d)) * ENTRY_WORDS;
    u32 w[ENTRY_WORDS];
    UNROLL
    for (int q = 0; q < ENTRY_WORDS / 4; q++) LOAD4(w + 4 * q, e + 4 * q);
    niels n;
    n.ypx = fe_unpack(w);
    n.ymx = fe_unpack(w + 8);
    n.xy2d = fe_unpack(w + 16);
    if (d < 0) {
        fe t = n.ypx;
        n.ypx = n.ymx;
        n.ymx = t;
        fe zero;
        UNROLL
        for (int k = 0; k < FE_LIMBS; k++) zero.v[k] = 0;
        n.xy2d = fe_sub(zero, n.xy2d);
    }
    return n;
}

/// 表の 1 行目の点（3 つ目が 2xy）を拡張座標に（X = 2x, Y = 2y, Z = 2, T = 2xy。乗算なし）
DEV ext to_extended(const niels n) {
    ext p;
    p.x = fe_carry(fe_sub(n.ypx, n.ymx));
    p.y = fe_carry(fe_add(n.ypx, n.ymx));
    UNROLL
    for (int k = 0; k < FE_LIMBS; k++) p.z.v[k] = 0;
    p.z.v[0] = 2;
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

/// needle の1文字に一致する 6 ビットの値の組。大小文字を区別しない（fold）なら needle は小文字なので、
/// 英字は小文字と大文字の2つ。それ以外は同じ値を2つ
DEV void needle_values(u8 c, bool fold, u32 v[2]) {
    v[0] = b64_value(c);
    v[1] = fold && c >= 'a' && c <= 'z' ? v[0] - 26 : v[0];
}

/// pk の base64 の判定の範囲（後半の 1 文字目から）に needle があるか。needle は fold なら小文字にしてある。
/// 文字にはせず 6 ビットの値のまま、全部の位置で先頭 2 文字（1 文字なら 1 文字）を比べて候補を絞り、
/// 候補があるときだけ残りを比べる。`sign_unknown` なら pk の x の符号のビットは 0 にしてあり、符号が 0 でも
/// 1 でも含まれる可能性があれば true（含まれる鍵はかならず true）
DEV bool pk_contains(const u8 pk[32], const u8* needle, u32 len, bool fold, bool sign_unknown) {
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
            // 符号が分からなければ、SIGN_CHAR は符号が 1 のときの値（ビット 1 を立てたもの）とも比べる
            u32 w = sign_unknown && 4 * g + k == SIGN_CHAR ? v | 2 : v;
            m0 |= (u64)(v == c0[0] || v == c0[1] || w == c0[0] || w == c0[1]) << (4 * g + k);
            m1 |= (u64)(v == c1[0] || v == c1[1] || w == c1[0] || w == c1[1]) << (4 * g + k);
        }
    }
    u64 candidates = len >= 2 ? m0 & (m1 >> 1) : m0;
    // 判定の範囲（SEARCH_FROM から）で、needle が収まる位置（s + len ≤ 44）だけ
    candidates &= ((2ull << (TAIL_B64 - len)) - 1) & ~((1ull << SEARCH_FROM) - 1);
    if (candidates == 0) return false;
    if (len <= 2) return true;
    return b64_contains_at(pk, fold, needle, len, candidates, sign_unknown);
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

/// x/Z の符号（最下位ビット）
DEV u8 x_sign(const fe x, const fe zinv) {
    u8 b[32];
    fe_to_bytes(fe_mul(x, zinv), b);
    return b[0] & 1;
}

/// 通し番号 `first` から MEIKEY_BATCH 本の公開鍵を計算し、1本できるたびに emit(j, pk, x, zinv) を呼ぶ
/// （j は 0 始まりの番号。後ろの鍵から順に呼ぶ）。pk は圧縮形式の x の符号のビットを 0 にしたもので、
/// 符号は要るときだけ x_sign(x, zinv) で計算する。`counter` ならシードは counter_seed（低セキュリティモード）、
/// そうでなければ ChaCha20
template <class Emit>
DEV void derive_batch_each(const u32* table, const u32 key[8], u64 first, bool counter, Emit emit) {
    fe xs[MEIKEY_BATCH], ys[MEIKEY_BATCH], zs[MEIKEY_BATCH], prefix[MEIKEY_BATCH];
    // 偶数の通し番号でブロックを作り、次の奇数の通し番号ではその後半を使う
    u32 block[16];
    for (int j = 0; j < MEIKEY_BATCH; j++) {
        u8 seed[32], s[32];
        u64 index = first + (u64)j;
        if (counter) {
            counter_seed(key, index, seed);
        } else {
            if (j == 0 || (index & 1) == 0) chacha20_block(key, index >> 1, block);
            seed_from_block(block, (index & 1) != 0, seed);
        }
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
        u8 pk[32];
        fe_to_bytes(fe_mul(ys[j], zinv), pk);
        emit(j, pk, xs[j], zinv);
    }
}

/// 通し番号 `first` から MEIKEY_BATCH 本の公開鍵（圧縮形式）を pks に書く
DEV void derive_batch(const u32* table, const u32 key[8], u64 first, bool counter, u8 pks[MEIKEY_BATCH][32]) {
    derive_batch_each(table, key, first, counter, [&](int j, u8 pk[32], const fe& x, const fe& zinv) {
        pk[31] |= (u8)(x_sign(x, zinv) << 7);
        for (int b = 0; b < 32; b++) pks[j][b] = pk[b];
    });
}

/// スレッド `gid`（全 `nthreads` 本）の仕事: 1回に MEIKEY_BATCH 本を `iters` 回。
/// 鍵の通し番号は base + ((it·nthreads + gid)·MEIKEY_BATCH + j)。
/// 見つけたら result[0] を 0 → 1 にした最初のスレッドが result[1] に通し番号を書く
DEV u32 thread_work(u64 gid, u64 nthreads, const u32* table, const u32 key[8], u32 counter, const u8* needle,
                    u32 needle_len, u32 fold, u64 base, u32 iters, u32 stop_on_hit, u64* result) {
    u32 sink = 0;
    for (u32 it = 0; it < iters; it++) {
        if (stop_on_hit && VOLATILE_LOAD64(result) != 0) break;
        u64 first = base + ((u64)it * nthreads + gid) * MEIKEY_BATCH;
        // 当たった鍵の番号 j をビットで集め、まとめて計算し終えてから、いちばん前の当たりを書く
        // （公開鍵は後ろの鍵から順にできる）
        u64 hits = 0;
        // x の符号は、文字列が含まれそうな鍵だけ計算する（ほとんどの鍵は符号によらず含まれない）
        derive_batch_each(table, key, first, counter != 0, [&](int j, u8 pk[32], const fe& x, const fe& zinv) {
            sink ^= pk[31];
            bool hit = pk_contains(pk, needle, needle_len, fold != 0, true);
            if (hit) {
                pk[31] |= (u8)(x_sign(x, zinv) << 7);
                hit = pk_contains(pk, needle, needle_len, fold != 0, false);
            }
            hits |= (u64)hit << j;
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
// 1 SM に 4 ブロック（512 スレッド）載るよう、レジスタを 128 本以内にする（足りない分は少しスピルする。
// 制限しないと 160〜180 本になり、載るブロックが 2〜3 に減る）
extern "C" __global__ void __launch_bounds__(MEIKEY_BLOCK, 4)
    meikey_search(const u32* __restrict__ table, const u32* __restrict__ key, u32 counter,
                  const u8* __restrict__ needle, u32 needle_len, u32 fold, u64 base, u32 iters, u32 stop_on_hit,
                  u64* result, u32* sink) {
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 nthreads = (u64)gridDim.x * blockDim.x;
    u32 k[8];
    UNROLL
    for (int i = 0; i < 8; i++) k[i] = key[i];
    sink[gid] = thread_work(gid, nthreads, table, k, counter, needle, needle_len, fold, base, iters, stop_on_hit,
                            result);
}

/// テスト用: スレッド gid が通し番号 first + gid·MEIKEY_BATCH から MEIKEY_BATCH 本の公開鍵を out に書く
extern "C" __global__ void meikey_derive(const u32* __restrict__ table, const u32* __restrict__ key, u32 counter,
                                         u64 first, u8* out) {
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u32 k[8];
    UNROLL
    for (int i = 0; i < 8; i++) k[i] = key[i];
    u8 pks[MEIKEY_BATCH][32];
    derive_batch(table, k, first + gid * MEIKEY_BATCH, counter != 0, pks);
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

extern "C" void TEST_FN(seed)(const u32 key[8], u64 index, u8 out[32]) { chacha20_seed(key, index, out); }

extern "C" void TEST_FN(counter_seed)(const u32 key[8], u64 index, u8 out[32]) { counter_seed(key, index, out); }

extern "C" void TEST_FN(secret)(const u8 seed[32], u8 out[32]) { secret_scalar(seed, out); }

extern "C" void TEST_FN(digits)(const u8 s[32], i32 out[ROWS]) { signed_digits(s, out); }

#if MEIKEY_LIMBS == 10
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

// 32 ビット × 8 語で渡し、結果は正規化したバイト列で返す（どちらのリムの数でも使える。10 リムでは
// 2^255 未満の値だけ、8 リムでは 2^256 未満の任意の値）。op: 0 = 加算、1 = 減算、2 = 乗算、3 = 2乗、4 = 逆元
extern "C" void TEST_FN(fe_op_words)(u32 op, const u32 a[8], const u32 b[8], u8 out[32]) {
    fe x = fe_unpack(a), y = fe_unpack(b), r;
    switch (op) {
        case 0: r = fe_carry(fe_add(x, y)); break;
        case 1: r = fe_carry(fe_sub(x, y)); break;
        case 2: r = fe_mul(x, y); break;
        case 3: r = fe_sq(x); break;
        default: r = fe_invert(x); break;
    }
    fe_to_bytes(r, out);
}

extern "C" u32 TEST_FN(limbs)(void) { return MEIKEY_LIMBS; }

extern "C" u32 TEST_FN(block)(void) { return MEIKEY_BLOCK; }

extern "C" void TEST_FN(derive_batch)(const u32* table, const u32 key[8], u32 counter, u64 first,
                                      u8 out[MEIKEY_BATCH][32]) {
    derive_batch(table, key, first, counter != 0, out);
}

extern "C" int TEST_FN(contains)(const u8 pk[32], const u8* needle, u32 len, u32 fold) {
    return pk_contains(pk, needle, len, fold != 0, false);
}

extern "C" int TEST_FN(contains_unsigned)(const u8 pk[32], const u8* needle, u32 len, u32 fold) {
    return pk_contains(pk, needle, len, fold != 0, true);
}

extern "C" void TEST_FN(tail_b64)(const u8 pk[32], u8 out[TAIL_B64]) { tail_b64(pk, false, out); }

/// カーネル全体を CPU で真似る（スレッドを順に回す）
extern "C" void TEST_FN(search)(u64 nthreads, const u32* table, const u32 key[8], u32 counter, const u8* needle,
                                   u32 needle_len, u32 fold, u64 base, u32 iters, u32 stop_on_hit, u64* result) {
    for (u64 gid = 0; gid < nthreads; gid++) {
        thread_work(gid, nthreads, table, key, counter, needle, needle_len, fold, base, iters, stop_on_hit, result);
    }
}

extern "C" u32 TEST_FN(batch)(void) { return MEIKEY_BATCH; }

extern "C" u32 TEST_FN(width)(void) { return MEIKEY_W; }

#endif
