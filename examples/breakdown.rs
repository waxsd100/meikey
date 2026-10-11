//! 1鍵あたりの計算時間を、高速化の組み合わせごとに計測する: cargo make breakdown
use meikey::derive::{BATCH, Backend, Deriver, PublicKeys, Seeds};
use meikey::key::{PUBLIC_B64_LEN, encode_public_b64};
use meikey::{Find, Pattern};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let rounds = 400;
    let mut seeds: Seeds = [[7u8; 32]; BATCH];
    let mut out: PublicKeys = [[0u8; 32]; BATCH];

    println!("--- 公開鍵の計算");
    let s = Instant::now();
    for i in 0..(rounds * BATCH) as u64 {
        let mut seed = [7u8; 32];
        seed[..8].copy_from_slice(&i.to_le_bytes());
        black_box(
            ed25519_dalek::SigningKey::from_bytes(&seed)
                .verifying_key()
                .to_bytes(),
        );
    }
    let base = s.elapsed().as_secs_f64() * 1e6 / (rounds * BATCH) as f64;
    // 全角文字は {:<N} で揃わないので、名前は行末に置く
    println!("{base:>6.2} µs/鍵  (基準)    ed25519-dalek の SigningKey で1鍵ずつ");

    // 高速化前（全部なし）から、1つずつ足して、全部ありまで。CPU が対応していない SIMD は
    // 下のものと同じ計算になるので飛ばす
    let mut publics = Vec::with_capacity(rounds * BATCH);
    for backend in Backend::ALL
        .into_iter()
        .rev()
        .filter(|b| b.simd.supported())
    {
        let mut d = Deriver::new(backend);
        publics.clear();
        let s = Instant::now();
        for r in 0..rounds {
            seeds
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| x[..8].copy_from_slice(&((r * BATCH + i) as u64).to_le_bytes()));
            d.public_keys(black_box(&seeds), &mut out);
            publics.extend_from_slice(black_box(&out));
        }
        let us = s.elapsed().as_secs_f64() * 1e6 / (rounds * BATCH) as f64;
        println!("{us:>6.2} µs/鍵  ({:.2}倍)  {backend}", base / us);
    }

    // 上で作った公開鍵を base64 にして、文字列を探す（探索ループの残りの部分）。
    // 公開鍵の計算に比べて小さいので、ns で表示する
    for (needle, case_sensitive) in [("wakokara", false), ("wakokara", true)] {
        let pattern = Pattern::new(needle, case_sensitive).unwrap();
        let label = if case_sensitive {
            "区別あり"
        } else {
            "区別なし"
        };
        println!("--- base64 化 + 「{needle}」を探す（大小文字{label}）");
        for find in [Find::Naive, Find::Memmem] {
            let matcher = pattern.matcher(find);
            let mut b64 = [0u8; PUBLIC_B64_LEN];
            let s = Instant::now();
            let mut hits = 0usize;
            for public in &publics {
                encode_public_b64(black_box(public), &mut b64);
                hits += usize::from(matcher.is_match(black_box(&b64)));
            }
            black_box(hits);
            let ns = s.elapsed().as_secs_f64() * 1e9 / publics.len() as f64;
            println!("{ns:>6.1} ns/鍵  {find}");
        }
    }
}
