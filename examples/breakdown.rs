//! 1鍵あたりの計算時間を、高速化の組み合わせごとに計測する: cargo make breakdown
use meikey::derive::{BATCH, Backend, Deriver, PublicKeys, Seeds};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let rounds = 400;
    let mut seeds: Seeds = [[7u8; 32]; BATCH];
    let mut out: PublicKeys = [[0u8; 32]; BATCH];

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

    // 高速化前（両方なし）から、1つずつ足して、両方ありまで
    for backend in Backend::ALL.into_iter().rev() {
        let mut d = Deriver::new(backend);
        let s = Instant::now();
        for r in 0..rounds {
            seeds
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| x[..8].copy_from_slice(&((r * BATCH + i) as u64).to_le_bytes()));
            d.public_keys(black_box(&seeds), &mut out);
            black_box(&out);
        }
        let us = s.elapsed().as_secs_f64() * 1e6 / (rounds * BATCH) as f64;
        println!("{us:>6.2} µs/鍵  ({:.2}倍)  {backend}", base / us);
    }
}
