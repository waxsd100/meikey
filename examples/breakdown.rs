//! 1鍵あたりの計算時間を計測する: cargo make breakdown
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
    println!(
        "{:<34} {base:>6.2} µs/鍵",
        "ed25519-dalek（1鍵ずつ・改善前）"
    );

    for backend in [Backend::ConstantTime, Backend::Fast] {
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
        println!(
            "{:<34} {us:>6.2} µs/鍵  ({:.2}倍)",
            format!("{backend:?}（{BATCH}鍵まとめて）"),
            base / us
        );
    }
}
