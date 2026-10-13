//! テストで、CPU が SIMD の命令セットに対応しているかを確かめる
//!
//! 対応していない命令セットのテストは飛ばす（そのまま成功になる）。ただし環境変数 `MEIKEY_REQUIRE_SIMD`
//! に名前（`avx2`・`avx512`・`avx512-ifma`・`amx`・GPU の `nvrtc`・`cuda` をカンマ区切り）があれば、飛ばさずに失敗させる。
//! 対応した CI や Intel SDE（命令のエミュレーター）で回すときに、黙って飛ばされないようにするため。

use similar_asserts::assert_eq;
use std::env::var;

/// `token` が None（CPU が対応していない）ならテストを飛ばす旨を表示して None を返す。
/// `MEIKEY_REQUIRE_SIMD` に `name` があれば、そこで失敗させる
pub fn available<T>(name: &str, token: Option<T>) -> Option<T> {
    if token.is_none() {
        let required = var("MEIKEY_REQUIRE_SIMD").unwrap_or_default();
        assert_eq!(
            required.split(',').any(|r| r.trim() == name),
            false,
            "MEIKEY_REQUIRE_SIMD に {name} があるのに、この CPU は {name} に対応していません"
        );
        eprintln!("{name} が使えない環境なので飛ばします");
    }
    token
}
