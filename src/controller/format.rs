//! 件数・時間を人が読みやすい形にする（日本語は「万」「億」「3日18時間」、英語は「K」「M」「B」「3d 18h」）

use crate::lang::{Lang, current, tf};
use std::time::Duration;

/// 件数を、日本語なら「万」「億」、英語なら「K」「M」「B」で表す
pub fn count(x: f64) -> String {
    match current() {
        Lang::Ja => match x {
            x if x >= 1e8 => format!("{:.1}億", x / 1e8),
            x if x >= 1e4 => format!("{:.1}万", x / 1e4),
            x => format!("{x:.0}"),
        },
        Lang::En => match x {
            x if x >= 1e9 => format!("{:.1}B", x / 1e9),
            x if x >= 1e6 => format!("{:.1}M", x / 1e6),
            x if x >= 1e3 => format!("{:.1}K", x / 1e3),
            x => format!("{x:.0}"),
        },
    }
}

/// 時間を「3日18時間」「1時間1分」「0分59秒」（英語は「3d 18h」「1h 1m」「0m 59s」）のように
/// 大きい単位2つで表す
pub fn duration(d: Duration) -> String {
    let s = d.as_secs();
    let (days, hours, mins, secs) = (s / 86_400, s / 3_600 % 24, s / 60 % 60, s % 60);
    match (days, hours) {
        (0, 0) => tf!("{mins}分{secs}秒", "{mins}m {secs}s"),
        (0, h) => tf!("{h}時間{mins}分", "{h}h {mins}m"),
        (d, h) => tf!("{d}日{h}時間", "{d}d {h}h"),
    }
}

/// 状態表示の1行（`ラベル  値`）。ラベルを表示幅 14 桁にそろえる（全角は2桁と数える）
pub fn row(label: &str, value: &str) -> String {
    let width: usize = label
        .chars()
        .map(|c| if c.is_ascii() { 1 } else { 2 })
        .sum();
    format!(
        "{label}{}{value}",
        " ".repeat(14usize.saturating_sub(width).max(1))
    )
}

/// 秒数（f64）版。期待値など無限大になりうる値用
pub fn duration_secs(secs: f64) -> String {
    if secs.is_finite() && secs < Duration::MAX.as_secs_f64() {
        duration(Duration::from_secs_f64(secs.max(0.0)))
    } else {
        "∞".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::{Lang, set};
    use crate::property::check;
    use proptest::prelude::*;
    use rstest::rstest;
    use similar_asserts::assert_eq;
    use std::ops::Range;

    /// ラベルは表示幅 14 桁にそろう（全角は2桁）。長すぎても値との間に空白を1つ入れる
    #[rstest]
    #[case::ascii("CPU", "CPU           x")]
    #[case::wide("パスフレーズ", "パスフレーズ  x")]
    #[case::too_long("とても長いラベルの例", "とても長いラベルの例 x")]
    fn row_aligns_labels(#[case] label: &str, #[case] want: &str) {
        assert_eq!(row(label, "x"), want);
    }

    #[rstest]
    #[case::zero(Lang::Ja, 0.0, "0")]
    #[case::below_10k(Lang::Ja, 9_999.0, "9999")]
    #[case::man(Lang::Ja, 27_000.0, "2.7万")]
    #[case::oku(Lang::Ja, 3.0e10, "300.0億")]
    #[case::en_zero(Lang::En, 0.0, "0")]
    #[case::en_below_1k(Lang::En, 999.0, "999")]
    #[case::en_k(Lang::En, 27_000.0, "27.0K")]
    #[case::en_m(Lang::En, 62_501_000.0, "62.5M")]
    #[case::en_b(Lang::En, 7.0e8, "700.0M")]
    #[case::en_b_large(Lang::En, 3.0e10, "30.0B")]
    fn count_works(#[case] lang: Lang, #[case] x: f64, #[case] want: &str) {
        set(lang);
        assert_eq!(count(x), want);
    }

    /// 表示を数に戻して `unit` 倍すると、元の値との差は表示の最小の刻み `step` の半分以内
    #[rstest]
    #[case::plain(Lang::Ja, 0.0..1e4, "", 1.0, 1.0)]
    #[case::man(Lang::Ja, 1e4..1e8, "万", 1e4, 1e3)]
    #[case::oku(Lang::Ja, 1e8..1e15, "億", 1e8, 1e7)]
    #[case::en_plain(Lang::En, 0.0..1e3, "", 1.0, 1.0)]
    #[case::en_k(Lang::En, 1e3..1e6, "K", 1e3, 1e2)]
    #[case::en_m(Lang::En, 1e6..1e9, "M", 1e6, 1e5)]
    #[case::en_b(Lang::En, 1e9..1e15, "B", 1e9, 1e8)]
    fn count_rounds_within_its_unit(
        #[case] lang: Lang,
        #[case] range: Range<f64>,
        #[case] suffix: &str,
        #[case] unit: f64,
        #[case] step: f64,
    ) {
        set(lang);
        check(1024, range, |x| {
            let s = count(x);
            let n: f64 = s
                .strip_suffix(suffix)
                .and_then(|n| n.parse().ok())
                .ok_or_else(|| TestCaseError::fail(format!("{x} → {s}")))?;
            prop_assert!(
                (n * unit - x).abs() <= step / 2.0 + x * 1e-12,
                "{} → {}",
                x,
                s
            );
            Ok(())
        });
    }

    /// 秒数の表示（f64 版。Duration で表せる値なら Duration 版も同じ表示）
    #[rstest]
    #[case::zero(0.0, "0分0秒", "0m 0s")]
    #[case::seconds(59.0, "0分59秒", "0m 59s")]
    #[case::truncated(59.9, "0分59秒", "0m 59s")]
    #[case::minutes(125.0, "2分5秒", "2m 5s")]
    #[case::hours(3_661.0, "1時間1分", "1h 1m")]
    #[case::hours_and_minutes(3_600.0 * 2.0 + 60.0 * 34.0 + 5.0, "2時間34分", "2h 34m")]
    #[case::just_below_a_day(86_399.0, "23時間59分", "23h 59m")]
    #[case::days(86_400.0 * 3.0 + 3_600.0 * 18.0, "3日18時間", "3d 18h")]
    #[case::negative(-1.0, "0分0秒", "0m 0s")]
    #[case::beyond_duration_max(Duration::MAX.as_secs_f64(), "∞", "∞")] // 2^64 秒は Duration に入らない
    #[case::infinite(f64::INFINITY, "∞", "∞")]
    #[case::nan(f64::NAN, "∞", "∞")]
    fn duration_display(
        #[case] secs: f64,
        #[case] ja: &str,
        #[case] en: &str,
        #[values(Lang::Ja, Lang::En)] lang: Lang,
    ) {
        set(lang);
        let want = if lang == Lang::Ja { ja } else { en };
        assert_eq!(duration_secs(secs), want);
        if let Ok(d) = Duration::try_from_secs_f64(secs) {
            assert_eq!(duration(d), want);
        }
    }

    /// 表示の形と、そのときの最小の単位（秒）
    #[derive(Debug, Clone, Copy)]
    enum Shape {
        /// 「0分59秒」
        MinutesSeconds,
        /// 「1時間1分」
        HoursMinutes,
        /// 「3日18時間」
        DaysHours,
    }

    impl Shape {
        /// 表示を秒に戻す（表示の最小の単位より細かい部分は切り捨てられている）。形が違えば None
        fn parse_back(self, lang: Lang, s: &str) -> Option<u64> {
            let (big, small, big_unit, small_unit) = match (lang, self) {
                (Lang::Ja, Shape::MinutesSeconds) => ("分", "秒", 60, 1),
                (Lang::Ja, Shape::HoursMinutes) => ("時間", "分", 3_600, 60),
                (Lang::Ja, Shape::DaysHours) => ("日", "時間", 86_400, 3_600),
                (Lang::En, Shape::MinutesSeconds) => ("m ", "s", 60, 1),
                (Lang::En, Shape::HoursMinutes) => ("h ", "m", 3_600, 60),
                (Lang::En, Shape::DaysHours) => ("d ", "h", 86_400, 3_600),
            };
            let (a, rest) = s.split_once(big)?;
            let b = rest.strip_suffix(small)?;
            Some(a.parse::<u64>().ok()? * big_unit + b.parse::<u64>().ok()? * small_unit)
        }

        fn unit(self) -> u64 {
            match self {
                Shape::MinutesSeconds => 1,
                Shape::HoursMinutes => 60,
                Shape::DaysHours => 3_600,
            }
        }
    }

    /// 大きい単位2つの形で表示され、表示を秒に戻すと元の値を最小の単位で切り捨てたものになる
    #[rstest]
    #[case::below_an_hour(0..3_600, Shape::MinutesSeconds)]
    #[case::below_a_day(3_600..86_400, Shape::HoursMinutes)]
    #[case::days(86_400..10_000_000_000, Shape::DaysHours)]
    fn duration_is_truncated_to_two_units(
        #[case] range: Range<u64>,
        #[case] shape: Shape,
        #[values(Lang::Ja, Lang::En)] lang: Lang,
    ) {
        set(lang);
        check(2048, range, |secs| {
            let s = duration(Duration::from_secs(secs));
            let back = shape.parse_back(lang, &s).ok_or_else(|| {
                TestCaseError::fail(format!("{secs} 秒 → {s}（{shape:?} の形でない）"))
            })?;
            prop_assert_eq!(back, secs - secs % shape.unit(), "{} 秒 → {}", secs, s);
            Ok(())
        });
    }

    #[rstest]
    fn duration_secs_never_panics() {
        check(4096, any::<f64>(), |secs| {
            prop_assert!(!duration_secs(secs).is_empty());
            Ok(())
        });
    }
}
