//! 件数・時間を人が読みやすい形にする

use std::time::Duration;

/// 件数を「万」「億」で表す
pub fn count(x: f64) -> String {
    match x {
        x if x >= 1e8 => format!("{:.1}億", x / 1e8),
        x if x >= 1e4 => format!("{:.1}万", x / 1e4),
        x => format!("{x:.0}"),
    }
}

/// 時間を「3日18時間」「1時間1分」「0分59秒」のように大きい単位2つで表す
pub fn duration(d: Duration) -> String {
    let s = d.as_secs();
    let (days, hours, mins, secs) = (s / 86_400, s / 3_600 % 24, s / 60 % 60, s % 60);
    match (days, hours) {
        (0, 0) => format!("{mins}分{secs}秒"),
        (0, h) => format!("{h}時間{mins}分"),
        (d, h) => format!("{d}日{h}時間"),
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
    #[case::zero(0.0, "0")]
    #[case::below_10k(9_999.0, "9999")]
    #[case::man(27_000.0, "2.7万")]
    #[case::oku(3.0e10, "300.0億")]
    fn count_works(#[case] x: f64, #[case] want: &str) {
        assert_eq!(count(x), want);
    }

    /// 表示を数に戻して `unit` 倍すると、元の値との差は表示の最小の刻み `step` の半分以内
    #[rstest]
    #[case::plain(0.0..1e4, "", 1.0, 1.0)]
    #[case::man(1e4..1e8, "万", 1e4, 1e3)]
    #[case::oku(1e8..1e15, "億", 1e8, 1e7)]
    fn count_rounds_within_its_unit(
        #[case] range: Range<f64>,
        #[case] suffix: &str,
        #[case] unit: f64,
        #[case] step: f64,
    ) {
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
    #[case::zero(0.0, "0分0秒")]
    #[case::seconds(59.0, "0分59秒")]
    #[case::truncated(59.9, "0分59秒")]
    #[case::minutes(125.0, "2分5秒")]
    #[case::hours(3_661.0, "1時間1分")]
    #[case::hours_and_minutes(3_600.0 * 2.0 + 60.0 * 34.0 + 5.0, "2時間34分")]
    #[case::just_below_a_day(86_399.0, "23時間59分")]
    #[case::days(86_400.0 * 3.0 + 3_600.0 * 18.0, "3日18時間")]
    #[case::negative(-1.0, "0分0秒")]
    #[case::beyond_duration_max(Duration::MAX.as_secs_f64(), "∞")] // 2^64 秒は Duration に入らない
    #[case::infinite(f64::INFINITY, "∞")]
    #[case::nan(f64::NAN, "∞")]
    fn duration_display(#[case] secs: f64, #[case] want: &str) {
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
        fn parse_back(self, s: &str) -> Option<u64> {
            let (big, small, big_unit, small_unit) = match self {
                Shape::MinutesSeconds => ("分", "秒", 60, 1),
                Shape::HoursMinutes => ("時間", "分", 3_600, 60),
                Shape::DaysHours => ("日", "時間", 86_400, 3_600),
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
    fn duration_is_truncated_to_two_units(#[case] range: Range<u64>, #[case] shape: Shape) {
        check(2048, range, |secs| {
            let s = duration(Duration::from_secs(secs));
            let back = shape.parse_back(&s).ok_or_else(|| {
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
