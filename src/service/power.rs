//! 探している間、OS に CPU を絞らせない
//!
//! Windows 11 は、前面にないウィンドウのプロセスを数秒で「省電力（EcoQoS）」にし、
//! 効率コアに寄せたりクロックを下げたりする。探索は CPU を回し続けるので、これで数分の一まで遅くなる
//! （GPU を操作するスレッドも遅れ、GPU の速さも落ちる）。それをやめるよう、プロセスに印を付ける

/// このプロセスを省電力の対象から外す（Windows 以外では何もしない）。失敗しても探索は続ける
pub fn keep_full_speed() {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_STATE,
            ProcessPowerThrottling, SetProcessInformation,
        };
        // ControlMask で「実行速度」を自分で決めると伝え、StateMask を 0（絞らない）にする
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask: 0,
        };
        // SAFETY: GetCurrentProcess は閉じなくてよい疑似ハンドルを返す。state は呼び出しの間生きていて、
        // 大きさは渡した型そのもの
        unsafe {
            SetProcessInformation(
                GetCurrentProcess(),
                ProcessPowerThrottling,
                (&raw const state).cast(),
                size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    /// 何度呼んでも落ちない
    #[test]
    fn can_be_called_repeatedly() {
        super::keep_full_speed();
        super::keep_full_speed();
    }
}
