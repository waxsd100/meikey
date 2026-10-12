// GPU カーネル（kernel.cu）を CPU 向けの C++ としてコンパイルする（テストで Rust の実装と突き合わせるため）。
// build.rs が feature cuda のときだけコンパイルする
#include "kernel.cu"
