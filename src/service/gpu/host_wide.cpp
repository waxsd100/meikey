// host.cpp と同じく kernel.cu を CPU 向けにコンパイルする。こちらは GPU で選べる上限側の設定
// （表の幅 W = 16、逆元をまとめる鍵の数 32）で、テスト用の関数は meikey_wide_test_* という名前にする
#define MEIKEY_W 16
#define MEIKEY_BATCH 32
#define TEST_FN(name) meikey_wide_test_##name
#include "kernel.cu"
