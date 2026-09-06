# W1 当前布局的静态计算

由 `w1_handcheck.py` 生成；输入、源码及hash在 `evidence/`。每项取最后一轮 g00 main、layer 0、head 0 八通道。不是 Ramulator 延迟，不把行/列数当作 TBT。

| 输入 | 请求数 | main context | worker context | main有效diff token | 完整读集最忙通道QK列请求 A3b,A4c,A4e |
|---|---|---|---|---|---|
| W1_S1_rounds_12_turns | 73 | 4176 | 4688 | 160 | 176,178,96 |
| W1_S1_rounds_48_turns | 289 | 18576 | 19088 | 736 | 752,770,418 |
| W1_S2_workers_1_turns | 97 | 6160 | 9488 | 176 | 370,386,144 |
| W1_S2_workers_4_turns | 241 | 14608 | 9488 | 704 | 544,562,448 |
| W1_S3_sessions_1_turns | 73 | 8976 | 9488 | 352 | 278,306,224 |
| W1_S3_sessions_4_turns | 289 | 8976 | 9488 | 352 | 370,386,240 |
| W1_S4_worker_lout_256_turns | 145 | 14608 | 12432 | 352 | 544,552,384 |
| W1_S4_worker_lout_32_turns | 145 | 4752 | 7280 | 352 | 236,248,176 |
| W1_S5_main_lout_32_turns | 145 | 6768 | 9488 | 352 | 224,238,192 |
| W1_S5_main_lout_512_turns | 145 | 17808 | 9488 | 352 | 370,386,356 |
| W1_S6_doc_tokens_1024_turns | 145 | 8976 | 27920 | 352 | 368,380,192 |
| W1_S6_doc_tokens_512_turns | 145 | 8976 | 15632 | 352 | 368,380,192 |
| W1_turns | 145 | 8976 | 9488 | 352 | 368,380,192 |

详细计数：K物理占行与QK命令分开；full包含master、shadow和diff。

| 输入 | 档 | context | diff token | diff K行总和 | diff最忙K行 | diff QK命令总和 | full最忙QK命令 | full最忙K行 | full最忙token |
|---|---|---|---|---|---|---|---|---|---|
| W1_S1_rounds_12_turns | A3b | 4176 | 160 | 10 | 3 | 20 | 176 | 11 | 1408 |
| W1_S1_rounds_12_turns | A4c | 4176 | 160 | 3 | 1 | 40 | 178 | 12 | 1408 |
| W1_S1_rounds_12_turns | A4e | 4176 | 160 | 1 | 1 | 40 | 96 | 6 | 768 |
| W1_S1_rounds_48_turns | A3b | 18576 | 736 | 46 | 12 | 92 | 752 | 47 | 6016 |
| W1_S1_rounds_48_turns | A4c | 18576 | 736 | 12 | 2 | 184 | 770 | 49 | 6080 |
| W1_S1_rounds_48_turns | A4e | 18576 | 736 | 3 | 2 | 184 | 418 | 25 | 3088 |
| W1_S2_workers_1_turns | A3b | 6160 | 176 | 22 | 11 | 44 | 370 | 35 | 2960 |
| W1_S2_workers_1_turns | A4c | 6160 | 176 | 3 | 1 | 44 | 386 | 25 | 3024 |
| W1_S2_workers_1_turns | A4e | 6160 | 176 | 1 | 1 | 44 | 144 | 9 | 1152 |
| W1_S2_workers_4_turns | A3b | 14608 | 704 | 22 | 6 | 88 | 544 | 34 | 4352 |
| W1_S2_workers_4_turns | A4c | 14608 | 704 | 12 | 2 | 176 | 562 | 36 | 4416 |
| W1_S2_workers_4_turns | A4e | 14608 | 704 | 3 | 3 | 176 | 448 | 20 | 2880 |
| W1_S3_sessions_1_turns | A3b | 8976 | 352 | 22 | 3 | 44 | 278 | 20 | 2224 |
| W1_S3_sessions_1_turns | A4c | 8976 | 352 | 3 | 1 | 88 | 306 | 19 | 2320 |
| W1_S3_sessions_1_turns | A4e | 8976 | 352 | 2 | 2 | 88 | 224 | 14 | 1792 |
| W1_S3_sessions_4_turns | A3b | 8976 | 352 | 22 | 11 | 44 | 370 | 35 | 2960 |
| W1_S3_sessions_4_turns | A4c | 8976 | 352 | 12 | 2 | 88 | 386 | 26 | 3024 |
| W1_S3_sessions_4_turns | A4e | 8976 | 352 | 2 | 1 | 88 | 240 | 12 | 1664 |
| W1_S4_worker_lout_256_turns | A3b | 14608 | 352 | 22 | 6 | 44 | 544 | 23 | 4352 |
| W1_S4_worker_lout_256_turns | A4c | 14608 | 352 | 6 | 1 | 88 | 552 | 24 | 4384 |
| W1_S4_worker_lout_256_turns | A4e | 14608 | 352 | 2 | 1 | 88 | 384 | 12 | 3072 |
| W1_S4_worker_lout_32_turns | A3b | 4752 | 352 | 22 | 6 | 44 | 236 | 23 | 1888 |
| W1_S4_worker_lout_32_turns | A4c | 4752 | 352 | 6 | 1 | 88 | 248 | 24 | 1936 |
| W1_S4_worker_lout_32_turns | A4e | 4752 | 352 | 2 | 1 | 88 | 176 | 12 | 1408 |
| W1_S5_main_lout_32_turns | A3b | 6768 | 352 | 22 | 6 | 44 | 224 | 23 | 1792 |
| W1_S5_main_lout_32_turns | A4c | 6768 | 352 | 6 | 1 | 88 | 238 | 24 | 1840 |
| W1_S5_main_lout_32_turns | A4e | 6768 | 352 | 2 | 1 | 88 | 192 | 12 | 1536 |
| W1_S5_main_lout_512_turns | A3b | 17808 | 352 | 22 | 11 | 44 | 370 | 23 | 2960 |
| W1_S5_main_lout_512_turns | A4c | 17808 | 352 | 6 | 1 | 88 | 386 | 19 | 3024 |
| W1_S5_main_lout_512_turns | A4e | 17808 | 352 | 2 | 1 | 88 | 356 | 18 | 2848 |
| W1_S6_doc_tokens_1024_turns | A3b | 8976 | 352 | 22 | 6 | 44 | 368 | 23 | 2944 |
| W1_S6_doc_tokens_1024_turns | A4c | 8976 | 352 | 6 | 1 | 88 | 380 | 24 | 2992 |
| W1_S6_doc_tokens_1024_turns | A4e | 8976 | 352 | 2 | 1 | 88 | 192 | 12 | 1536 |
| W1_S6_doc_tokens_512_turns | A3b | 8976 | 352 | 22 | 6 | 44 | 368 | 23 | 2944 |
| W1_S6_doc_tokens_512_turns | A4c | 8976 | 352 | 6 | 1 | 88 | 380 | 24 | 2992 |
| W1_S6_doc_tokens_512_turns | A4e | 8976 | 352 | 2 | 2 | 88 | 192 | 13 | 1536 |
| W1_turns | A3b | 8976 | 352 | 22 | 6 | 44 | 368 | 23 | 2944 |
| W1_turns | A4c | 8976 | 352 | 6 | 1 | 88 | 380 | 24 | 2992 |
| W1_turns | A4e | 8976 | 352 | 2 | 1 | 88 | 192 | 12 | 1536 |

## Prefill 收支平衡预算

LLAMA3-8B、A100a、FlashAttention、NVLink、GQA4，单层；capacity=2；PIM每sweep时间必须来自Ramulator。最后一列仅为 `(GPU价格−context返回)/sweeps`。

| 请求 | m | N | sweeps | GPU µs | context返回 µs | PIM允许平均sweep µs |
|---|---|---|---|---|---|---|
| a0_corpus | 12544 | 12544 | 6272 | 13010.835 | 348.595 | 2.019 |
| g00_m_t000 | 32 | 32 | 16 | 1.512 | 0.874 | 0.040 |
| g00_m_t002 | 32 | 576 | 16 | 25.287 | 0.874 | 1.526 |
| g00_m_t023 | 32 | 8976 | 16 | 302.999 | 0.874 | 18.883 |
| g00_w0_t023 | 24 | 9488 | 12 | 319.880 | 0.655 | 26.602 |

## Decode MQ

```json
{
  "last_tier_batch_size": 6,
  "all_member_common_master_tokens": 0,
  "same_worker_cross_session_common_tokens": 6144,
  "main_cross_session_common_tokens": 0
}
```
