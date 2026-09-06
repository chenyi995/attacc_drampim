Existing small-run extraction; computed from the JSON files listed in the accompanying evidence.

| Rung | E2E ms | weighted TBT us | PIM lane sum ms | PIM energy J | full-run avg lane busy % | decode-window avg lane busy % |
|---|---:|---:|---:|---:|---:|---:|
| A3b | 96.991918 | 235.622266 | 403.582840 | 0.543205950 | 26.0062 | 41.8208 |
| A4c | 96.735997 | 234.625769 | 393.610891 | 0.547638368 | 25.4307 | 40.9612 |
| A4e | 96.490291 | 233.659977 | 393.500253 | 0.547622925 | 25.4883 | 41.1179 |

| Transition | E2E reduction % | weighted TBT reduction % | lane sum reduction % | PIM energy reduction % |
|---|---:|---:|---:|---:|
| A3b -> A4c | 0.263857 | 0.422922 | 2.470856 | -0.815974 |
| A4c -> A4e | 0.253997 | 0.411631 | 0.028108 | 0.002820 |
| A3b -> A4e | 0.517184 | 0.832811 | 2.498269 | -0.813131 |

A3b GPU work = 70.770256 ms; GPU busy / E2E = 72.9651%; loose zero-PIM E2E speedup ceiling = 1.370518x.

Around 42% is reproduced only after dividing lane service by 16 channels and the decode observation window. Full-run denominator yields around 26%. No events are retained, so individual lane busy intervals/critical-path fraction cannot be recovered.

Run reports contain no commit/GPU-model/config fingerprint. Log launch 17:53:17 follows da220d1 commit 17:51:35; same checkout is an inference, not a pinned provenance guarantee. Session16 documents earlier bb19f31 timing before 9c40891 and is a different run. Model CSV and platform log identify CACHEBLEND-TINY/A100a x1; flash and one-HBM setup follow documented small-run context, not an embedded run manifest.

Follow-up provenance/resource checks:
Workload: /data2/chenyi9/KV-PIM/scratch_0905/small/wl_small_A4R4.json; SHA256 c1a09ee22da4de0d5aaf4c41f72c4f0df2eb4484fb4b3bdfde47815fe2c742c9
Four doc/doc/user groups are concatenated inside each of four worker prompts; one prefill and one 256-token decode per worker, plus the cache owner. There are no inter-round parent edges, no parent_out segments and no tier boundaries. This is not B1/T9 turns performance.
TBT: 5 request summaries, 255 post-first-token intervals per request = 1275 observations; 1024 saved decode batches = 256 output positions x 4 layers.
Recompute k=8 in report; 30 reused segments have 8 correction rows and 2 have 0, total 240 correction rows per layer.
With these SAME GPU event durations on this SAME single serial GPU resource, makespan >= gpu_time_s_unoverlapped. Current A3b/A4c/A4e report the exact same sum. Not a lower bound if a hypothetical future layout change also changes GPU work or its batching.
Flash is the documented/intended setup and script default, not independently self-certified by these JSON/log/CSV artifacts. Do not turn this inference into a pinned configuration guarantee.
All observed energy deltas are in decode_pim_kv_scan_score_softmax_pv; the saved breakdown does not retain command-level ACT/MAC/MV/SFM terms, so the physical cause cannot be identified from these results alone.
