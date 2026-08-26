# chenyi-822 migration plan: reviewed re-derivation of chenyi-822-dirty

(Tracked in the chenyi-822 repository by Chenyi's instruction,
2026-08-25 -- superseding the earlier never-commit rule for this file.)

Working copies (2026-08-25 layout):
- THIS folder `/data2/chenyi9/KV-PIM/attacc_drampim` = the CLEAN branch
  `chenyi-822` (currently 70a914c = phase-4 state of the old plan);
- `/data2/chenyi9/KV-PIM/attacc_drampim_822` = the reference branch
  `chenyi-822-dirty`, HEAD **3b330c8** (fast-integration lane, fully
  verified: 41/41 unit tests, 145/145 matrix, claims checked);
- `/data2/chenyi9/KV-PIM/attacc_drampim_xinyao` = `chenyi-experiment-821`
  (C series; not a migration source here).

Reference: every step's "target text" is taken from dirty HEAD 3b330c8
(not intermediate commits).  Intermediate dirty states that were later
superseded (split-mode 5.L2 insertion form, (n_q,n_c) wording, selection
sweep) are NOT ported -- each phase lands directly in its final form.

## Working rules (unchanged, fixed by Chenyi 2026-08-22/23)

1. Per step: file, function, exact original text -> new text with line
   numbers, full code (no diff-only), why, verification command.
2. Edit only on explicit per-step approval ("你来改/做/改/批准").
3. LOGIC fine-grained (a loaded line may be its own step); PLUMBING
   (signatures/forwarding/CLI/report fields) batched per phase; DOCS
   batched; RESULTS copied as one data step.
4. Suite green at every phase end + one English commit per phase;
   suite count trajectory: 31 -> 32 -> 38 -> 38 -> 39 -> 39 -> 40 -> 41.
5. Seed copies `pim_ramulator_src/` byte-identical whenever
   `ramulator2/` changes.
6. Close-out = comment-stripped parity diff of every ported file vs
   dirty HEAD 3b330c8 (expected zero); push chenyi-822 on instruction.

## Phase table (M0-M12)

| Phase | Content (dirty source commit) | Steps | Suite |
|---|---|---|---|
| M0 | Environment: rebuild ramulator2 here (gcc-toolset-11), run baseline | 1 | 31/31 |
| M1 | D_i bitmap events (2e2cbf6; reviewed once already as 5.L1) | 1L | 31/31 + event/makespan closure vs dirty |
| M2 | Bank-whole prefill branch, final "pim"-mode form: landing-order stores + full-range scan + sweep loop; DIE causal-drop assembly; validator contract (654aeee/0755694 shape, sourced from dirty HEAD) | 3L | 31/31 |
| M3 | Phase-5 plumbing: `_run_cacheblend_prefill` mode param + validation + report field (menu values gpu/pim/dynamic enter HERE in final form, dynamic body stubbed to "pim" until M7) + port `test_bank_whole...` | 1P+1T | 32/32 |
| M4 | Agentic history, both paths (11L)、plumbing(--history-len 等)、6 tests | 11L+1P+1T | 38/38 |
| M5 | Streaming-P wording sync: wrapper docstrings, trace-gen phase comments (+ seed sync) — comment-only | 1 | 38/38 |
| M6 | A-ladder rebuild (654aeee): PRESETS/labels/modes (split 废除), batching-coupled presets, ablation dynamic block (final estimate=charge form from b649674), split arm removal + overlap rename, module docstring; CLI (--pim-batch-command follow-ladder etc.); ladder tests | 5L+1P+1T | 39/39 |
| M7 | Physical DAG alignment (0755694): side resolution + dynamic estimator; gpu branch; split path removal (incl. LSE tuple events); validator kv_pim_to_gpu rule; `pim_prefill_sides` report; main.py menu flag; placement-menu & history tests rewritten | 5L+1P+1T | 39/39 |
| M8 | Policy families (31b5689): CACHEBLEND/EPIC family tuples + gates; promptcache/cachecraft/cachetune (cachecraft overlap rule, cachetune no-selection-layer guard); ReuseConfig knob; CLI; family test.  Ruling note: policies are recompute-count knobs, NOT an experiment axis | 4L+1P+1T | 40/40 |
| M9 | Audited fixes (b649674 + 0305d4c): (a) 15/1 pools both paths; (b) naive tracked-channel pools + same-channel serialization; (c) balance-point presets (2.6 GHz/768 B, PROVISIONAL) + query-batch follows capacity; (d) memory owner-copy fix + unit test (+ native marker); each with its PAPER-TIE comments | 4L+1T | 41/41 |
| M10 | Legacy-path parallel pre-warm (dfde565): df lock + 32-thread warm block; bit-identity check on relay/7B/A1 | 2L | 41/41 |
| M11 | Docs set (batched): README + A1-A6 + software_upstream + workloads + experiments + external/ + delta-vs-xinyao + audit ladder issues + consolidated findings ledger + turnaround entry (f4cc6dc…3b330c8 doc states, final wording) | 3 doc steps | 41/41 |
| M12 | paper_ladder infrastructure: MATRIX/CLAIMS_CHECK/TABLES + drivers (run_matrix, collect, repair) + workloads snapshot (incl. mooncakemt); .gitignore logs | 2 | 41/41 |
| **M12b** | **Full matrix RE-RUN on this branch** (2026-08-25 ruling): `run_matrix.py` all 135 jobs (90 ladder + 10 DAG + baseline extras) with the same knobs, 64 cores; then compare `summary.json` cell by cell against the dirty results (expect bit-identical TTFT/TBT/compression/sides).  The re-run results are what gets COMMITTED here -- the clean branch carries numbers it produced itself; dirty results serve only as the comparison oracle | 2 | 41/41 + matrix parity |
| Close | Comment-stripped parity diff vs 3b330c8 (all ported files, expect zero); push on instruction | 1 | — |

## Notes

- M2/M3 ordering intentionally lands the FINAL menu ("pim" branch under
  gpu/pim/dynamic validation) instead of the historical split-mode
  insertion -- the superseded split form is never introduced here.
- M6 dynamic block is ported directly in its FIXED form (estimate ==
  charge, audit issue 1) -- the buggy intermediate never lands.
- The C1/C2 comparison drivers and the selection-variant sweep are NOT
  ported (rulings 2026-08-24/25).
- M12b makes the clean branch self-verifying: identical code (parity
  diff) x identical inputs => identical numbers; any cell mismatch is a
  migration bug found BEFORE the push.  Budget ~4-6 h wall on 64 cores.
- mooncakemt converter lives outside the repo
  (`/data2/chenyi9/KV-PIM/workload/convert_mooncake_multiturn.py`);
  only the emitted workload JSON snapshot is ported (M12).
- Deliberate divergences vs dirty at close-out: this PORTING_PLAN.md
  itself (tracked HERE by the 2026-08-25 instruction, untracked on
  dirty); everything else expects zero difference, docs included.
