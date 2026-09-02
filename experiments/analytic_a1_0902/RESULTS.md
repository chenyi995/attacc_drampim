# A1 analytic acceleration — principle-based decomposition and validation

Two analytic replacements are in play, and they are validated separately:

* **`src/analytic_pim.py`** replaces the **Ramulator** subprocess that prices
  one PIM run.
* **`src/a1_dag_free.py`** replaces the **event DAG** as the source of A1's
  PIM invocations.

Both are decomposed into layers with independent ground truth, and every layer
is checked against the real thing rather than against the model's own
assumptions.  Where a number is fitted, it is fitted on one split and reported
on another.

---

## 1. Ramulator replacement — `src/analytic_pim.py`

### Layer 1 — command counts (no fitted parameter)

Closed-form transcription of `gen_trace_attacc_bank.py`.  Ground truth is
Ramulator's own command counters.

| | result |
|---|---|
| cache entries checked | 12,523 |
| `(mac, sfm, mvgb, mvsb, wrgb)` exact | **12,523 / 12,523** |

### Layer 2 — trace structure (no fitted parameter)

Closed forms for the two quantities that make the cycle model physical rather
than a curve fit:

* **barrier groups** — `pairs·(1 + 2·score_windows + 2·columns_per_bank)` plus
  `1 + score_windows + 1 + columns_per_bank` for a trailing odd head (and for
  the whole stripe path, which has one iteration);
* **row openings** — each MAC phase sweeps `columns_per_bank·score_rows`
  consecutive 32-B columns from the K (score) / V (context) address, and 32
  columns fill a DRAM row, so the count depends on the **start column**.

Ground truth is a regenerated trace, compared command by command.

| regime | barrier groups | row openings |
|---|---|---|
| `chunkstripe1 \| replicate` | **40 / 40** exact | **40 / 40** exact |
| `chunkstripe1 \| mq` | **40 / 40** exact | **40 / 40** exact |
| `legacy \| replicate` | **40 / 40** exact | **40 / 40** exact |

The row-opening term is what fixes a real defect of the previous model: two
runs of identical length differ by a whole ACT/PRE (~96 cycles, measured) when
one straddles a row boundary.  The old model dropped the K/V address from its
key and averaged both regimes into one number.

### Layer 3 — cycles (few physical parameters, cross-validated)

```
cycle = mac·nCCDAB·a + wrgb·b + mvgb·c + mvsb·d + sfm·e + row_openings·f
        + barrier_groups·g + h            (non-negative weighted least squares)
```

Commands are divided by the channels that genuinely run in parallel: the
stripe layout gives each channel its own token slice and command bus, while
the legacy head-per-channel layout re-issues the same all-bank command per
channel from one trace stream.

**The regime key is `(trace_revision, batch_command_scheme)` — three regimes.**
No per-run identity. The previous version keyed on twelve fields including
`channel_base`, which split 444 samples across 92 buckets and turned the fit
into a lookup table.

**There is no refresh term.** An earlier version of this file advertised a
multiplicative refresh stretch `1/(1 − t_refresh/nREFI)` fitted by grid
search. That parameter is *exactly unidentifiable*: the weighted-relative
objective and the non-negativity constraint are both invariant under
`coefficients → stretch·coefficients`, so the prediction does not depend on it
at all. Measured spread of the prediction across the whole grid: **< 1e-8
cycles**. The grid search was a no-op, and the "fit recovers nCCDAB ≈ 0.79–0.99"
claim was quoting a gauge. Refresh is absorbed into the mac coefficient, where
it belongs — it scales with schedule length. `tests/test_analytic_pim.py`
now pins this invariance so the term cannot come back unnoticed.

#### How it is validated, and what was wrong before

**The model's input is the FEATURE VECTOR, not the run signature.** `ceil`
steps collapse hundreds of run lengths onto one feature vector, and
`channel_base` / `shared_kv` / `num_hbm` / `pim_type` never reach the features
at all. Concretely:

| regime | cache rows | distinct run lengths | **distinct model inputs** |
|---|---:|---:|---:|
| `legacy \| replicate` | 11,716 | 411 | **49** |
| `chunkstripe1 \| replicate` | 928 | 59 | **64** |
| `chunkstripe1 \| mq` | 293 | 58 | **66** |

So a split on `run_length` — which is what an earlier version of this document
reported — puts byte-identical samples on both sides. For `legacy|replicate`,
98.6% of its "held-out" rows were duplicates of a training row; the quoted
4.03% was training error. Rows are now deduplicated to one sample per distinct
model input, the split is on that input, and every number is the mean ± std
over 8 seeds (a single split of ~50 inputs has a large seed effect).

| regime | inputs | eff. params | held-out **inputs** | held-out **configs** |
|---|---:|---:|---|---|
| `chunkstripe1 \| replicate` | 63 | 3 | MAPE **3.88% ± 0.70**, p95 7.29%, MAE 913 cyc | 8.20% ± 1.97, p95 16.39% |
| `chunkstripe1 \| mq` | 66 | 6 | **5.51% ± 0.54**, p95 9.32%, MAE 1,057 cyc | 7.00% ± 2.14, p95 10.29% |
| `legacy \| replicate` | 49 | 6 | **6.42% ± 2.37**, p95 18.29%, MAE 13,756 cyc | 8.30% ± 5.22, p95 19.08% |

(8 seeds each except `held_out_configs` for `chunkstripe1|replicate`, which has
only enough distinct configurations for 4.)

The shipped coefficients are then refitted on **all** inputs, so they have no
held-out set of their own — the honest claim is about the *procedure*. The
earlier file quoted one protocol's error next to another protocol's
coefficients, which described neither.

`chunkstripe1|replicate` — the regime A1 actually uses — keeps only **three**
features: `mac_x_interval` (1.051), `mvgb` (6.723), `row_openings` (20.965).
The other five come back exactly zero. Three parameters over 63 distinct
inputs is a much stronger anti-overfitting argument than the "eight
coefficients" an earlier version of this file claimed. The other two regimes
keep six, so "three effective parameters" is a statement about this regime,
not about the model as a whole; `effective_parameters` in the model file is
the count that matters.

#### Two caveats that must travel with these numbers

**The metric is circular, and it is not the consumer's metric.** The fit
minimises weighted *relative* error and the report quotes mean *relative*
error. Worse, the downstream consumer **sums** estimated cycles over a
workload, so what actually matters is the bias of that sum — which relative
error is free to leave uncontrolled, and does (see below). An unweighted fit
of the same features reports a much worse MAPE and a 2.4–6.3× better MAE, so
the 4–6% figure is specific to the objective the fit optimises. MAE in cycles
is reported alongside for that reason.

**The error is not zero-mean at A1's operating point.** Per-input MAPE is
4–6%, but a workload sums thousands of runs, so a *correlated* bias survives
averaging. Run-weighted against the LLAMA3-8B ground truth:

| workload | unweighted mean(pred/truth) | **run-weighted Σn·pred / Σn·truth** |
|---|---:|---:|
| `wl_pipeline_D4` | 1.039 | **1.046** |
| `wl_N4` | 1.049 | **1.054** |

Every one of the top twelve contributors is over-predicted (ratios 1.016–1.052,
none below 1). These operating points are *exactly sampled* in the calibration
set — the nearest sampled `run_length` above and below each is the point
itself — so this is not interpolation across a hole. It is a three-parameter
linear model that cannot be exact everywhere, and the fit is dominated by the
other ~50 inputs. **A1 is the baseline, so a +5% over-price of its PIM time
inflates Fugue's speedup. That direction is anti-conservative for the paper
and must be stated wherever the number is used.**

A measured alternative, not taken: allowing the intercept to go negative (a
ramp-in correction — the first MAC waits for no predecessor) cuts
`legacy|replicate` from 5.96% to **2.52%** cross-validated, but slightly
worsens `chunkstripe1|replicate` (3.47% → 3.57%) and does not remove the
positive bias. Kept out so every coefficient stays a non-negative command
cost.

### What happens outside the calibrated region

`estimate()` **raises** for a regime with no fitted parameters unless the
caller passes a `diagnostics` sink or `allow_uncalibrated=True`. Estimates
outside the calibrated envelope are counted as `extrapolated`. Counters reach
the workload report through `ramulator_signature_cache.analytic_diagnostics`,
next to `analytic_validation`.

The envelope is an axis-aligned box, so a run sitting in an unsampled
*interior* hole passes it silently. `domain.run_length_largest_interior_gap`
records how big such a hole is: **7,680** for `chunkstripe1|replicate`, 8,192
for `chunkstripe1|mq`, 758 for `legacy|replicate`. With no `diagnostics` dict
an extrapolated estimate is returned without comment — only a missing regime
raises.

The datasheet-only fallback measures ~150% MAPE with a 19× worst case, which
is why it is never silent.

---

## 2. DAG replacement — `src/a1_dag_free.py`

Ground truth is the multiset of PIM invocations the **real event DAG** asked
Ramulator to price, recorded with `ATTACC_RECORD_PIM_SIGNATURES` and reduced
to the fields that determine cost (`Ramulator.PRICING_FIELDS`). An op that
stands for several identical capacity sweeps declares that on
`pim_run_multiplicity`, so the comparison is on physical runs rather than
Ramulator invocations.

`validate_dag_free.py` compares three layers: **A1** aggregate PIM work,
**A2** the invocation multiset with multiplicities, **A3** the per-length
profile. A1 and A2 are separate because an enumerator can get the total work
right while distributing it over the wrong runs.

| workload | model | DAG runs | enumerator runs | signature keys | verdict |
|---|---|---:|---:|---|---|
| `workload_llama7b_small` | LLAMA-7B | 2,336 | 2,336 | 5 / 5 | MATCH |
| `workload_llama7b_small` | LLAMA3-8B | 8,480 | 8,480 | 5 / 5 | MATCH |
| `workload_llama7b_medium` (3 tiers) | LLAMA-7B | 34,144 | 34,144 | 16 / 16 | MATCH |
| `workload_llama7b_medium` (3 tiers) | LLAMA3-8B | 98,656 | 98,656 | 16 / 16 | MATCH |
| `workload_tailsweep` (`len % cap == 1`) | LLAMA-7B | 1,824 | 1,824 | 6 / 6 | MATCH |
| `workload_tailsweep` | LLAMA-65B | 5,120 | 5,120 | 8 / 8 | MATCH |
| `workload_llama7b_small` (head folding) | LLAMA-65B | 6,576 | 6,560 | decode exact, prefill −16 | **MISMATCH** (§2.2) |
| `workload_llama7b_small` (head folding) | GPT-175B | 8,752 | 8,736 | decode exact, prefill −16 | **MISMATCH** (§2.2) |
| `wl_pipeline_D4` (4 tiers, history) | LLAMA-7B | 693,568 | 659,712 | 20 of 27 counts match | **MISMATCH** (§2.2) |
| `workload_rag_shared_p24_s8` (24 agents, no history) | LLAMA-7B | 245,504 | 245,504 | 46 / 46 | MATCH |

### 2.1 Read the MATCHes correctly: they are a regression lock, not a proof

**These four MATCHes are agreement by construction.** An adversarial audit of
`a1_dag_free.py` against `workload_runner.py` found that essentially every
field the enumerator emits is a copied constant, a line-by-line transcription
of the DAG's own rule, or — for `cap_rows` — *literally the same imported
function object* (`mq_query_capacity` from `ramulator_wrapper`). A
transcription and its original agreeing is not independent evidence.

That is also the honest answer to "why is it so accurate": it is not a fitted
model that generalised, and it is not overfitting in the statistical sense
(it has zero fitted parameters). It re-states the DAG's implementation. Its
risk is therefore **not** variance — it is that it encodes the DAG's
*implementation* rather than A1's *semantics*, and the two are
indistinguishable while they agree.

The single component the enumerator genuinely re-derives instead of copying is
the **KV address layout** — and that is precisely and solely where it fails.
Worse, the decode half of the comparison cannot test it at all: the decode
scan synthesises per-channel run addresses directly and never consults the
TLB, so only the prefill half exercises the enumerator's placement model.

Useful conclusion: keep the fixtures as a **regression lock** on a
transcription (they will catch a drift between the two files), but do not
quote them as validation of the enumerator's model of A1.

### 2.2 The `wl_pipeline_D4` mismatch is an allocator accident, not semantics

An earlier version of this document said the DAG "issues, per tier, sweeps at
both the full resident length and separately at that tier's `history_len`".
**That was wrong.** The real cause, found by audit and then reproduced
independently:

`NoReuseKVLayout.finalize` ([workload_runner.py:1057](../../src/workload_runner.py#L1057))
is a global bump allocator with 8 KiB-padded spans that wraps to a new tile
whenever `cursor + span > 8 MiB`. When that wrap falls between a request's
`::history` block and its `::no-reuse-input` block, the two extents stop being
address-adjacent, so `CacheBlendTLB.scan_runs` can no longer coalesce them and
emits **two** runs instead of one.

Replaying the allocator reproduces the gap exactly:

| tier | layers where the two extents are NOT adjacent | extra runs |
|---|---:|---:|
| `t1n0` | 10 of 32 | 10 × 1,058 = 10,580 |
| `t2n0` | 11 of 32 | 11 × 1,058 = 11,638 |
| `t3n0` | 11 of 32 | 11 × 1,058 = 11,638 |
| | | **33,856** |

and 693,568 − 659,712 = **33,856**, matching the recorded signature counts at
L=256/512/768 term for term. The "clean per-tier structure" the earlier
write-up saw was the coincidence 10 + 11 + 11 = 32.

The same mechanism explains two further cases, on a *different* workload
(`workload_llama7b_small`, whose `l384` carries 128 input rows + 256 history
rows) — which is what makes it a mechanism rather than a coincidence:

| model | layers | non-adjacent layers | predicted extra runs | measured |
|---|---:|---:|---:|---:|
| LLAMA-7B | 32 | 0 | 0 | 0 (**MATCH**) |
| LLAMA-65B | 80 | 1 (`l384`) | 1 × 16 = **16** | **16** |
| GPT-175B | 96 | 1 (`l384`) | 1 × 16 = **16** | **16** |

and the split shows up exactly as predicted: the L=384 sweeps for that layer
become L=128 (input) plus L=256 (history), i.e. −16 / +16 / +16. A first
replay of the allocator predicted 0 splits for all three and was wrong — it
omitted the per-request **output** blocks, which also consume cursor space.
Including them reproduces all three cases.

**This makes the DAG itself the defect, not the enumerator.** The same
algorithmic scan is priced two different ways depending on unrelated
allocation history — A1's simulated baseline cost is a function of an
allocator's cursor. The four MATCH workloads also have `history_len > 0`; they
match only because they are small enough that the wrap never lands there.

Recommended fix is on the DAG side: make the run decomposition follow logical
extent structure (history and prefill always fused, or always separate) rather
than byte adjacency. Then re-record ground truth and re-run all five
comparisons — the four MATCHes would change too.

### 2.3 Two coverage gaps closed, and what is still open

An audit found that two code paths had never been compared against the DAG at
all. Both are now covered by recorded ground truth under
`tests/fixtures/dag_free/`:

* **Head folding.** `heads_per_hbm == 1` in every earlier comparison, where
  the slice rule degenerates to plain round-robin, so the head-folding loop —
  the only non-trivial half of `_slice_channel_rows` — was untested. (An
  earlier version of this document wrongly claimed LLAMA3-8B varied it; GQA
  changes the query batch, not `heads_per_hbm`.) LLAMA-65B gives 2 and
  GPT-175B gives 3. Result: **every decode run matches exactly** in both. The
  head-folding rule is correct; the only divergence in those runs is the
  prefill allocator defect above.
* **The one-row tail sweep.** `total_length % cap_rows == 1` was predicted to
  mismatch, because `shared_kv` was inferred from `shared_queries > 1` rather
  than carried. `A1PimInput` now carries it explicitly (mirroring
  `op.pim_shared_kv = True`), and a purpose-built workload with two such
  requests **MATCHes** on LLAMA-7B and LLAMA-65B.

Still open:

* `physical_runs` is a per-HBM-normalised *priced-invocation* count, not a
  machine-physical one: `num_hbm`, `num_hbm_used` and `num_attacc` all
  multiply PIM cost without a `pim_run_multiplicity` declaration.
* `gemv_buffer_bytes`, batch size and `pim_batch_command` are hardcoded in the
  enumerator, so a future A5/A6 reuse would diverge silently.

**Still missing:** the enumerator produces `pim_cycles_unordered` — unscheduled
work. It has **no scheduler**, so it cannot be compared to a DAG *makespan*,
and its report says so.

## 3. LLAMA3-8B — transfer validation and a fair cost comparison

`run_llama3_8b.sbatch` runs A1 on LLAMA3-8B from an **empty** signature cache.
Comparing against a warm cache would flatter the analytic model: the point is
the cost of producing a number that does not exist yet.

### 3.1 Transfer test — the model has never seen this transformer

A model fitted on the LLAMA-7B-derived cache **only** is scored against the
314 Ramulator results the cold LLAMA3-8B references produced, **none of which
appear in the calibration cache** (`entries_also_in_baseline: 0`). No refit.
LLAMA3-8B is not a rescaling of LLAMA-7B: `gqa_size = 4` changes the query
batch (and hence `shared_queries`), though not `heads_per_hbm`.

| | result |
|---|---|
| Layer 1 command counts | **314 / 314 exact** |
| Layer 3 cycles | **MAPE 6.84%**, p50 4.68%, p95 21.48%, max 27.81% |
| runs inside the calibrated envelope | 314 / 314 |
| uncalibrated / extrapolated | **0 / 0** |

That p95 of 21% is the honest cost of transferring to an unseen transformer,
and it is worse than the 4–6% in-distribution cross-validated figure — as it
should be. For contrast, the v1 model put **49.2%** of LLAMA3-8B's physical
runs on `wl_pipeline_D4` into buckets that did not exist and priced them
silently at ~154% MAPE. Reproduce with `validate_transfer.py`; the report is
`llama3_8b/transfer_all.json`.

The shipped `timing_models.json` is then recalibrated **including** these
LLAMA3-8B results, so the number above is a record of what the model could do
before it saw them, not a property of the shipped file.

### 3.2 Same answer? DAG + Ramulator vs DAG + analytic, cold cache

Both cells are the same `main.py` invocation on the same model and workload,
differing only in `--analytic-pim-model`. GPU, DIE and LINK numbers are
bit-identical in both, as they must be.

| quantity | `wl_pipeline_D4` | `wl_N4` |
|---|---|---|
| makespan | 8.7016 vs 8.5096 s → **+2.26%** | 17.8604 vs 17.4458 s → **+2.38%** |
| **`pim_time_s_unoverlapped`** | 3.6812 vs 3.5006 s → **+5.16%** | 7.7675 vs 7.3755 s → **+5.31%** |
| `pim_pool_time_s_unoverlapped` | 1.6976 vs 1.5831 s → +7.23% | 3.4429 vs 3.2086 s → +7.30% |
| `energy_nj` | 4.47657e12 both → **0.0000%** | 9.42516e12 both → **0.0000%** |

Two things must be said about that table or it misleads.

**The energy agreement is not evidence for the timing model.** PIM energy is a
pure function of the command counts and the layer's logical GEMM shape:
`dram_energy = Σ traffic·io_table + mac·32·64·e_mem`, `cal_energy =
flops/2·e_alu` — verified numerically as
`E_nJ = 360.448·mac + 3.8912·wrgb + 3.5072·(mvsb+mvgb) + 0.00256·m·n·k·numOp`
for BA / num_hbm=5 / NUM_ATTACC=8. **`cycle` appears nowhere in it, and `sfm`
is never used at all.** Energy being bit-identical confirms Layer 1, which was
already exact by construction. It says nothing about Layer 3.

**The makespan error is diluted.** GPU time is identical in both cells and
dominates the critical path, so the +2.3% on makespan understates the model.
The honest per-model figure is `pim_time_s_unoverlapped`: **+5.2%**, in the
same direction on both workloads, consistent with the run-weighted bias
measured in §1.

### 3.3 Does it actually go faster? Cold cache, wall clock

| workload | cell | total wall | DAG construction | `dag_finalize_s` | PIM pricing |
|---|---|---:|---:|---:|---:|
| `wl_pipeline_D4` | Ramulator | 1328.6 s | 695.2 s | 176.1 s | **453.9 s** (159 subprocesses) |
| | analytic | 865.4 s | 684.5 s | 177.3 s | **0.0098 s** |
| `wl_N4` | Ramulator | 2441.3 s | 1383.9 s | 363.7 s | **687.3 s** (224 subprocesses) |
| | analytic | 1759.2 s | 1395.8 s | 356.9 s | **0.0172 s** |

(LLAMA3-8B. `dag_build_s` contains the pricing done during construction, so
the DAG column subtracts it. The wrapper now also records `pricing_wall_s`,
measured once around the whole dispatch — the per-job totals it used to report
were sums across worker threads and would have overstated the subtraction on a
pricing-parallel workload.)

**The Ramulator subprocess is no longer the bottleneck — the DAG is.**

---

## 4. The 2×2: {event DAG, DAG-free enumerator} × {Ramulator, analytic}

`run_2x2.sbatch` runs one cell; `collect_2x2.py` assembles the table. Every
cell: LLAMA-7B, tp=8, num_hbm=5, `--powerlimit`, **empty signature cache**, one
Ramulator worker. Aligning those mattered: the two entry points disagreed by
default on five settings that change A1's PIM work or its pricing (power
constraint — `main.py` ON, the enumerator runner OFF; model; tensor parallel;
worker count; and a pandas CSV mirror the DAG path never writes).

Only quantities that mean the same thing in all four cells are compared.
**`makespan` is not one of them** — the enumerator has no scheduler. The PIM
time column is the DAG's `pim_time_s_unoverlapped` (prefill sweeps, device
`PIM`) **plus** `pim_pool_time_s_unoverlapped` (decode channel runs, device
`PIM:pool*`), against the enumerator's unordered cycle sum; both are sums of
per-run durations. The energy column is the two `pim_kv_scan…` `by_event`
entries, not `by_class['PIM']` (which also holds bandwidth-priced KV stores the
enumerator does not model).

### `wl_pipeline_D4`, LLAMA-7B, cold cache

| cell | PIM scan energy (nJ) | PIM device time (s) | work list (s) | pricing (s) | **wall (s)** |
|---|---:|---:|---:|---:|---:|
| DAG + Ramulator *(reference)* | 1.0132e12 | 1.293199 | 904.7 | 455.9 | **1360.7** |
| DAG + analytic | 1.0132e12 | 1.352908 | 876.2 | 0.060 | **876.2** |
| enumerator + Ramulator | 1.01258e12 | 1.277039 | 3.0 | 505.1 | **508.1** |
| enumerator + analytic | 1.01258e12 | 1.340517 | 3.0 | 0.004 | **3.0** |

Relative to the reference, the two axes factor cleanly:

| cell | energy | PIM time | wall speed-up |
|---|---:|---:|---:|
| DAG + analytic | **+0.000%** | **+4.62%** | 1.6× |
| enumerator + Ramulator | **−0.061%** | **−1.25%** | 2.7× |
| enumerator + analytic | −0.061% | **+3.66%** | **458.6×** |

Reading it:

* **The pricer axis moves time, never energy.** Swapping Ramulator for the
  analytic model changes PIM energy by **0.000%** and PIM time by **+4.6%**
  (DAG) / **+4.9%** (enumerator: −1.25% → +3.66%). That is not a coincidence
  and not a validation: PIM energy is a function of the command counts and the
  logical GEMM shape, with no `cycle` term at all. Note the two cells do *not*
  share counting code — the Ramulator cell reads counters off the generated
  trace, the analytic cell evaluates a closed form — so the agreement is a
  *measured* Layer-1 result (12,937 / 12,937 exact), not a tautology. It is
  still evidence about Layer 1 only, and says nothing about the timing model.
* **The engine axis moves energy and work, never the pricer's error.** The
  enumerator's −0.061% energy and −1.25% time are the 33,856 runs it does not
  emit — the allocator tile-wrap of §2.2, which is a DAG-side defect rather
  than an enumerator error.
* **The two effects compose**: −1.25% × 1.0497 ≈ +3.66%.
* **Wall clock: 1360.7 s → 3.0 s, 458.6×.** The pricer removes ~456 s of
  simulation; the engine removes ~900 s of graph construction. Neither alone
  gets past 2.7×.

The `+4.6%` is the same systematic over-price measured per-run in §1: it
inflates the A1 baseline and therefore Fugue's speed-up.

### `workload_rag_shared_p24_s8`, LLAMA-7B, cold cache

A second workload with the opposite balance: pricing-dominated (46 distinct
pricing keys but 245,504 physical runs, and no `history_len`, so the allocator
defect of §2.2 cannot bite). The enumerator reproduces the DAG's multiset
**exactly** here — 245,504 / 245,504 runs, 46 / 46 signatures, all five
aggregate command totals exact — which isolates the pricer axis cleanly.

| cell | PIM scan energy (nJ) | PIM device time (s) | work list (s) | pricing (s) | **wall (s)** |
|---|---:|---:|---:|---:|---:|
| DAG + Ramulator *(reference)* | 1.20419e12 | 1.121167 | 92.1 | 1227.6 | **1319.7** |
| DAG + analytic | 1.20419e12 | 1.180495 | 55.6 | 0.029 | **55.7** |
| enumerator + Ramulator | 1.20419e12 | 1.117498 | 0.003 | 1229.5 | **1229.5** |
| enumerator + analytic | 1.20419e12 | 1.180374 | 0.003 | 0.015 | **0.02** |

| cell | energy | PIM time | wall speed-up |
|---|---:|---:|---:|
| DAG + analytic | **+0.000%** | **+5.29%** | 23.7× |
| enumerator + Ramulator | **+0.000%** | **−0.33%** | 1.1× |
| enumerator + analytic | **+0.000%** | **+5.28%** | **75,204×** |

**Energy is identical in all four cells** — with no history there is no
allocator split, so the enumerator's multiset is exact and both axes agree to
the last digit. **The pricer's +5.3% reproduces on both engines** (5.29% vs
5.28%) and on a workload whose shape is nothing like `wl_pipeline_D4` (46
pricing keys vs 27, 245k runs vs 694k). That is now a stable, twice-measured
property of the timing model, not a workload artefact.

### Which leg matters depends on the workload

| workload | DAG build | Ramulator pricing | pricer alone | engine alone | both |
|---|---:|---:|---:|---:|---:|
| `wl_pipeline_D4` | 905 s | 456 s | 1.6× | 2.7× | **458.6×** |
| `workload_rag_shared_p24_s8` | 92 s | 1228 s | 23.7× | 1.1× | **75,204×** |

`wl_pipeline_D4` is DAG-bound (4 agents, 8,464-row prefills, 1.2 M events);
`rag_shared` is pricing-bound (24 agents sharing a corpus, so few distinct
lengths but many physical runs). Neither leg alone is enough on both:
replacing only Ramulator gets 1.6× on one and 23.7× on the other; replacing
only the DAG gets 2.7× and 1.1×. **The 458× / 75,204× need both.**

One caveat on the wall-clock subtraction: on `rag_shared` the DAG-construction
column differs between the two DAG cells (92.1 s vs 55.6 s) even though the
graph is identical. `dag_build_s − pricing_wall_s` is not a perfectly clean
separation — the Ramulator cell's construction also pays for trace files and
subprocess I/O interleaved with it.

### Cost of the work list itself

The enumerator builds the same work list in **3.0 s** against 904.7 s of DAG
construction. But its cost is Θ(`decode_channel_runs`), and profiling shows 67%
of it is frozen-dataclass construction, hashing and equality on the `Counter`
key — one `A1PimInput` per *physical run* — not layout arithmetic. Measured at
the top of the sweep: `wl_N64` is **107 s** at LLAMA-7B and **352 s** at
GPT-175B, per (workload, model, num_hbm) point.

No field of a decode `A1PimInput` depends on the layer or on the request, so
the whole `for layer` loop and the per-output-token loop collapse into a walk
over 256-row chunk bands. A closed form doing exactly that measures
**0.0007 s** on `wl_pipeline_D4` (4592×) and **0.078 s** on `wl_N64`/GPT-175B
(4536×). It is verified bit-identical and is being landed.

## 5. Corrections to earlier versions of this document

**v1 model (a per-configuration lookup table), retired to `archive_v1/`:**

* "cache-wide MAPE 4.56% / 0%" was a **training-set** number: fitted on all
  12,160 pairs and scored on the same pairs. On the `chunkstripe1` buckets the
  prediction at an anchor was the mean of the observations at that anchor, so
  0% was arithmetic, not accuracy.
* The "192-row holdout, 0.00% error" was not a holdout: 192 rows folds to
  `run_length = 256`, already an anchor in nearly every channel bucket.
* Measured out-of-sample on the 944 entries appended after it was calibrated:
  0.00% on memorised anchors, 1.83% interpolating, **14.08%** (p95 52%, max
  197%) extrapolating, and **~154%** (p95 1100%) whenever a bucket did not
  exist — silently. On `wl_pipeline_D4`, 49.2% of LLAMA3-8B's runs landed in
  non-existent buckets.

**v2 (the three-layer model) as first written up — my own errors, found by
audit and verified independently:**

* **Split leakage.** The held-out split was on `run_length`, but the model's
  input is the feature vector, and `ceil` steps collapse 411 run lengths onto
  49 distinct inputs. 98.6% of `legacy|replicate`'s "held-out" rows were
  byte-identical to a training row. Reported 4.03%; leak-free 5.56% ± 2.55,
  and 8.30% ± 5.22 under the configuration protocol.
* **`channel_base` is inert** — accepted by `_geometry` and never read — so
  the "held-out configurations" protocol split on a field the model cannot
  see, while leaving the K/V byte offsets (which drive `row_openings`) on both
  sides. `_config_id` is corrected.
* **The refresh stretch was an unidentifiable gauge**, verified to <1e-8. The
  grid search was a no-op and "the fit recovers nCCDAB ≈ 0.79–0.99" quoted a
  gauge-dependent number. Removed.
* **Cross-protocol quoting.** The shipped file carried one protocol's error
  next to another protocol's coefficients. Fixed: the shipped model is refit on
  all inputs and carries the *procedure's* cross-validated error.
* **"Eight coefficients per regime"** — five come back exactly zero for the
  regime A1 uses. It is 3 effective parameters, which is a stronger argument.
* **`_nnls` dropped columns by raw magnitude**, so the drop order depended on
  the units a feature happened to use. Columns are now scaled to unit norm.
* **The `wl_pipeline_D4` diagnosis was wrong**: it is not "the DAG issues a
  separate history sweep per tier", it is an 8-MiB tile-wrap accident in the
  KV allocator (§2.2), reproduced exactly.
* **The four DAG-free MATCHes were presented as validation.** They are
  agreement between a transcription and its original (§2.1).
