# Does the analytic model's error grow with the workload?

Short answer: **no, in both senses of "bigger", and both were measured rather
than argued.**

The question splits, because "a bigger workload" can mean two different things
and they have different answers:

| "bigger" means | what happens | why |
|---|---|---|
| more **runs** (more requests, more tokens, more layers) | the aggregate error **converges** and stops moving | it is a truth-weighted mean of per-run ratios |
| longer **contexts** (each run scans more KV rows) | the error **flattens to an asymptote**, and changes sign | the pinned MAC term comes to dominate everything else |

Neither case diverges. What does change with context length is the *sign*: the
model reads high on short runs and low on very long ones.

---

## 1. More runs cannot make it worse

The workload-level number is `sum(predicted) / sum(measured)` over every PIM
run. That is a truth-weighted mean of the per-run ratios, so it is bounded by
their min and max and converges as the count grows. Resampling the 314 measured
A1 runs (`aggregate_vs_workload_size.csv`):

| runs in the workload | aggregate pred / truth |
|---:|---:|
| 10 | 1.0294 |
| 100 | 1.0303 |
| 1,000 | 1.0284 |
| 10,000 | 1.0280 |
| 100,000 | 1.0281 |
| 1,000,000 | 1.0281 |

Per-run ratios span **0.885 – 1.238**. The aggregate has settled by a thousand
runs and does not move over the next three orders of magnitude. A workload ten
times larger reports the same few percent, not ten times more.

## 2. Longer contexts: measured to 6.7x past the calibrated edge

The model is calibrated on run lengths up to **39,168** rows. To test what
happens beyond that, A1's two real run shapes were priced with Ramulator itself
at lengths up to **262,144** rows and compared against the model
(`long_context_probe.csv`, reproduce with `probe_long_context.py`):

| KV rows in one run | decode (1 channel) | prefill (16 channels) | |
|---:|---:|---:|---|
| 8,192 | 0.9784 | 0.9694 | inside |
| 16,384 | 0.9751 | 1.0298 | inside |
| 32,768 | 0.9561 | 0.9914 | inside |
| **39,168** | 0.9622 | 0.9922 | **calibrated edge** |
| 49,152 | 0.9542 | 0.9932 | extrapolating |
| 65,536 | 0.9562 | 0.9866 | extrapolating |
| 98,304 | 0.9500 | 0.9900 | extrapolating |
| 131,072 | 0.9484 | 0.9843 | 3.3x past |
| 196,608 | 0.9479 | 0.9811 | 5.0x past |
| 262,144 | **0.9469** | **0.9776** | **6.7x past** |

The error **flattens** instead of growing: decode settles near -5.3%, prefill
near -2.2%, and both are still there at a quarter-million rows.

That is structural, not luck. As the run gets longer the MAC term dominates
every other term, and the MAC coefficient is **pinned to the datasheet
`nCCDAB`, not fitted**, so the ratio has to converge to a constant. A model
whose leading coefficient were fitted would have no such guarantee.

### The decode asymptote has a name

Decode converges to **0.947**, i.e. the model reads 5.3% low. The HBM3
refresh duty cycle in the simulated part is
`nRFCSB / nREFI = 260 / 5070 = 5.13%` -- the fraction of time an all-bank
refresh takes the bus away from the command schedule. A pure command-schedule
model cannot contain that, and the gap is **0.6 percentage points** from the
datasheet value.

**A refresh term was tested and deliberately not added.** Pinning the stretch
to the datasheet made the held-out error *worse* inside the calibrated range
(2.78% -> 3.34% for `chunkstripe1|replicate`): where the runs actually are, the
other coefficients already absorb refresh, and forcing the split double-counts
it. Refresh only becomes separable in the asymptote. See
`../RESULTS.md` for the cross-validation protocol.

## 3. Error against run length, inside the calibrated range

`error_vs_run_length.csv`, one row per distinct run length in the ground truth:

| run length | n | mean ratio | MAPE |
|---|---:|---:|---:|
| <= 256 | 85 | 1.0094 | 5.84% |
| 256 - 1K | 86 | 1.0389 | 5.05% |
| 1K - 4K | 157 | 1.0276 | 3.70% |
| 4K - 8K | 151 | 0.9999 | 1.21% |
| 8K - 16K | 306 | 1.0172 | 2.41% |
| 16K - 32K | 147 | 0.9727 | 2.73% |
| > 32K | 4 | 0.9616 | 3.84% |

Trend of the ratio against `ln(run_length)`: **-0.0086 per e-fold**. Over the
whole calibrated range (6.0 e-folds) that is a drift of -0.052 -- the model
crosses from over- to under-prediction around 4-8K rows.

## 4. What this does and does not license

* **Safe:** scaling the number of requests, agents, layers or output tokens.
  The reported error is a property of the run *shapes*, not of how many there
  are.
* **Safe with a known offset:** contexts well past 39,168 rows. The model reads
  about 5% low on decode runs and 2% low on prefill sweeps, and
  `estimate()` counts every such run as `extrapolated` in
  `analytic_diagnostics` so the report says how much of the workload was
  outside the calibrated box.
* **Not covered:** a different `pim_type` (only BA is calibrated), a different
  `dbyte`, or a batch-command scheme with no fitted regime -- those raise
  rather than guess.
* **Worth remembering:** the sign flips. Quoting "the model is ~4% high" is
  right at A1's current operating point and wrong at 100K-row contexts.
