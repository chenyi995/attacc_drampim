# Final paper microbenchmarks

This package reproduces the final two-panel attention crossover (Fig. 8), the frequency calculation (Fig. 9), and the area values in the evaluation text. It also provides the numeric Fig. 1a attribution for the fresh Llama 7B / Readers 4 request run. Historical six-panel figures, other workloads and an area plot are not part of this package.

Run these commands from the artifact root after building `simulator`. Every output directory must be new.

```bash
python3 microbench/run.py crossover --simulator simulator --runtime results/build/runtime --output results/crossover --jobs 8
python3 microbench/run.py frequency --simulator simulator --output results/frequency
python3 microbench/run.py area --output results/area
```

Adjust the runtime path to the fresh build directory selected by the artifact's top-level runner. `--no-plot` keeps all fresh numeric outputs and omits rendering. `python3 microbench/run.py crossover --plan` prints the complete final sampling input without executing a simulator. Dependencies are Python 3.10 or newer, NumPy, pandas, PyYAML and Matplotlib, plus the artifact's built Ramulator executable and shared library. The analytical frequency and area commands do not require a simulator binary or a licensed synthesis tool.

## Attention crossover

The frozen input is Llama 7B, tensor parallelism **1**, 1,024 cached tokens and a 300 GB/s GPU–memory link. Tensor parallelism comes from the source experiment's recorded plan, not from the name of a shared simulator build directory. The plain path has 35 query-count samples; MQ has 275. These are exactly the final figure's 310 samples, including retained points outside the visible 16–1,536 range. No additional panels or parameter sweep is executed.

The runner calls the frozen `fugue.kvchime_model.Model`, native trace generator and real Ramulator. It prepares 537 logical profile requests, shares only exactly matching trace bytes, resident lists, YAML timing and implementation identities inside the new run, and saves full traces, commands, stdout, timing and hash manifests. The private-K overlap window is read from emitted commands. The original grouping and complete GPU/PIM service equations are included as unchanged source excerpts in `source/crossover_kernel.py`; the adapter only manages paths, jobs, CSVs and plotting.

The expected sampled crossing brackets are 54–55 queries for plain PIM and a nine-transition envelope of 512–545 for MQ. The runner compares both service operands at every sample with the final paper reference, checks identical GPU values at common query counts, and fails if values differ. Throughput is query count divided by per-layer attention service time, in tokens/s. It is not complete request generation throughput. The two panels use the same union of 290 GPU samples; neither curve is numerically interpolated to invent timing samples.

Outputs include `service-operands.csv`, `checks.json`, `run.json`, `profiles/`, `exact-cache/` and `plot/figure.pdf`. A failure retains all generated records. `reference/crossover/` is a comparison target, never the timing input for a fresh run.

## Frequency

The original exact-Fraction derivation parses the frozen `src/config.py` and `fugue/attention.py` expressions. It regenerates all 626 frequency samples and compares every interval to the retained final reference. The balance is exactly `1000/769` GHz (displayed as 1.3 GHz), at eight DRAM cycles with eight resident queries.

This is the paper's analytical command-interval model with fixed column and operation energies. It is not a measured RTL frequency/power sweep. Outputs include the original operands and limitations in `derivation.json`, the four plotting CSVs, and the regenerated PDF/PNG.

## Area

`area/source-measurements.csv` contains the exact top-level Cell-Area and required QOR operands extracted from the 13 final Genus synthesis points. Every row identifies the original reports by path and SHA-256; the raw reports remain local. `run.py area` checks each extracted period/slack/violation count against the component table and invokes the original physical-layer roll-up function. All 13 points have nonnegative saved pre-layout slack and zero reported violating paths.

Bank/BG totals use the recorded counts, the adopted 10× DRAM density factor, eight DRAM dies per stack and the full 121 mm² DRAM-die denominator. The 0.83 mm² bank footprint includes the adopted footprint convention; the remainder is an accounting budget, not post-layout verified free area. Buffer/controller components use their synthesized logic-process areas and are counted once, without either DRAM scaling factor.

The final arithmetic gives 1.357029818% added Bank/BG logic, 8.202814942% total PIM logic and 3.995532165% remaining accounting margin per DRAM die. Selected buffer/controller components add 1.034143149 mm² and total 1.625371614 mm². `paper-area.csv`, `area.json` and the parsed component table retain full precision.

The measurements came from Cadence Genus with ASAP7 at the recorded TT 0.7 V corner and hierarchical synthesis flow. This artifact reproduces the published arithmetic from those extracted measurements. It does not claim to rerun licensed synthesis: the original flow requires Genus and the external process libraries, and those tools/libraries are not bundled. Synthesis is not invoked by any command above. No area figure is generated because the final manuscript reports these numbers in prose.

## Fig. 1a request integration

The public adapter is:

```python
from microbench.fig1a import derive_prefill, export
report = derive_prefill(impl, model, raw, case, build_view, moved,
                        profiles_root, workload_id=case["id"])
export(report, output_directory)
```

Here `impl` is the final `hbm_2k_model` module, `model` is the configured model used by the completed request worker, `raw` contains the fresh `event_blocks`, `decisions`, `summary` and `baseline_residency`, and the view helpers are the request runner's own helpers. This entry accepts only Llama 7B, four readers, resident GPU KV and prefill F0/F1/F4. It invokes no simulator, profile preparation or selector prediction.

The original operator inspection class is compiled from an unchanged AST excerpt. A shallow inspection copy of the configured model evaluates the same analytical operators, leaving simulated timing untouched. Attribution sums each operator's memory bound plus exposed compute, keeps `min(compute, memory)` as overlap inside memory, adds only exposed external transfers and retains PIM scan as coupled service. Every event sum and complete TTFT must match the fresh request result. Only the three prefill rows and their blocks are exported.

The numeric plot uses the final standalone Readers 4 rendering parameters. The manuscript's hand-maintained composite also contains an illustrative execution timeline; the adapter produces its numeric panel and does not claim to reconstruct the composite artwork. Display compression of the tall Full GPU bar is recorded separately from its actual milliseconds.

## Source identity and validation

`copied-sources.json` records original paths, SHA-256 values and line spans for the exact source excerpts. The callable grouping/service statements and operator attribution retain their original AST; no timing, energy, operator or area equations are reimplemented. Only portable roots, file selection and orchestration are new. `simulator-source-sha256.json` pins the source identities used by the microbenchmarks independently of the request-model wrapper revision.

The request API was checked against the final saved Readers 4 run: all 21 total/category/overlap values match (three configurations × seven fields), with complete event coverage. The frequency and area commands were executed in fresh staging directories and passed their reference checks. The final crossover validation used a newly compiled Ramulator, 537 logical profiles and 156 distinct physical replays in an empty cache. All 620 GPU/PIM service operands matched the paper references, and the figure rendered successfully. The fresh request-run Fig. 1a callback also matched all 21 final reference values exactly. Run and checks files identify each execution. Raw outputs remain outside this source package.
