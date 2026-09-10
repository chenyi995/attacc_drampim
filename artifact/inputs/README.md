# Frozen paper inputs

The simulator consumes these five saved workload configurations: object lengths,
stable object identities, reader views, warm-pool membership, and exact selected
logical positions. They are derived input configurations, not saved timings or
expected speedups. The package contains no source text, token-ID arrays, KV
tensors, model weights, generated answers, or raw experiment outputs.

From the repository root, prepare a verified working input directory with:

```bash
python3 artifact/inputs/prepare_inputs.py --output output/inputs
```

This command needs only Python's standard library and works offline. The request
runner can also read `artifact/inputs` directly. Running `prepare_inputs.py`
without `--output` verifies the bundled files without writing anything. Different
existing files in the destination cause an error and are preserved.

The verifier checks frozen file hashes, exact case and selected-position hashes,
object-length sums, reader counts, and warm-pool sizes. During packaging, every
object ID was also checked against the original token-ID array and token count.
Those arrays remain in the author's source archive; the public verifier does not
claim to re-tokenize the source text or validate numerical attention.

| Workload | Prompt N | Selected Q | Retained | Replaced | Private input | Readers | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Document 8k | 8,026 | 58 | 7,968 | 48 | 10 | 1 | 16 |
| Readers 4 | 8,028 | 60 | 7,968 | 48 | 12 | 4 | 16 |
| MuSiQue source request | 8,082 | 1,341 | 6,741 | 1,283 | 58 | 1 | 32 |
| Long context 32k | 32,034 | 188 | 31,846 | 176 | 12 | 2 | 16 |
| LTM recall | 15,832 | 144 | 15,688 | 64 | 80 | 1 | 16 |

All token counts are per reader. In selective layers, `N = retained + replaced +
private input` and `Q = replaced + private input`. Output lengths are fixed timing
horizons, not generated-answer lengths or observed EOS positions. TTFT includes
the first output token; the subsequent decode horizon is `output_tokens - 1`.

## Original software and data sources

The local source checkouts and frozen source manifests establish these versions:

| Source | Original repository or dataset | Pinned revision |
| --- | --- | --- |
| CacheBlend | [YaoJiayi/CacheBlend](https://github.com/YaoJiayi/CacheBlend) | `55ad02675939f783a38d579393527d218a7fd581` |
| EPIC | [DerekHJH/epic](https://github.com/DerekHJH/epic) | `3204410f1723ed0f39575b4eba8c17578a1ee1a1` |
| LongMemEval prompt | [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) | `9e0b455f4ef0e2ab8f2e582289761153549043fc` |
| LongMemEval oracle evidence | [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | `98d7416c24c778c2fee6e6f3006e7a073259d48f` |

`input-provenance.json` records upstream revisions, source-file hashes, original
case hashes, the recorded EPIC corpus order, and the exact tokenizer checksum.
`source-downloads.json` provides immutable download URLs and byte hashes for the
source files used in these inputs. For optional source inspection, run:

```bash
python3 artifact/inputs/download_sources.py --output output/upstream-sources
```

This downloads and verifies 35 files totaling 19,516,011 bytes (about 18.6 MiB).
It does not install or execute the upstream software. `--list` shows the pinned
files without downloading; `--check-only --output output/upstream-sources`
verifies an existing download offline. These files are provenance evidence and
are not needed by the simulator. They remain local and outside version control.

EPIC's repository contains an [Apache-2.0 license](https://github.com/DerekHJH/epic/blob/3204410f1723ed0f39575b4eba8c17578a1ee1a1/LICENSE).
CacheBlend contains an [Apache-2.0 license in its vllm_blend subtree](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/LICENSE);
the inspected commit has no root license or separate MuSiQue data license.
The LongMemEval [code license](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/LICENSE)
and the saved [dataset card](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/blob/98d7416c24c778c2fee6e6f3006e7a073259d48f/README.md)
state MIT. Repository code licenses are not treated here as separate grants for
third-party essay or benchmark text. This artifact therefore distributes the
derived shape configuration and source references; original source files are
fetched directly from their publishers when requested.

## Software-policy and timing boundaries

The hardware runner models request shapes and memory commands. It does not run
CacheBlend, EPIC, vLLM, or LongMemEval neural inference; it does not evaluate
answer quality. Upstream software supplies the source text, formatting rules,
and reuse-policy definitions used to prepare the frozen inputs.

All five inputs use the same frozen native-LLAMA SentencePiece adaptation,
including the recorded segment BOS convention. They are not re-tokenized per
simulated model. The tokenizer's SHA256 is
`9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347`.
The inspected records do not identify its original download URL; the artifact
does not substitute another tokenizer. The frozen input package requires none.

- **Document 8k and Readers 4:** the EPIC sentence-boundary splitter starts at
  2,048 tokens, producing document lengths 3,075, 3,074, and 1,854. It selects
  the first 16 tokens of each document and the complete query suffix in every
  layer. Readers 4 is a constructed synchronous fan-out over the same documents;
  all four readers are ready after warmup. Its 12-token suffix differs from the
  10-token Document 8k suffix, so the pair does not vary reader count alone.
- **Long context 32k:** the same EPIC rule uses 11 documents and two synchronous
  readers. The 32k source-text control was selected by the recorded capacity
  admission rule, without selecting a length by measured speedup. The exact
  document lengths and reader identities remain in `selected.json`.
- **MuSiQue:** the source is zero-based row 75 in CacheBlend's pinned
  `inputs/musique_s.json`, with all ten source documents. The saved shape selects
  `floor(0.16 * 8024) = 1283` cached positions plus all 58 query-suffix tokens.
  Cached positions are deterministic, count-preserving evenly spaced positions;
  they are not measured numerical top-k outputs. The first layer is full,
  the second uses full QKV updates with selective attention, and subsequent
  layers are selective. The complete suffix is recomputed under the frozen
  wrapper's recorded adaptation.
- **LTM recall:** source question `d851d5ba` is a single read of four past
  LongMemEval oracle-evidence sessions. Retrieval is assumed complete. The
  sessions contain 15,752 tokens and are independently prewarmed at reference
  position zero; consumer starts are 35, 4,159, 7,754, and 11,596. The 80 fresh
  prefix/date/question/answer-cue tokens remain private and are fully computed.
  The adapted EPIC rule also recomputes 16 tokens per history session. There is
  no preceding generated answer or continuing runtime KV state, so this is not
  a stateful multi-round execution. The reuse adaptation has no measured
  LongMemEval answer-quality result.

Warmup is separate from the timed request. The frozen non-LTM controls include
system, document, and query objects in their warm pools; their query positions
are nevertheless fully recomputed. LTM prewarms only past history, excluding
the current prefix, date, question, and answer cue. The exact pool definitions
are preserved, including this difference. Warm-pool totals are 8,026, 8,064,
8,082, 32,046, and 15,752 tokens, respectively, in table order.

The five common inputs and five models are the paper's selected showcase,
not a complete-dataset or model-family average. All models use the same input
objects and selected positions. `scope.json` preserves the paper's stated
selection reason. Original experiment output and omitted cases remain separate
from this runtime input package.
