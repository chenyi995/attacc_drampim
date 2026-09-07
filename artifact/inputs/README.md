# Fixed simulation inputs and provenance

`cacheblend.json` contains the two complete CacheBlend demo inputs selected by smallest native LLAMA token count among demo inputs 1–10. Text files `2.json` and `3.json`, per-segment token IDs, and the full selection-count record are included. Source commit, relevant software file hashes and tokenizer SHA-256 are recorded in `cacheblend-provenance.json`.

`epic.json` contains EPIC LongContext 4K/8K/16K source text shapes and the constructed 1/2/4-reader fan-out over the 8K documents. The three `Fugue-asplos-context-*.json` files retain source text, original chunked token IDs and construction metadata. The actual kvlink-16 recomputation indices and content-addressed cache objects are included. `epic-provenance.json` records the EPIC commit, relevant software hashes and tokenizer SHA-256.

These are preprocessed simulator inputs, not trace timings, expected speedups, generated answers or an inference cache. `python3 -m fugue check-inputs` checks all token IDs against object hashes, token counts, CacheBlend's 16% rule, EPIC's exact first-16/document plus full-query rule, and unique shared-cache capacities. The hardware simulator consumes these fixed tokenized inputs directly. It does not need the tokenizer model, source repositories or numerical LLM weights at runtime; it does not claim to retokenize or execute EPIC/vLLM inference. Source selection/tokenizer adaptation and fixed output-length caps remain part of the experiment definition.

`sweep.json` contains only the final experiment 1/2 `(Q,C)` points and bandwidth values. The union is simulated afresh. Numeric publication references are separate in `artifact/reference/` and are read only by the result checker after the run.

CacheBlend source: the supplied CacheBlend checkout, with its exact commit recorded in provenance. EPIC source: the source commit and source-file hashes identify the supplied EPIC checkout; its Apache-2.0 license is included. Text corpus filenames are recorded in `epic.json`. The included LLAMA tokenizer outputs are an adaptation to the native AttAcc LLAMA-7B model, not EPIC's default Llama-3.1-8B tokenizer/model configuration.
