# 1-core vs 4-core comparison

Existing single-core result: `outputs/rerun_rope_nopipe_noff/output_agent256_rope.csv`

4-core command:

```bash
python3 main.py --system dgx-attacc --pim bank --lin 2048 --lout 128 --batch 1 --num-agent 256 --sim-cores 4 --powerlimit --rerun-ramulator
```

Parameters matched: GPT-175B, W16A16, A100a x8, PIM bank, Lin=2048, Lout=128, batch=1, agent=256, RoPE on, powerlimit=True, pipe=False, ff_parallel=False.

Rows compared: single-core=1, four-core=1
Headers identical: True
Exact differing fields: 0

The two output rows are exactly identical.

Archived files:

- `output_single_core_existing.csv`
- `output_sim_cores_4.csv`
