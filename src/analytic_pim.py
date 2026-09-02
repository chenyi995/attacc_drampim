"""Principle-based analytic replacement for a bank-level AttAcc Ramulator run.

The model is a THREE-LAYER decomposition.  Each layer is calibrated and
validated against ground truth on its own, so a regression in one of them
cannot be hidden by a fit in another:

  Layer 1 -- command counts ``(mac, sfm, mvgb, mvsb, wrgb)``.
      Exact closed-form transcription of ``gen_trace_attacc_bank.py``.  NO
      fitted parameter.  Ground truth = Ramulator's own command counters;
      ``calibrate.py`` requires a 100% exact match over the whole cache.

  Layer 2 -- trace structure ``(barrier_groups, row_openings)``.
      Exact closed form of the SAME generator's command ORDER and address
      stream.  NO fitted parameter.  Ground truth = a regenerated trace;
      ``calibrate.py`` regenerates a sample and requires an exact match.
      This layer is what makes the cycle model physical rather than a curve:
      a barrier is a real all-channel synchronisation, and a row opening is a
      real ACT/PRE pair on the critical path.

  Layer 3 -- cycle count.
      A small linear form over Layer-1/Layer-2 quantities in DATASHEET units.
      The MAC term's coefficient is NOT fitted: ``mac_x_interval`` is already
      ``mac * nCCDAB`` in tCK, so its physical value is exactly 1, and pinning
      it is what removes the systematic over-price described in ``_fit_pinned``.
      The remaining command terms are non-negative; the intercept is free, and
      comes out negative -- it is a ramp-in correction, not a command cost.
      ``effective_parameters`` in the model file is the count that matters,
      not the length of FEATURE_NAMES.  The regime key is physical
      (trace revision x batch-command scheme) and contains no nuisance
      identity: keying on identity is what turned the previous version into
      a lookup table.

      There is deliberately NO separate refresh term.  A multiplicative
      "refresh stretch" 1/(1 - t_refresh/nREFI) is exactly unidentifiable in
      this parameterisation: the weighted-relative-error objective and the
      non-negativity constraint are both invariant under coefficients ->
      stretch * coefficients, so the prediction does not depend on it at all
      (verified numerically to 1e-9 across the whole grid).  Refresh cost is
      absorbed into the mac coefficient, which is where it physically lives:
      it is proportional to schedule length.

VALIDATION HONESTY.  The model's actual input is the FEATURE VECTOR, not the
run signature.  ``ceil`` steps collapse hundreds of distinct run_lengths onto
one feature vector (11,716 legacy cache rows carry only 49 distinct model
inputs), and ``channel_base`` / ``shared_kv`` / ``num_hbm`` / ``pim_type``
never reach the features at all.  A split on run_length or on a metadata
"configuration" therefore puts byte-identical samples on both sides and
reports training error.  ``calibrate.py`` splits on the feature vector and
reports the mean +/- std over several seeds, together with the effective
sample size.

Anything outside the calibrated bounding box is COUNTED when the caller
passes a ``diagnostics`` dict; with no diagnostics sink an extrapolated
estimate is returned without comment (only a missing regime raises).  The
box is axis-aligned, so an estimate inside an unsampled interior hole is not
flagged -- ``domain['run_length_largest_interior_gap']`` records how big such
a hole can be.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Result = Tuple[int, int, int, int, int, int]

# ---------------------------------------------------------------------------
# HBM3-PIM geometry and timing.  Every constant below is read off the
# simulated part, not fitted: geometry from gen_trace_attacc_bank.py, timing
# from ramulator2/src/dram/impl/HBM3-PIM.cpp preset HBM3_5.2Gbps[_NPC].
# ---------------------------------------------------------------------------
PREFETCH_BYTES = 32          # HBM_GS['col']
COLUMNS_PER_ROW = 32         # n_col
ROW_BYTES = COLUMNS_PER_ROW * PREFETCH_BYTES
N_BANK = 4
N_BG = 4
N_RANK = 2
N_PCH = 2
BANKS_IN_PARALLEL = N_PCH * N_RANK * N_BG      # 16 score rows per MAC step
NCCDAB_PC = 6                # power-constrained MAC_AB command interval
NCCDAB_NPC = 4               # unconstrained
NREFI = 5070                 # refresh interval, tCK

FEATURE_NAMES = ("mac_x_interval", "wrgb", "mvgb", "mvsb", "sfm",
                 "row_openings", "barrier_groups", "const")


def _ceil(x, y) -> int:
    return -(-int(x) // int(y))


def command_interval(power_constraint: bool, nccdab_override=None) -> int:
    """MAC_AB command-to-command interval in tCK (nCCDAB)."""
    if nccdab_override:
        return int(nccdab_override)
    return NCCDAB_PC if power_constraint else NCCDAB_NPC


# ===========================================================================
# Layer 1: command counts.  Exact; no fitted parameter.
# ===========================================================================
def command_counts(*, run_length: int, num_ops_per_hbm: int, dbyte: int,
                   dhead: int, channel_count: int = 16,
                   channel_base=None, shared_queries: int = 1,
                   mq_command: bool = False, phase: str = "full",
                   trace_revision: str = "legacy") -> Tuple[int, int, int, int, int]:
    """Return exact ``(mac, sfm, mvgb, mvsb, wrgb)`` trace command counts.

    Closed-form transcription of ``gen_trace_attacc_bank.py``.  The
    ``channel_base is not None`` convention is identical to the wrapper: it
    selects the head/HBM stripe path.  The legacy two-head assembly includes
    its historical final-context replay, intentionally, because calibration
    must reproduce the trace actually fed to Ramulator.
    """
    if phase not in ("full", "score", "context"):
        raise ValueError("phase must be full, score, or context")
    geometry = _geometry(run_length=run_length, num_ops_per_hbm=num_ops_per_hbm,
                         dbyte=dbyte, dhead=dhead, channel_count=channel_count,
                         channel_base=channel_base, trace_revision=trace_revision)
    q = max(1, int(shared_queries))
    columns_per_bank = geometry["columns_per_bank"]
    score_rows = geometry["score_rows"]
    score_windows = geometry["score_windows"]
    length = geometry["striped_length"]
    striped = geometry["striped"]
    groups = geometry["groups"]

    def one(valid: int):
        score_wrgb = 4 * columns_per_bank * valid
        score_mac = score_rows * columns_per_bank * valid
        score_mvsb = score_windows * 8 * valid
        sfm = valid
        ctx_mvgb = 8 * _ceil(length, 16 * (32 // int(dbyte))) * valid
        ctx_mac = columns_per_bank * score_rows * valid
        ctx_mvsb = columns_per_bank * 8 * valid
        return score_wrgb, score_mac, score_mvsb, sfm, ctx_mvgb, ctx_mac, ctx_mvsb

    stats = [one(valid) for valid in groups]
    sw = sm = ss = sf = cv = cm = cs = 0
    if striped or len(stats) % 2:
        # Pairs first; the final odd iteration takes the simple one-head path.
        pair_end = len(stats) if striped else len(stats) - 1
    else:
        pair_end = len(stats)
    for i in range(0, pair_end - 1, 2):
        a, b = stats[i], stats[i + 1]
        sw += a[0] + b[0]; sm += a[1] + b[1]; ss += a[2] + b[2]; sf += a[3] + b[3]
        cv += a[4] + b[4]; cm += 2 * a[5]; cs += 2 * a[6]
    if striped:
        # The generator's stripe path has exactly one single-head iteration.
        a = stats[0]
        sw += a[0]; sm += a[1]; ss += a[2]; sf += a[3]
        cv += a[4]; cm += a[5]; cs += a[6]
    elif len(stats) % 2:
        a = stats[-1]
        sw += a[0]; sm += a[1]; ss += a[2]; sf += a[3]
        cv += a[4]; cm += a[5]; cs += a[6]

    if phase == "score":
        cm = cv = cs = 0
    elif phase == "context":
        sw = sm = ss = sf = 0
    # Replicate expands every data command.  MQ shares only MAC_AB.
    non_mac_scale = q
    mac_scale = 1 if mq_command and q > 1 else q
    return (sm * mac_scale + cm * mac_scale, sf * non_mac_scale,
            cv * non_mac_scale, (ss + cs) * non_mac_scale, sw * non_mac_scale)


def _geometry(*, run_length: int, num_ops_per_hbm: int, dbyte: int, dhead: int,
              channel_count: int, channel_base, trace_revision: str) -> Dict:
    """Shared shape quantities of one generated trace (no fitted parameter).

    ``channel_base`` is accepted to mirror the caller's signature but is
    deliberately NOT read: the generator's ``--pool-base`` only renames which
    physical channels a run lands on, and neither the command counts nor the
    barrier/row structure depend on that.  Layer 1 and Layer 2 being exact on
    every cached channel base is the evidence.  It is spelled out here because
    an inert field in a split key silently defeats a held-out protocol.
    """
    del channel_base
    c = max(1, int(channel_count))
    heads = max(1, int(num_ops_per_hbm))
    lanes = 32 // int(dbyte)                       # n_mac
    columns_per_bank = _ceil(dhead, N_BANK * lanes)
    # ``chunkstripe1`` was appended when channel-base runs began passing
    # --head-hbm-stripe.  Older cache entries carry channel_base but used the
    # original head-per-channel stream, so revision is part of the geometry.
    striped = trace_revision == "chunkstripe1"
    if striped:
        length = _ceil(run_length, max(1, c // heads))
        groups = [c]
        num_itr = 1
    else:
        length = int(run_length)
        groups = [min(c, heads - start) for start in range(0, heads, c)]
        num_itr = len(groups)
    score_rows = _ceil(length, BANKS_IN_PARALLEL)
    return dict(striped=striped, striped_length=length, groups=groups,
                num_itr=num_itr, columns_per_bank=columns_per_bank,
                score_rows=score_rows,
                score_windows=_ceil(score_rows, 16),
                # Channels that genuinely run in parallel.  In the stripe
                # layout each channel owns a different token slice and its own
                # command bus; the legacy head-per-channel layout re-issues the
                # same all-bank command per channel from one trace stream, so
                # its replicas serialise.
                active_channels=c if striped else 1)


# ===========================================================================
# Layer 2: trace structure.  Exact; no fitted parameter.
# ===========================================================================
def trace_structure(*, run_length: int, num_ops_per_hbm: int, dbyte: int,
                    dhead: int, channel_count: int = 16, channel_base=None,
                    key_addr=None, value_addr=None, phase: str = "full",
                    trace_revision: str = "legacy") -> Dict[str, int]:
    """Barrier groups and DRAM row openings of the generated trace.

    Both are closed forms of the generator's command ORDER, and both are
    checked exactly against regenerated traces by ``calibrate.py``.

    * ``barrier_groups`` -- the generator emits one 16-command PIM_BARRIER
      block per synchronisation point.  Per head pair it emits
      ``1 + 2*score_windows + 2*columns_per_bank``; a trailing odd head (and
      the whole stripe path, which has exactly one iteration) emits
      ``1 + score_windows + 1 + columns_per_bank``.
    * ``row_openings`` -- both MAC phases sweep ``columns_per_bank*score_rows``
      consecutive 32-B columns from the K (score) / V (context) address.  With
      32 columns per DRAM row the number of rows a phase touches depends on
      the START column, which is why the K/V byte offset must be part of the
      model: two runs of identical length differ by a whole ACT/PRE when one
      of them straddles a row boundary.

    ``key_addr``/``value_addr`` accept either an absolute byte address or the
    wrapper's address-mapping tuple ``(ch, pch, rank, bg, ba, byte_in_row)``.
    """
    g = _geometry(run_length=run_length, num_ops_per_hbm=num_ops_per_hbm,
                  dbyte=dbyte, dhead=dhead, channel_count=channel_count,
                  channel_base=channel_base, trace_revision=trace_revision)
    windows = g["score_windows"]
    cpb = g["columns_per_bank"]
    num_itr = g["num_itr"]
    pairs = 0 if g["striped"] else num_itr // 2
    barriers = pairs * (1 + 2 * windows + 2 * cpb)
    if g["striped"] or num_itr % 2:
        barriers += 1 + windows + 1 + cpb

    columns = cpb * g["score_rows"]

    def rows_touched(address) -> int:
        start = (_row_byte_offset(address) % ROW_BYTES) // PREFETCH_BYTES
        return (start + columns - 1) // COLUMNS_PER_ROW + 1

    score_rows_opened = rows_touched(key_addr) if phase != "context" else 0
    ctx_rows_opened = rows_touched(value_addr) if phase != "score" else 0
    return dict(barrier_groups=barriers,
                row_openings=(score_rows_opened + ctx_rows_opened) * num_itr,
                num_itr=num_itr, score_windows=windows,
                columns_per_bank=cpb, active_channels=g["active_channels"])


def _row_byte_offset(address) -> int:
    """Byte offset inside the DRAM row, for either address representation."""
    if address is None:
        return 0
    if isinstance(address, (list, tuple)):
        # Wrapper mapping tuple (ch, pch, rank, bg, ba, byte_in_row).
        return int(address[-1])
    return int(address) % ROW_BYTES


# ===========================================================================
# Layer 3: cycle count.  Few physical parameters, fitted with a held-out split.
# ===========================================================================
def regime_key(meta: Mapping) -> str:
    """Physical execution regime -- NOT a per-configuration identity.

    Only two things change the shape of the command schedule itself: which
    trace assembly the generator used (two-head pipeline vs channel stripe)
    and whether a query batch replicates MAC_AB or shares it (MQ).  Every
    other field of a run enters the model through Layer-1/Layer-2 quantities,
    so an unseen channel, head count, batch size or length is interpolated by
    the physics instead of missing its bucket.
    """
    revision = meta.get("trace_revision") or "legacy"
    scheme = "mq" if meta.get("mq_command") else "replicate"
    return "{}|{}".format(revision, scheme)


def timing_features(counts: Sequence[int], structure: Mapping, *,
                    power_constraint: bool, nccdab_override=None,
                    **_ignored) -> List[float]:
    """Physical features, in datasheet units, for one run."""
    mac, sfm, mvgb, mvsb, wrgb = counts
    interval = command_interval(power_constraint, nccdab_override)
    lanes = max(1, int(structure["active_channels"]))
    return [mac / lanes * interval, wrgb / lanes, mvgb / lanes, mvsb / lanes,
            sfm / lanes, float(structure["row_openings"]),
            float(structure["barrier_groups"]), 1.0]


_COUNT_FIELDS = ("run_length", "num_ops_per_hbm", "dbyte", "dhead",
                 "channel_count", "channel_base", "shared_queries",
                 "mq_command", "phase", "trace_revision")
_STRUCT_FIELDS = ("run_length", "num_ops_per_hbm", "dbyte", "dhead",
                  "channel_count", "channel_base", "key_addr", "value_addr",
                  "phase", "trace_revision")


def features_from_meta(meta: Mapping) -> List[float]:
    """Layer-1 + Layer-2 -> Layer-3 features for one cached run signature."""
    counts = command_counts(**{k: meta[k] for k in _COUNT_FIELDS})
    structure = trace_structure(**{k: meta.get(k) for k in _STRUCT_FIELDS})
    return timing_features(counts, structure,
                           power_constraint=meta["power_constraint"],
                           nccdab_override=meta.get("nccdab_override"))


def _nnls(matrix, target, weight):
    """Non-negative weighted least squares (active set); no scipy on athena.

    Negative coefficients are physically meaningless here -- no command can
    shorten a schedule -- so they are dropped rather than fitted away.
    """
    import numpy as np
    a = np.asarray(matrix, float) * np.asarray(weight, float)[:, None]
    b = np.asarray(target, float) * np.asarray(weight, float)
    # Columns are scaled to unit norm before the drop test.  Without this the
    # "most negative coefficient" is a raw magnitude, so which feature gets
    # dropped depends on the units the feature happens to be measured in.
    scale = np.maximum(np.linalg.norm(a, axis=0), 1e-12)
    a_scaled = a / scale
    active = list(range(a.shape[1]))
    coefficients = np.zeros(a.shape[1])
    solution = None
    for _ in range(a.shape[1] + 1):
        if not active:
            raise ValueError("every feature was dropped; the design matrix is "
                             "degenerate for this regime")
        solution = np.linalg.lstsq(a_scaled[:, active], b, rcond=None)[0]
        if (solution >= -1e-9).all():
            break
        worst = active[int(np.argmin(solution))]
        active = [column for column in active if column != worst]
    if solution is None:                        # pragma: no cover - safety net
        raise ValueError("non-negative least squares did not converge")
    for index, column in enumerate(active):
        coefficients[column] = max(float(solution[index]) / scale[column], 0.0)
    return coefficients


MAC_FEATURE = FEATURE_NAMES.index("mac_x_interval")
CONST_FEATURE = FEATURE_NAMES.index("const")


def _fit_pinned(matrix, target, weight):
    """Fit with the MAC term pinned to its datasheet value and a free intercept.

    ``mac_x_interval`` is already in tCK -- it is ``mac * nCCDAB`` -- so its
    physically correct coefficient is exactly 1.  Letting the fit choose it
    looked harmless and was not: with every coefficient forced non-negative,
    the only way to give SHORT runs their fixed overhead was to raise the
    per-MAC slope, and the slope then over-charges the long runs.  Measured on
    ``chunkstripe1|replicate``: the free fit chose 1.0509, the MAC term is 84%
    of a long run's prediction, and the resulting +4.3% is essentially the
    whole +5% over-price seen at A1's operating point.

    So the slope is pinned and the intercept is freed instead.  A NEGATIVE
    intercept is the physically right shape for the correction it carries: the
    first MAC of a run waits for no predecessor, and the barrier and
    row-opening terms double-count part of that ramp-in.  Everything else
    stays a non-negative command cost.

    Cross-validated on distinct model inputs, this is better in all three
    regimes (chunkstripe1|replicate 3.71% -> 2.75%, legacy 6.08% -> 2.52%).
    """
    import numpy as np
    matrix = np.asarray(matrix, float)
    target = np.asarray(target, float)
    residual = target - matrix[:, MAC_FEATURE]
    free = [index for index in range(matrix.shape[1])
            if index not in (MAC_FEATURE, CONST_FEATURE)]
    def evaluate(intercept):
        partial = _nnls(matrix[:, free], np.maximum(residual - intercept, 0.0), weight)
        predicted = matrix[:, MAC_FEATURE] + matrix[:, free] @ partial + intercept
        error = np.abs(predicted - target) / np.maximum(1.0, target)
        return float(error.mean()), partial

    # Coarse-to-fine.  A single linear grid over the residual's own scale is
    # useless here: the regimes' cycle counts differ by four orders of
    # magnitude, so one step of that grid can be larger than the whole
    # correction being searched for.
    span = float(np.percentile(np.abs(residual), 90)) if len(residual) else 1.0
    low, high = -max(span, 1.0), 0.2 * max(span, 1.0)
    best = None
    for _ in range(6):
        candidates = np.linspace(low, high, 41)
        scored = [(evaluate(value), value) for value in candidates]
        (score, partial), intercept = min(scored, key=lambda item: item[0][0])
        if best is None or score < best[0]:
            best = (score, partial, float(intercept))
        step = (high - low) / 40.0
        low, high = intercept - 2 * step, intercept + 2 * step
    _, partial, intercept = best
    coefficients = np.zeros(matrix.shape[1])
    coefficients[MAC_FEATURE] = 1.0
    for slot, index in enumerate(free):
        coefficients[index] = float(partial[slot])
    coefficients[CONST_FEATURE] = intercept
    return coefficients


def _errors(matrix, coefficients, stretch, truth):
    import numpy as np
    predicted = (np.asarray(matrix, float) @ coefficients) / stretch
    truth = np.asarray(truth, float)
    return np.abs(predicted - truth) / np.maximum(1.0, truth)


def _summary(ape, predicted=None, truth=None) -> Dict[str, float]:
    """Relative error PLUS absolute error and signed bias.

    MAPE alone is the metric the fit optimises, so it flatters itself.  MAE
    says how wrong the model is in cycles, and the aggregate ratio says
    whether the errors cancel -- they do not: a workload sums many runs, so a
    correlated +3% bias survives averaging while MAPE hides its sign.
    """
    import numpy as np
    ape = np.asarray(ape, float)
    out = {"mape": float(ape.mean()), "p95": float(np.percentile(ape, 95)),
           "max": float(ape.max()), "n": int(ape.size)}
    if predicted is not None and truth is not None:
        predicted = np.asarray(predicted, float)
        truth = np.asarray(truth, float)
        out["mae_cycles"] = float(np.abs(predicted - truth).mean())
        out["aggregate_ratio"] = float(predicted.sum() / max(1.0, truth.sum()))
        out["median_ratio"] = float(np.median(predicted / np.maximum(1.0, truth)))
    return out


def _domain(matrix, samples) -> Dict[str, object]:
    """Calibrated envelope, used to flag extrapolation at estimate time.

    The box is axis-aligned, so a run sitting in an unsampled INTERIOR hole
    passes it.  ``run_length_largest_interior_gap`` records how large such a
    hole is, so a reader can see what "inside the domain" is worth.
    """
    import numpy as np
    matrix = np.asarray(matrix, float)
    lengths = sorted({int(meta["run_length"]) for _, meta in samples})
    gaps = [b - a for a, b in zip(lengths, lengths[1:])]
    domain = {"run_length": [lengths[0], lengths[-1]],
              "run_length_sampled_points": len(lengths),
              "run_length_largest_interior_gap": max(gaps) if gaps else 0}
    for index, name in enumerate(FEATURE_NAMES):
        if name == "const":
            continue
        domain[name] = [float(matrix[:, index].min()), float(matrix[:, index].max())]
    return domain


def fit_regime(samples: Sequence[Tuple[Sequence[int], Mapping]],
               validation_mask: Sequence[bool]) -> Dict[str, object]:
    """Fit one regime's Layer-3 parameters on the TRAIN split only.

    ``cycle = features . coefficients``, non-negative weighted least squares.

    There is no refresh stretch term.  ``cycle = X.c / (1 - t/nREFI)`` is
    exactly unidentifiable here: both the weighted-relative-error objective
    and the non-negativity constraint are invariant under ``c -> s*c``, so
    every value of ``t`` yields the same prediction (measured spread across
    the whole grid: <1e-8 cycles).  The earlier grid search was a no-op that
    made the mac coefficient look gauge-dependent.  Refresh is absorbed into
    the mac term, where it belongs -- it scales with schedule length.

    The fit minimises WEIGHTED RELATIVE error, which is the same quantity the
    report quotes as MAPE.  That is circular unless stated, so mean absolute
    error in cycles is reported next to it: an unweighted fit of the same
    features gets a much better MAE and a much worse MAPE.
    """
    import numpy as np
    matrix = np.asarray([features_from_meta(meta) for _, meta in samples], float)
    truth = np.asarray([result[0] for result, _ in samples], float)
    mask = np.asarray(validation_mask, bool)
    train = ~mask
    if int(train.sum()) < len(FEATURE_NAMES) + 1:
        raise ValueError("regime has too few training samples to fit")
    weight = 1.0 / np.maximum(1.0, truth[train])
    coefficients = _fit_pinned(matrix[train], truth[train], weight)
    used = [name for name, value in zip(FEATURE_NAMES, coefficients) if value]
    report = {
        "coefficients": [float(c) for c in coefficients],
        "features": list(FEATURE_NAMES),
        "effective_parameters": len(used),
        "features_used": used,
        "n_train": int(train.sum()),
        "n_validation": int(mask.sum()),
        "train": _summary(_errors(matrix[train], coefficients, 1.0, truth[train]),
                          matrix[train] @ coefficients, truth[train]),
        "domain": _domain(matrix[train], [s for s, m in zip(samples, mask) if not m]),
    }
    if mask.any():
        report["validation"] = _summary(
            _errors(matrix[mask], coefficients, 1.0, truth[mask]),
            matrix[mask] @ coefficients, truth[mask])
    return report


def fit_timing(samples: Iterable[Tuple[Sequence[int], Mapping]],
               validation: Optional[Mapping[int, bool]] = None
               ) -> Dict[str, object]:
    """Fit every regime present in ``samples``.

    ``validation`` maps a sample index to True when it belongs to the held-out
    split.  Callers must supply one: a model fitted on 100% of the data has no
    honest error estimate, and reporting its training error as accuracy is the
    exact failure this module was rewritten to remove.
    """
    samples = list(samples)
    if validation is None:
        raise ValueError("fit_timing requires an explicit validation split")
    grouped: Dict[str, List[int]] = {}
    for index, (_, meta) in enumerate(samples):
        grouped.setdefault(regime_key(meta), []).append(index)
    regimes: Dict[str, Dict] = {}
    for key, indices in grouped.items():
        subset = [samples[i] for i in indices]
        mask = [bool(validation.get(i, False)) for i in indices]
        if len(subset) - sum(mask) < len(FEATURE_NAMES) + 1:
            regimes[key] = {"insufficient_data": True, "n": len(subset)}
            continue
        regimes[key] = fit_regime(subset, mask)
    return {"version": 2, "features": list(FEATURE_NAMES), "regimes": regimes}


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
class UncalibratedRegime(RuntimeError):
    """Raised when a run has no fitted regime and no diagnostics sink."""


def _bootstrap_cycle(counts, structure, power_constraint, nccdab_override) -> int:
    """Datasheet-only fallback, used ONLY when explicitly allowed.

    Its measured error against the cache is ~150% MAPE with a 19x worst case,
    so it is never a silent default: it exists so a caller that wants a rough
    number can ask for one, and every use is counted in ``diagnostics``.
    """
    mac, sfm, mvgb, mvsb, wrgb = counts
    lanes = max(1, int(structure["active_channels"]))
    interval = command_interval(power_constraint, nccdab_override)
    return max(1, int(mac / lanes * interval + (wrgb + mvgb + mvsb + sfm) / lanes
                      + structure["row_openings"] * 16))


def estimate(*, pim_type, run_length: int, num_ops_per_hbm: int, dbyte: int,
             power_constraint: bool, dhead: int, num_hbm: int = 5,
             channel_count: int = 16, shared_kv: bool = False,
             shared_queries: int = 1, channel_base=None, mq_command: bool = False,
             nccdab_override=None, key_addr=None, value_addr=None,
             phase: str = "full", trace_revision: str = "legacy",
             timing_models: Optional[Mapping] = None,
             diagnostics: Optional[dict] = None,
             allow_uncalibrated: bool = False) -> Result:
    """Return a Ramulator-compatible ``[cycle, mac, sfm, mvgb, mvsb, wrgb]``.

    ``diagnostics``, when given, is a mutable dict that accumulates
    ``estimates`` / ``uncalibrated`` / ``extrapolated`` counters plus the
    offending regimes and features.  Callers are expected to surface it: an
    estimate outside the calibrated domain is a result the reader must be told
    about, not one to discover later.
    """
    counts = command_counts(
        run_length=run_length, num_ops_per_hbm=num_ops_per_hbm, dbyte=dbyte,
        dhead=dhead, channel_count=channel_count, channel_base=channel_base,
        shared_queries=shared_queries, mq_command=mq_command, phase=phase,
        trace_revision=trace_revision)
    structure = trace_structure(
        run_length=run_length, num_ops_per_hbm=num_ops_per_hbm, dbyte=dbyte,
        dhead=dhead, channel_count=channel_count, channel_base=channel_base,
        key_addr=key_addr, value_addr=value_addr, phase=phase,
        trace_revision=trace_revision)
    mac, sfm, mvgb, mvsb, wrgb = counts

    key = regime_key({"trace_revision": trace_revision, "mq_command": mq_command})
    regimes = (timing_models or {}).get("regimes", {})
    model = regimes.get(key)
    if diagnostics is not None:
        diagnostics["estimates"] = diagnostics.get("estimates", 0) + 1

    if model is None or model.get("insufficient_data"):
        if diagnostics is None and not allow_uncalibrated:
            raise UncalibratedRegime(
                "no fitted timing regime for {!r}; calibrate it, or pass a "
                "diagnostics dict / allow_uncalibrated=True and accept the "
                "datasheet fallback (measured ~150% MAPE)".format(key))
        if diagnostics is not None:
            diagnostics["uncalibrated"] = diagnostics.get("uncalibrated", 0) + 1
            regimes_seen = diagnostics.setdefault("uncalibrated_regimes", {})
            regimes_seen[key] = regimes_seen.get(key, 0) + 1
        cycle = _bootstrap_cycle(counts, structure, power_constraint, nccdab_override)
        return (cycle, mac, sfm, mvgb, mvsb, wrgb)

    features = timing_features(counts, structure,
                               power_constraint=power_constraint,
                               nccdab_override=nccdab_override)
    coefficients = model["coefficients"]
    cycle = max(1, int(round(sum(f * c for f, c in zip(features, coefficients)))))

    outside = _outside_domain(model.get("domain", {}), features, run_length)
    if outside and diagnostics is not None:
        diagnostics["extrapolated"] = diagnostics.get("extrapolated", 0) + 1
        seen = diagnostics.setdefault("extrapolated_features", {})
        for name in outside:
            seen[name] = seen.get(name, 0) + 1
    return (cycle, mac, sfm, mvgb, mvsb, wrgb)


def _outside_domain(domain: Mapping, features: Sequence[float],
                    run_length: int) -> List[str]:
    """Names of the calibrated quantities this run sits outside of."""
    outside = []
    bounds = domain.get("run_length")
    if isinstance(bounds, (list, tuple)) and not bounds[0] <= run_length <= bounds[1]:
        outside.append("run_length")
    for index, name in enumerate(FEATURE_NAMES):
        bounds = domain.get(name)
        if isinstance(bounds, (list, tuple)) and not bounds[0] <= features[index] <= bounds[1]:
            outside.append(name)
    return outside


def validation_report(models: Mapping) -> Dict[str, Dict]:
    """Cross-validated error per regime, for a report or a test to assert on.

    The shipped coefficients are fitted on every input, so a regime carries no
    ``validation`` of its own; what travels is the cross-validated error of
    the procedure that produced them, plus how many distinct model inputs and
    how many effective (non-zero) parameters that fit actually had.
    """
    out = {}
    for key, regime in (models or {}).get("regimes", {}).items():
        cross = regime.get("cross_validated")
        if cross is None:
            out[key] = regime.get("validation", {"missing": True})
            continue
        out[key] = {"cross_validated": cross,
                    "distinct_model_inputs": regime.get("n_train"),
                    "effective_parameters": regime.get("effective_parameters"),
                    "run_length_largest_interior_gap":
                        (regime.get("domain") or {}).get("run_length_largest_interior_gap")}
    return out
