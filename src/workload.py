"""Workload input validation and reuse planning for AttAcc.

The original simulator accepts one rectangular request shape (batch, Lin,
Lout).  Real service workloads are not rectangular: RAG samples are
independent requests whose segments can share a content fingerprint, and
supervisor workloads are request DAGs.  This module is deliberately kept
separate from the analytic hardware model so that accepting a workload never
changes the legacy command-line result.

Two input formats are supported without dropping source fields:

* legacy RAG list: ``sample``, ``seg_lens``, ``seg_sha``, ``seg_role``,
  ``L`` and ``lout``;
* v2-dag supervisor object: ``meta`` plus ``agents`` and their ``segs``.

``cacheblend`` and ``epic`` are reuse *policies*, not workload types.  The
planner intentionally exposes the rows which a policy may reuse/recompute;
the cycle model must consume this plan rather than infer policy from an
illustrative CSV trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


class WorkloadValidationError(ValueError):
    """Raised when a workload cannot describe a valid inference request."""


VALID_REUSE_POLICIES = ("no-reuse", "cacheblend", "epic", "recompute", "promptcache",
                        "cachecraft", "cachetune")
# PAPER TIE (software upstream): the paper's placement claims must hold
# under ANY reasonable reuse software, so the upstream is an axis, not a
# constant.  Experiment ruling (2026-08-25): the EXPERIMENT matrix uses only
# the guaranteed-recompute selection family -- every member recomputes some
# tokens and members differ ONLY in which tokens they select (cacheblend =
# deviation-sampled ratio; epic = the "first k tokens of each shifted
# segment" special case; cachecraft = overlap-scaled prefix; cachetune =
# offline-selected ratio).  promptcache (zero recompute) stays implemented
# as the endpoint baseline but is EXCLUDED from the matrix.
# Policy families (software-upstream enrichment, 2026-08-24): members share
# the anchor policy's plan machinery and differ in the recompute-selection
# rule, mirroring the published chunk-reuse family:
# - cacheblend family: ratio-sampled recompute rows in designated layers.
#   "cachetune" (arXiv:2605.24022-style) selects the rows OFFLINE, so it has
#   no full-recompute selection layers and pays no online selection pass.
# - epic family: per-segment leading-prefix recompute at chunk boundaries.
#   "promptcache" (MLSys'24) is the zero-recompute endpoint; "cachecraft"
#   (SIGMOD'25-style) sizes the prefix per chunk from the context overlap
#   between consumer and owner (knob: cachecraft_alpha).
CACHEBLEND_FAMILY = ("cacheblend", "cachetune")
# "recompute" (2026-08-27): general count-based policy, k random rows per
# shifted chunk; structurally EPIC-family (per-segment corrected rows).
EPIC_FAMILY = ("epic", "recompute", "promptcache", "cachecraft")
_LEGACY_REQUIRED = ("sample", "seg_lens", "seg_sha", "seg_role", "L", "lout")


@dataclass(frozen=True)
class Segment:
    role: str
    fingerprint: str
    length: int
    source: str = "online"
    position_delta: int = 0
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class Request:
    request_id: str
    tier: int
    parent_id: Optional[str]
    lout: int
    segments: Tuple[Segment, ...]
    total_length: int
    # KV rows this agent already holds from its own earlier turns.  They are
    # never recomputed -- prefill and decode only attend over them -- and they
    # are not part of ``total_length`` or any segment.
    history_len: int = 0
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class Workload:
    kind: str  # ``rag`` or ``supervisor``
    requests: Tuple[Request, ...]
    raw: Any = field(compare=False)

    @property
    def tiers(self) -> Dict[int, Tuple[Request, ...]]:
        result: Dict[int, List[Request]] = {}
        for request in self.requests:
            result.setdefault(request.tier, []).append(request)
        return {tier: tuple(result[tier]) for tier in sorted(result)}


@dataclass(frozen=True)
class ReuseDecision:
    request_id: str
    segment_index: int
    fingerprint: str
    length: int
    owner_request_id: str
    owner_tier: int
    epic_prefix_rows: Tuple[int, ...]


@dataclass(frozen=True)
class ReuseConfig:
    """Policy controls independent of workload kind and model dimensions."""

    policy: str
    cacheblend_full_recompute_layers: Tuple[int, ...] = ()
    cacheblend_partial_recompute_layers: Tuple[int, ...] = ()
    cacheblend_recompute_ratio: float = 0.0
    epic_prefix_recompute_tokens: int = 1
    random_seed: int = 0
    # recompute policy only (ruling chenyi9 2026-08-27, option 1): when the
    # serving layout is position-INSENSITIVE (masked/diff-pool layouts and
    # the GPU-gather A2 -- everything except the maskless naive A3), the k
    # recomputed rows are placed CANONICALLY at the chunk head: the cost
    # structure is identical (a masked row is streamed-and-dropped wherever
    # it sits; a diff row lives in its own pool either way) while the
    # physical scan shapes collapse back onto the shared signature cache.
    # Only A3, whose run splits physically depend on the positions, keeps
    # the true random draw.
    recompute_canonical: bool = False
    # cachecraft only: per-chunk recompute prefix = ceil(alpha * (1 -
    # context overlap) * chunk length), at least one row when shifted.
    cachecraft_alpha: float = 0.05


@dataclass(frozen=True)
class ReusePlan:
    config: ReuseConfig
    reusable: Tuple[ReuseDecision, ...]
    fresh_tokens: int
    reused_tokens: int
    # {layer: {request: {segment index: relative reused-token rows}}}
    cacheblend_partial_rows: Mapping[int, Mapping[str, Mapping[int, Tuple[int, ...]]]] = field(
        default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.config.policy,
            "cacheblend_full_recompute_layers": list(
                self.config.cacheblend_full_recompute_layers),
            "cacheblend_partial_recompute_layers": list(
                self.config.cacheblend_partial_recompute_layers),
            "cacheblend_recompute_ratio": self.config.cacheblend_recompute_ratio,
            "epic_prefix_recompute_tokens": self.config.epic_prefix_recompute_tokens,
            "recompute_canonical": self.config.recompute_canonical,
            "cachecraft_alpha": self.config.cachecraft_alpha,
            "random_seed": self.config.random_seed,
            "fresh_tokens": self.fresh_tokens,
            "reused_tokens": self.reused_tokens,
            "reusable_segments": [
                {
                    "request": d.request_id,
                    "segment_index": d.segment_index,
                    "fingerprint": d.fingerprint,
                    "length": d.length,
                    "cache_owner": d.owner_request_id,
                    "owner_tier": d.owner_tier,
                    "epic_prefix_rows": list(d.epic_prefix_rows),
                }
                for d in self.reusable
            ],
            "cacheblend_partial_rows": {
                str(layer): {
                    request: {str(index): list(rows) for index, rows in segments.items()}
                    for request, segments in requests.items()
                }
                for layer, requests in self.cacheblend_partial_rows.items()
            },
        }


def _error(where: str, message: str) -> WorkloadValidationError:
    return WorkloadValidationError("{}: {}".format(where, message))


def _positive_int(value: Any, where: str, *, allow_zero: bool = False) -> int:
    # bool is an int subclass, but is never a valid token count or tier.
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(where, "must be an integer")
    if value < 0 or (not allow_zero and value == 0):
        comparator = "non-negative" if allow_zero else "positive"
        raise _error(where, "must be {}".format(comparator))
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(where, "must be a non-empty string")
    return value


def _int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(where, "must be an integer")
    return value


def _segments_from_legacy(sample: Mapping[str, Any], where: str) -> Tuple[Segment, ...]:
    for name in _LEGACY_REQUIRED:
        if name not in sample:
            raise _error(where, "missing required field '{}'".format(name))
    lengths, hashes, roles = sample["seg_lens"], sample["seg_sha"], sample["seg_role"]
    if not all(isinstance(value, list) for value in (lengths, hashes, roles)):
        raise _error(where, "seg_lens, seg_sha and seg_role must be lists")
    if not lengths or len(lengths) != len(hashes) or len(lengths) != len(roles):
        raise _error(where, "segment lists must be non-empty and have equal length")
    segments = tuple(
        Segment(
            role=_string(role, "{}.seg_role[{}]".format(where, index)),
            fingerprint=_string(fingerprint, "{}.seg_sha[{}]".format(where, index)),
            length=_positive_int(length, "{}.seg_lens[{}]".format(where, index)),
            raw={"role": role, "sha": fingerprint, "len": length},
        )
        for index, (length, fingerprint, role) in enumerate(zip(lengths, hashes, roles))
    )
    roles = [segment.role for segment in segments]
    allowed_roles = {"sys", "doc", "query"}
    unknown_roles = set(roles) - allowed_roles
    if unknown_roles:
        raise _error(where, "unknown RAG segment roles: {}".format(
            sorted(unknown_roles)))
    if roles[0] != "sys" or roles[-1] != "query":
        raise _error(where, "RAG segments must begin with sys and end with query")
    if roles.count("sys") != 1 or roles.count("query") != 1 or "doc" not in roles:
        raise _error(where, "RAG segments must contain one sys, one or more docs, and one query")
    if any(role != "doc" for role in roles[1:-1]):
        raise _error(where, "only doc segments may appear between sys and query")
    return segments


def _parse_rag(data: List[Any]) -> Workload:
    requests: List[Request] = []
    ids = set()
    for index, sample in enumerate(data):
        where = "samples[{}]".format(index)
        if not isinstance(sample, Mapping):
            raise _error(where, "must be an object")
        segments = _segments_from_legacy(sample, where)
        sample_id = _string(str(sample["sample"]), where + ".sample")
        if sample_id in ids:
            raise _error(where, "duplicate sample id '{}'".format(sample_id))
        ids.add(sample_id)
        total = _positive_int(sample["L"], where + ".L")
        actual = sum(segment.length for segment in segments)
        if actual != total:
            raise _error(where, "L={} but segment lengths sum to {}".format(total, actual))
        history = _positive_int(sample.get("history_len", 0),
                                where + ".history_len", allow_zero=True)
        requests.append(Request(sample_id, 0, None,
                                _positive_int(sample["lout"], where + ".lout"),
                                segments, total, history, sample))
    if not requests:
        raise _error("samples", "must not be empty")
    return Workload("rag", tuple(requests), data)


def _parse_supervisor(data: Mapping[str, Any]) -> Workload:
    if not isinstance(data.get("agents"), list) or not data["agents"]:
        raise _error("agents", "must be a non-empty list")
    requests: List[Request] = []
    ids = set()
    for index, agent in enumerate(data["agents"]):
        where = "agents[{}]".format(index)
        if not isinstance(agent, Mapping):
            raise _error(where, "must be an object")
        request_id = _string(agent.get("id"), where + ".id")
        if request_id in ids:
            raise _error(where, "duplicate agent id '{}'".format(request_id))
        ids.add(request_id)
        tier = _positive_int(agent.get("tier"), where + ".tier", allow_zero=True)
        lout = _positive_int(agent.get("lout"), where + ".lout")
        parent = agent.get("parent")
        if parent is not None:
            parent = _string(parent, where + ".parent")
        segs = agent.get("segs")
        if not isinstance(segs, list) or not segs:
            raise _error(where + ".segs", "must be a non-empty list")
        segments: List[Segment] = []
        for seg_index, segment in enumerate(segs):
            seg_where = "{}.segs[{}]".format(where, seg_index)
            if not isinstance(segment, Mapping):
                raise _error(seg_where, "must be an object")
            segments.append(Segment(
                role=_string(segment.get("role"), seg_where + ".role"),
                fingerprint=_string(segment.get("sha"), seg_where + ".sha"),
                length=_positive_int(segment.get("len"), seg_where + ".len"),
                source=str(segment.get("src", "online")),
                position_delta=_int(segment.get("delta", 0), seg_where + ".delta"),
                raw=segment,
            ))
        history = _positive_int(agent.get("history_len", 0),
                                where + ".history_len", allow_zero=True)
        requests.append(Request(request_id, tier, parent, lout, tuple(segments),
                                sum(segment.length for segment in segments),
                                history, agent))

    by_id = {request.request_id: request for request in requests}
    for request in requests:
        if request.parent_id is None:
            if request.tier != 0:
                raise _error("agent '{}'".format(request.request_id),
                             "root agents must be in tier 0")
            if any(segment.role == "parent_out" for segment in request.segments):
                raise _error("agent '{}'".format(request.request_id),
                             "root agents cannot contain parent_out")
            continue
        if request.parent_id not in by_id:
            raise _error("agent '{}'".format(request.request_id),
                         "parent '{}' does not exist".format(request.parent_id))
        parent = by_id[request.parent_id]
        if parent.tier >= request.tier:
            raise _error("agent '{}'".format(request.request_id),
                         "parent must belong to an earlier tier")
        parent_segments = [segment for segment in request.segments
                           if segment.role == "parent_out"]
        if len(parent_segments) != 1:
            raise _error("agent '{}'".format(request.request_id),
                         "must contain exactly one parent_out segment")
        if parent_segments[0].length != parent.lout:
            raise _error("agent '{}'".format(request.request_id),
                         "parent_out length {} does not match parent lout {}".format(
                             parent_segments[0].length, parent.lout))
    return Workload("supervisor", tuple(requests), data)


def load_workload(path: str | Path) -> Workload:
    """Load and fully validate a legacy RAG or v2-dag supervisor workload."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise WorkloadValidationError("cannot read {}: {}".format(source, exc)) from exc
    except json.JSONDecodeError as exc:
        raise WorkloadValidationError("invalid JSON in {}: {}".format(source, exc)) from exc
    if isinstance(data, list):
        return _parse_rag(data)
    if isinstance(data, Mapping) and "agents" in data:
        return _parse_supervisor(data)
    raise WorkloadValidationError(
        "workload root must be a legacy RAG list or a v2-dag object with 'agents'")


def _layer_tuple(layers: Iterable[int], field_name: str) -> Tuple[int, ...]:
    try:
        values = tuple(layers)
    except TypeError as exc:
        raise WorkloadValidationError("{} must be an iterable of layer indices".format(
            field_name)) from exc
    if any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
           for layer in values):
        raise WorkloadValidationError(
            "{} must contain non-negative integer layer indices".format(field_name))
    if len(set(values)) != len(values):
        raise WorkloadValidationError("{} must not contain duplicate layers".format(field_name))
    return tuple(sorted(values))


def _sample_cacheblend_rows(decisions: Iterable[ReuseDecision], ratio: float,
                            rng: random.Random) -> Dict[Tuple[str, int], Tuple[int, ...]]:
    """Choose exactly ceil(ratio*N) positions, uniformly without replacement."""
    slots: List[Tuple[str, int, int]] = []
    for decision in decisions:
        slots.extend((decision.request_id, decision.segment_index, row)
                     for row in range(decision.length))
    count = math.ceil(len(slots) * ratio)
    selected = rng.sample(slots, count)
    rows: Dict[Tuple[str, int], List[int]] = {}
    for request_id, segment_index, row in selected:
        rows.setdefault((request_id, segment_index), []).append(row)
    return {key: tuple(value) for key, value in rows.items()}


def build_reuse_plan(workload: Workload,
                     policy: str,
                     cacheblend_recompute_ratio: float = 0.0,
                     random_seed: int = 0,
                     cacheblend_full_recompute_layers: Iterable[int] = (),
                     cacheblend_partial_recompute_layers: Iterable[int] = (),
                     epic_prefix_recompute_tokens: int = 1,
                     cachecraft_alpha: float = 0.05,
                     recompute_canonical: bool = False) -> ReusePlan:
    """Return an auditable policy plan without altering legacy simulation.

    Cache ownership is selected deterministically by ``(tier, request id,
    segment index)``.  A same-tier match is labelled as a shared-cache
    candidate, rather than an execution dependency: it does not change the
    supervisor DAG.  A scheduler may materialize this cache before dispatching
    that tier.
    CacheBlend uses an explicit full-layer set, partial-layer set and
    recompute ratio.  For *each partial layer* and request, exactly the target
    number of reusable rows is sampled uniformly without replacement.  EPIC
    instead selects a stable prefix of every reused segment; the simulator can
    apply that same prefix in all model layers.
    """
    if policy not in VALID_REUSE_POLICIES:
        raise WorkloadValidationError(
            "unknown reuse policy '{}'; choose one of {}".format(policy, VALID_REUSE_POLICIES))
    if (not isinstance(cacheblend_recompute_ratio, (int, float)) or
            isinstance(cacheblend_recompute_ratio, bool) or
            not 0 <= cacheblend_recompute_ratio <= 1):
        raise WorkloadValidationError("cacheblend recompute ratio must be in [0, 1]")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise WorkloadValidationError("random seed must be an integer")
    if (isinstance(epic_prefix_recompute_tokens, bool) or
            not isinstance(epic_prefix_recompute_tokens, int) or
            epic_prefix_recompute_tokens < 0):
        raise WorkloadValidationError("EPIC prefix recompute tokens must be non-negative")
    if (not isinstance(cachecraft_alpha, (int, float)) or
            isinstance(cachecraft_alpha, bool) or not 0 <= cachecraft_alpha <= 1):
        raise WorkloadValidationError("cachecraft alpha must be in [0, 1]")
    full_layers = _layer_tuple(cacheblend_full_recompute_layers,
                               "CacheBlend full recompute layers")
    partial_layers = _layer_tuple(cacheblend_partial_recompute_layers,
                                  "CacheBlend partial recompute layers")
    overlap = set(full_layers).intersection(partial_layers)
    if overlap:
        raise WorkloadValidationError(
            "CacheBlend full and partial recompute layers overlap: {}".format(
                sorted(overlap)))
    if policy == "cachetune" and full_layers:
        raise WorkloadValidationError(
            "cachetune selects recompute rows offline; it has no "
            "full-recompute selection layers")
    config = ReuseConfig(policy, full_layers, partial_layers,
                         float(cacheblend_recompute_ratio),
                         epic_prefix_recompute_tokens, random_seed,
                         cachecraft_alpha=float(cachecraft_alpha),
                         recompute_canonical=bool(recompute_canonical))
    rng = random.Random(random_seed)
    owners: Dict[str, Tuple[Request, int]] = {}
    by_request_id = {request.request_id: request for request in workload.requests}
    segment_offsets = {
        (request.request_id, index): sum(segment.length for segment in request.segments[:index])
        for request in workload.requests
        for index in range(len(request.segments))
    }
    decisions: List[ReuseDecision] = []
    total = sum(request.total_length for request in workload.requests)
    if policy == "no-reuse":
        return ReusePlan(config, (), total, 0)

    for request in sorted(workload.requests, key=lambda item: (item.tier, item.request_id)):
        for index, segment in enumerate(request.segments):
            # A relay segment names a concrete data dependency.  Its producer
            # must remain the declared parent even if a same-tier request has
            # the same output fingerprint.
            if segment.role == "parent_out" and request.parent_id is not None:
                owner = (by_request_id[request.parent_id], -1)
            else:
                owner = owners.get(segment.fingerprint)
            if owner is None:
                owners[segment.fingerprint] = (request, index)
                continue
            owner_request, owner_index = owner
            # EPIC's static AttnLink correction is needed at a context boundary
            # (or after a position shift), not for an unchanged prefix segment.
            shifted = (segment.role == "parent_out" or
                       segment.position_delta != 0 or
                       (owner_index >= 0 and
                        segment_offsets[(request.request_id, index)] !=
                        segment_offsets[(owner_request.request_id, owner_index)]))
            if policy == "epic" and shifted:
                correction = tuple(range(min(segment.length,
                                             epic_prefix_recompute_tokens)))
            elif policy == "recompute" and shifted:
                # General count-based recompute (chenyi9 2026-08-27, its own
                # policy -- the existing branches stay untouched): k tokens
                # drawn UNIFORMLY AT RANDOM inside the shifted chunk (count
                # shares the epic_prefix_recompute_tokens knob).  Upstream
                # only the COUNT matters (C0 ruling); the in-chunk positions
                # physically matter only to a maskless layout (A3 splits its
                # master run at every recomputed row).  Deterministic via
                # the plan's random seed.
                count = min(segment.length, epic_prefix_recompute_tokens)
                if recompute_canonical:
                    correction = tuple(range(count))
                else:
                    correction = tuple(sorted(rng.sample(range(segment.length),
                                                         count)))
            elif policy == "cachecraft" and shifted:
                # Cache-Craft-style variable prefix: the less of the chunk's
                # original context the consumer preserves, the more boundary
                # rows are recomputed (Jaccard overlap of the preceding
                # fingerprint sets; a relay parent has no visible context ->
                # overlap 0).
                consumer_prec = {seg.fingerprint
                                 for seg in request.segments[:index]}
                owner_prec = ({seg.fingerprint for seg in
                               owner_request.segments[:owner_index]}
                              if owner_index >= 0 else set())
                if consumer_prec or owner_prec:
                    union = consumer_prec | owner_prec
                    overlap_frac = len(consumer_prec & owner_prec) / len(union)
                else:
                    overlap_frac = 1.0
                rows = min(segment.length,
                           max(1, math.ceil(cachecraft_alpha *
                                            (1.0 - overlap_frac) *
                                            segment.length)))
                correction = tuple(range(rows))
            else:
                # promptcache reuses the chunk verbatim (zero recompute);
                # unshifted segments need no boundary fix in any policy.
                correction = ()
            decisions.append(ReuseDecision(request.request_id, index,
                                           segment.fingerprint, segment.length,
                                           owner_request.request_id,
                                           owner_request.tier, correction))

    reused = sum(decision.length for decision in decisions)
    partial_rows: Dict[int, Dict[str, Dict[int, Tuple[int, ...]]]] = {}
    if policy in CACHEBLEND_FAMILY:
        for layer in partial_layers:
            by_request: Dict[str, Dict[int, Tuple[int, ...]]] = {}
            decisions_by_request: Dict[str, List[ReuseDecision]] = {}
            for decision in decisions:
                decisions_by_request.setdefault(decision.request_id, []).append(decision)
            for request_id, request_decisions in decisions_by_request.items():
                selected = _sample_cacheblend_rows(request_decisions,
                                                    cacheblend_recompute_ratio, rng)
                for (_, segment_index), rows in selected.items():
                    by_request.setdefault(request_id, {})[segment_index] = rows
            partial_rows[layer] = by_request
    return ReusePlan(config, tuple(decisions), total - reused, reused, partial_rows)


def validate_reuse_plan(workload: Workload, plan: ReusePlan,
                        model_layers: Optional[int] = None) -> None:
    """Check the layer, shape and recompute-row invariants before execution."""
    if plan.config.policy not in VALID_REUSE_POLICIES:
        raise WorkloadValidationError("reuse plan has an unknown policy")
    requests = {request.request_id: request for request in workload.requests}
    if plan.config.policy == "no-reuse":
        if plan.reusable or plan.cacheblend_partial_rows:
            raise WorkloadValidationError("no-reuse plan must not contain reuse rows")
        return
    decisions_by_request: Dict[str, Dict[int, ReuseDecision]] = {}
    for decision in plan.reusable:
        request = requests.get(decision.request_id)
        if request is None or not 0 <= decision.segment_index < len(request.segments):
            raise WorkloadValidationError("reuse decision refers to an unknown segment")
        segment = request.segments[decision.segment_index]
        if segment.fingerprint != decision.fingerprint or segment.length != decision.length:
            raise WorkloadValidationError("reuse decision segment shape/fingerprint mismatch")
        if any(row < 0 or row >= decision.length for row in decision.epic_prefix_rows):
            raise WorkloadValidationError("EPIC recompute row is outside its segment")
        if plan.config.policy == "recompute":
            # The general count policy (2026-08-27) draws its rows uniformly
            # at random inside the chunk: sorted and unique, not a prefix.
            if decision.epic_prefix_rows != tuple(sorted(set(decision.epic_prefix_rows))):
                raise WorkloadValidationError(
                    "recompute rows must be sorted unique positions")
        else:
            expected_prefix = tuple(range(len(decision.epic_prefix_rows)))
            if decision.epic_prefix_rows != expected_prefix:
                raise WorkloadValidationError("EPIC recompute rows must be a leading prefix")
        decisions_by_request.setdefault(decision.request_id, {})[decision.segment_index] = decision

    if plan.config.policy in CACHEBLEND_FAMILY:
        full = set(plan.config.cacheblend_full_recompute_layers)
        partial = set(plan.config.cacheblend_partial_recompute_layers)
        if full.intersection(partial):
            raise WorkloadValidationError("CacheBlend full and partial layers overlap")
        if model_layers is not None:
            expected_layers = set(range(model_layers))
            if full.union(partial) != expected_layers:
                raise WorkloadValidationError("CacheBlend layers do not cover the model")
        if set(plan.cacheblend_partial_rows) != partial:
            raise WorkloadValidationError("CacheBlend partial-row layers differ from configuration")
        for layer, by_request in plan.cacheblend_partial_rows.items():
            for request_id, decisions in decisions_by_request.items():
                segment_rows = by_request.get(request_id, {})
                total_rows = sum(decision.length for decision in decisions.values())
                selected_rows = [row for rows in segment_rows.values() for row in rows]
                expected_count = math.ceil(total_rows * plan.config.cacheblend_recompute_ratio)
                if len(selected_rows) != expected_count:
                    raise WorkloadValidationError(
                        "CacheBlend layer {} request '{}' has {} recompute rows; expected {}".format(
                            layer, request_id, len(selected_rows), expected_count))
                for segment_index, rows in segment_rows.items():
                    decision = decisions.get(segment_index)
                    if decision is None or len(set(rows)) != len(rows):
                        raise WorkloadValidationError("CacheBlend rows refer to an invalid segment")
                    if any(row < 0 or row >= decision.length for row in rows):
                        raise WorkloadValidationError("CacheBlend recompute row is outside its segment")


def workload_summary(workload: Workload, plan: Optional[ReusePlan] = None) -> Dict[str, Any]:
    """Lossless-friendly summary for CLI validation and regression fixtures."""
    result: Dict[str, Any] = {
        "kind": workload.kind,
        "requests": len(workload.requests),
        "tiers": {
            str(tier): [request.request_id for request in requests]
            for tier, requests in workload.tiers.items()
        },
        "total_input_tokens": sum(request.total_length for request in workload.requests),
        "total_output_tokens": sum(request.lout for request in workload.requests),
        "total_history_tokens": sum(request.history_len for request in workload.requests),
    }
    if plan is not None:
        result["reuse"] = plan.to_dict()
    return result
