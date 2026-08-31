#!/usr/bin/env python3
"""Value-by-value identity check between two reference-matrix runs.

Strict: floats must be EQUAL, not close.  A change that is only supposed to
remove interpreter overhead must not move a single bit of the result, so any
tolerance here would defeat the purpose.

The one documented exclusion is the signature-cache statistics block: it counts
hits/misses/invocations of the Ramulator memoisation, which is run state, not a
simulation result.  Precedent: docs/sessions/2026-08-27.md night 10.
"""
import json, sys, os, glob

EXCLUDE_KEYS = {"ramulator_signature_cache"}


def walk(a, b, path, diffs, limit=40):
    if len(diffs) >= limit:
        return
    if type(a) is not type(b) and not (isinstance(a, (int, float))
                                       and isinstance(b, (int, float))):
        diffs.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
        return
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k in EXCLUDE_KEYS:
                continue
            if k not in a:
                diffs.append(f"{path}.{k}: missing in A")
            elif k not in b:
                diffs.append(f"{path}.{k}: missing in B")
            else:
                walk(a[k], b[k], f"{path}.{k}", diffs, limit)
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", diffs, limit)
    elif isinstance(a, float) or isinstance(b, float):
        # exact bit equality (repr round-trips through JSON losslessly)
        if a != b:
            diffs.append(f"{path}: {a!r} != {b!r}  (delta {b - a!r})"
                         if isinstance(a, float) and isinstance(b, float)
                         else f"{path}: {a!r} != {b!r}")
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")


def main():
    A, B = sys.argv[1], sys.argv[2]
    da, db = f"output/_verify/{A}", f"output/_verify/{B}"
    fa = sorted(os.path.basename(p) for p in glob.glob(f"{da}/*.json"))
    fb = sorted(os.path.basename(p) for p in glob.glob(f"{db}/*.json"))
    if fa != fb:
        print(f"FILE SET MISMATCH\n  only in {A}: {sorted(set(fa)-set(fb))}"
              f"\n  only in {B}: {sorted(set(fb)-set(fa))}")
        if not (set(fa) & set(fb)):
            print("VERDICT: cannot compare")
            return 1
    ok = bad = 0
    for name in sorted(set(fa) & set(fb)):
        try:
            # Fast path: identical data structures serialise to identical bytes,
            # so byte equality already proves bit-identity.  Only a mismatch
            # needs the (slow) recursive walk, which also applies the one
            # documented exclusion.
            ra = open(f"{da}/{name}", "rb").read()
            rb = open(f"{db}/{name}", "rb").read()
            if ra == rb:
                ok += 1
                continue
            a = json.loads(ra)
            b = json.loads(rb)
        except Exception as e:
            print(f"  {name}: UNREADABLE {e}")
            bad += 1
            continue
        diffs = []
        walk(a, b, "", diffs)
        if diffs:
            bad += 1
            print(f"  {name}: {len(diffs)} DIFF(S)")
            for d in diffs[:6]:
                print(f"      {d}")
        else:
            ok += 1
    print(f"\nidentical: {ok}   differing: {bad}   (of {len(set(fa) & set(fb))} compared)")
    print("VERDICT: BIT-IDENTICAL" if bad == 0 and fa == fb
          else "VERDICT: DIFFERENCES FOUND")
    return 0 if bad == 0 and fa == fb else 1


if __name__ == "__main__":
    sys.exit(main())
