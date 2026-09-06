#!/usr/bin/env python3
"""Construct the user-requested controlled workloads; does not simulate."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def segment(name, length, role="doc"):
    return {"role": role, "sha": hashlib.sha256(name.encode()).hexdigest()[:16],
            "len": length}


def request(rid, tier, segs, lout, parent=None):
    return dict(id=rid, tier=tier, parent=parent, history_len=0,
                lout=lout, segs=segs)


def advance(previous, prefix, additions, rid, tier, lout):
    """Keep every previous segment at its original offset, then append."""
    segs = [dict(s, role="user" if s["role"] == "parent_out" else s["role"])
            for s in (previous["segs"] if previous else prefix)]
    if previous:
        segs.append(segment(previous["id"] + "-output", previous["lout"], "parent_out"))
    segs.extend(additions)
    return request(rid, tier, segs, lout, previous["id"] if previous else None)


def rolling(rounds=32, sessions=1, extra_background=0):
    """A continuing summarizer and two unrelated short-writing workers/session."""
    docs = [segment("rolling-document-%d" % r, 256) for r in range(rounds)]
    agents = [request("a0_corpus", 0, [segment("corpus-prefix",256,"sys")]+docs,1)]
    prior = {}
    for r in range(rounds):
        for g in range(sessions):
            # Repeated traffic is fixed in the input, common to every rung.
            for who, count in (("a", 2 + extra_background), ("b", 1), ("s", 1)):
                key = (g, who)
                rid = "g%02d_%s_t%03d" % (g, who, r)
                additions = ([dict(docs[r]), segment(rid+"-instruction",16,"user")]
                             if who == "s" else
                             [segment(rid+"-note%d" % j,16,"user") for j in range(count)])
                a = advance(prior.get(key), [segment("g%d-%s-prefix"%(g,who),16,"sys")],
                            additions, rid, r, 128 if who == "s" else 8)
                prior[key] = a
                agents.append(a)
    return {"meta": dict(format="v2-dag", kind="rolling summarizer control",
                         rounds=rounds,sessions=sessions,extra_background=extra_background,
                         measured_prefix="g", measured_role="s",block_tokens=256),
            "agents": agents}


def retrieval(period=8, selected=64, workers=8, rounds=2,
              owner_group=8, own=4, lout=128, fresh=False, corpus_size=None):
    """Periodic co-read documents; owners ingest separate bounded documents."""
    corpus = max(selected*8, (selected-1)*period+1) if corpus_size is None else corpus_size
    assert corpus >= (selected-1)*period+1
    docs = [segment("retrieval-document-%d" % j,256) for j in range(corpus)]
    agents=[]
    for j in range(0,corpus,owner_group):
        agents.append(request("a_owner%04d" % j,0,[dict(x) for x in docs[j:j+owner_group]],1))
    targets = list(range(0,selected*period,period))
    prior={}
    for r in range(rounds):
        for g in range(workers):
            rid="s%02d_t%03d"%(g,r)
            additions=([dict(docs[j]) for j in targets] if r==0 else [])
            additions += [segment(rid+"-question",own,"user")]
            a=advance(prior.get(g),[segment("summary-prefix",16,"sys")],additions,rid,r,lout)
            agents.append(a);prior[g]=a
    if fresh:
        for j,n in enumerate((2048,4096,8192)):
            agents.append(request("z_fresh%d"%n,0,[segment("fresh%d"%j,n,"user")],8))
    return {"meta":dict(format="v2-dag",kind="periodic co-read control",period=period,
                        selected=selected,workers=workers,rounds=rounds,
                        owner_group=owner_group,own=own,lout=lout,fresh=fresh,
                        selected_documents=targets,corpus=corpus,block_tokens=256),"agents":agents}


def fixtures():
    cases={}
    for r in (16,32,64,128):
        cases["D_rolling_r%d"%r]=rolling(r)
    cases["D_rolling_r64_background_plus1"]=rolling(64,extra_background=1)
    cases["D_rolling_r64_sessions7"]=rolling(64,sessions=7)
    cases["D_rolling_r64_sessions8"]=rolling(64,sessions=8)
    for p in (1,4,8):
        cases["E_coread_stride%d"%p]=retrieval(period=p)
    cases["E_coread_stride8_monolithic_owner"]=retrieval(owner_group=512)
    cases["P_reuse_q4"]=retrieval(period=1,rounds=32,lout=8,corpus_size=64)
    cases["P_mixed_q4_fresh"]=retrieval(period=1,rounds=32,lout=8,fresh=True,corpus_size=64)
    return cases


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir",type=Path,default=HERE/"inputs")
    a=p.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    manifest={}
    for name,data in fixtures().items():
        raw=(json.dumps(data,indent=1)+"\n").encode()
        (a.outdir/(name+".json")).write_bytes(raw)
        manifest[name]={"sha256":hashlib.sha256(raw).hexdigest(),"requests":len(data["agents"]),"meta":data["meta"]}
    (a.outdir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Constructed %d workloads; no simulation"%len(manifest))
