#!/usr/bin/env python3
"""Turning points n_q* (A6/A4 = 1) per link x policy x model from the dense replay grid."""
import json, os, math, sys
import pathlib
GRID=sys.argv[1] if len(sys.argv)>1 else 'grid'
S=str(pathlib.Path(__file__).resolve().parent/'results'/GRID)+'/'
MODELS=('LLAMA-7B','LLAMA-65B','GPT-175B'); NDEC={'LLAMA-7B':32,'LLAMA-65B':80,'GPT-175B':96}
import glob, re
_qs=set()
for f in glob.glob(S+'*_replay_L8192_q*_A4_*.json'):
    m=re.search(r'_q(\d+)_A4_',f); _qs.add(int(m.group(1)))
QS=tuple(sorted(_qs))
TAG={'cacheblend':'cacheblendcacheblendrecomputeratio0','epic':'epicepicprefixrecomputetokens1'}
def ld(model,q,cfg,link,pol):
    f=f"{S}{model}_replay_L8192_q{q}_{cfg}_{link}_{TAG[pol]}.json"
    return json.load(open(f)) if os.path.exists(f) else None
def nq_of(d,model,pol):
    t=d['tiers'][1]; eff=t['effective_prefill_tokens']; padded=t['padded_prefill_tokens']; n=NDEC[model]
    return (eff*n-padded*2)/(n-2) if pol=='cacheblend' else eff
out=[]
def emit(s=''): out.append(s); print(s)
for link,lname in (('nvlink3','NVLink3 (600 GB/s)'),('pcie4','PCIe4 (64 GB/s)')):
    emit(f"\n# Link: {lname}")
    for pol,pname in (('cacheblend','CacheBlend (r = 0 here, so n_q = n_new + n_off; for r > 0 the same n_q* applies with n_q = n_new + n_off + r·L)'),('epic','EPIC (prefix 1 token/segment, n_q = n_new + n_off + 1)')):
        emit(f"\n## {pname}\n")
        emit("| model | n_q* (simulated, A6/A4 = 1) | bracketing grid points (n_q: A6/A4) | theory n_q* = 4·(t_link + t_gpu_attn)/t_pass | per-layer terms at n_q<=20: t_pass / t_link / t_gpu_attn [us] |")
        emit("|---|---:|---|---:|---|")
        for model in MODELS:
            pts=[]
            for q in QS:
                a=ld(model,q,'A4',link,pol); b=ld(model,q,'A6',link,pol)
                if a and b:
                    pts.append((nq_of(b,model,pol), b['tiers'][1]['prefill_s']/a['tiers'][1]['prefill_s'], a, b))
            pts.sort()
            if not pts: emit(f"| {model} | (no data yet) | | | |"); continue
            cross=None; brack='-'
            for (n0,r0,_,_),(n1,r1,_,_) in zip(pts,pts[1:]):
                if r0<=1.0<r1 or r0<1.0<=r1:
                    cross=n0+(1.0-r0)*(n1-n0)/(r1-r0); brack=f"{n0:.0f}: {r0:.2f} -> {n1:.0f}: {r1:.2f}"; break
            if cross is None:
                if pts[0][1]>1: cross_s=f"< {pts[0][0]:.0f} (A6/A4 {pts[0][1]:.2f} already > 1)"
                else: cross_s=f"> {pts[-1][0]:.0f} (A6/A4 {pts[-1][1]:.2f} still < 1)"
            else: cross_s=f"**{cross:.0f}**"
            # theory from the smallest-n_q point
            n0,r0,a,b=pts[0]
            nl=NDEC[model]; partial=nl-2 if pol=='cacheblend' else nl
            tb=b['tiers'][1]['prefill_breakdown_s']; ta=a['tiers'][1]['prefill_breakdown_s']
            passes=math.ceil(n0/4)
            t_pass=tb.get('pim_prefill_score',0)/partial/passes
            t_link=ta.get('link_kv_pim_to_gpu',0)/nl
            if pol=='epic':
                t_gattn=(ta.get('gpu_score',0)+ta.get('gpu_context',0))/nl
            else:
                # CacheBlend's A4 attention total includes the 2 full-recompute
                # layers; take the partial-layer (skinny) attention from the
                # EPIC run of the same model/link (identical kernel shape).
                ae=ld(model,QS[0],'A4',link,'epic')
                te=ae['tiers'][1]['prefill_breakdown_s'] if ae else ta
                t_gattn=(te.get('gpu_score',0)+te.get('gpu_context',0))/nl
            theory=4*(t_link+t_gattn)/t_pass if t_pass>0 else float('nan')
            emit(f"| {model} | {cross_s} | {brack} | {theory:.0f} | {t_pass*1e6:.1f} / {t_link*1e6:.1f} / {t_gattn*1e6:.1f} |")
        # full ratio table
        emit("\nA6/A4 over the grid (n_q: ratio):\n")
        emit("| model | "+" | ".join(str(q) for q in QS)+" |"); emit("|---|"+"---:|"*len(QS))
        for model in MODELS:
            row=[]
            for q in QS:
                a=ld(model,q,'A4',link,pol); b=ld(model,q,'A6',link,pol)
                row.append(f"{b['tiers'][1]['prefill_s']/a['tiers'][1]['prefill_s']:.2f}" if (a and b) else "·")
            emit(f"| {model} | "+" | ".join(row)+" |")
open(S+f'../turning_points_{GRID}.md','w').write("\n".join(out)+"\n")
