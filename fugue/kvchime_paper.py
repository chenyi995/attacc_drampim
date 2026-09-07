"""Make self-contained, single-metric figure packages from the multi-model run."""
import csv,json,math,shutil,os,hashlib,subprocess,sys,gzip
from pathlib import Path
from fugue.runtime import REPO,RUN,load,save,csvout,sha,check_sources
from fugue.kvchime_model import geometry

DEST=RUN/'paper';DEST.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(RUN/'matplotlib-cache')
MODELS=[r['name'] for r in load(REPO/'artifact/inputs/kvchime.json')['models']]
W=RUN/'kvchime/Fugue-asplos-results';S=RUN/'model-sweep/Fugue-asplos-results'
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def num(r,k):return float(r[k])
summary=read(W/'summary.csv');sweep=read(S/'sweep.csv');shared=read(W/'shared-mq.csv');selection=read(W/'selection.csv')
load(RUN/'kvchime/Fugue-asplos-audit/checks.json');load(RUN/'model-sweep/Fugue-asplos-audit/checks.json')
experiments={1:('prefill-boundary','Prefill attention 的 GPU/PIM/MQ 边界'),2:('link-cache-sensitivity','MQ 对 link 与 cache 的敏感性'),
    3:('software-reuse-F0-F4','CacheBlend/EPIC 的 F0–F4'),4:('shared-query-MQ','共享 query 与跨 agent MQ'),5:('device-selection','四种 prefill 选边策略')}
roots={};packages=[]
for i,(name,title) in experiments.items():
    root=DEST/f'KVChime-experiment-{i}-{name}';root.mkdir(exist_ok=True);(root/'paper').mkdir(exist_ok=True);(root/'data').mkdir(exist_ok=True)
    roots[i]=root

def package(i,name,rows,cfg,source,notes=''):
    d=roots[i]/'paper'/name;d.mkdir(exist_ok=True)
    csvout(d/'raw-data.csv',rows);save(d/'plot-config.json',cfg)
    shutil.copy2(REPO/'fugue/kvchime_figure.py',d/'plot.py')
    save(d/'models.json',[dict(geometry(s['name']),tensor_parallel=s['tensor_parallel']) for s in load(REPO/'artifact/inputs/kvchime.json')['models']])
    profiles=read((S if i in [1,2] else W)/'profiles.csv')
    csvout(d/'raw-profiles.csv',profiles)
    save(d/'provenance.json',dict(source_table=str(source.relative_to(RUN)),source_sha256=sha(source),
        simulator='This attacc-fugue checkout, native AttAcc operators and Ramulator',
        raw_run=str(RUN),native_source_hashes=check_sources(),
        plotting_reference='AttAcc ASPLOS 2024, Fig. 13: grouped, single-metric bars by model and workload.',
        note='raw-data.csv contains selected unnormalized source rows; raw-profiles.csv retains native cycle/command records. Full grid is in experiment data/.'))
    (d/'README.md').write_text(f"# {name}\n\n本图只表示 **{cfg['ylabel']}**。{notes}\n\n[PDF](figure.pdf) · [PNG](figure.png) · [原始结果](raw-data.csv) · [绘图脚本](plot.py)\n\n`raw-data.csv` 是未归一化的原始结果行；`raw-profiles.csv` 是该实验的命令 profile 原值。`plot-config.json` 记录取数/分组/归一化，`models.json` 记录模型几何。全量采样在实验上层 `data/`。\n\n```bash\npython3 plot.py\n# 输出到其它目录：\npython3 plot.py --output-dir redraw\n```\n\n只需 Python、numpy 和 matplotlib；把本文件夹单独复制出去仍能重画，不需要仿真器代码或旧 output。\n")
    command=[sys.executable,str(d/'plot.py')];p=subprocess.run(command,env=os.environ,capture_output=True,text=True)
    (d/'plot.log').write_text(p.stdout+p.stderr);assert p.returncode==0,(d,p.stderr)
    assert load(d/'render-check.json')['axes']==1
    packages.append(dict(experiment=i,name=name,path=str(d.relative_to(DEST)),source_rows=len(rows),response_metric=cfg['ylabel']))

crossings=[]
for model in MODELS:
    for c,bw in [(0,300),(1024,300),(1024,32),(1024,450)]:
        rows=sorted((r for r in sweep if r['model']==model and int(r['cached'])==c and num(r,'link_GBps')==bw),key=lambda r:int(r['q']))
        cc=[]
        for mode,color in [('plain','#26749A'),('mq','#36886B')]:
            for a,b in zip(rows,rows[1:]):
                da=num(a,mode+'_service_us')-num(a,'gpu_service_us');db=num(b,mode+'_service_us')-num(b,'gpu_service_us')
                if da*db<0:
                    record=dict(model=model,cached=c,link_GBps=bw,mode=mode,q_low=int(a['q']),q_high=int(b['q']),color=color)
                    cc.append(record);crossings.append(record)
        cfg=dict(kind='line',x_key='q',xlabel='New query tokens, Q',ylabel='Attention service (µs / layer)',
            title=f'{model} · C={c} · {bw} GB/s',logx=True,logy=True,crossings=cc,
            series=[dict(name='GPU',key='gpu_service_us',color='#C9754E',style='--s'),dict(name='PIM',key='plain_service_us',color='#26749A',style='-o'),dict(name='MQ PIM',key='mq_service_us',color='#36886B',style='-^')])
        package(1,f'{model}-C{c}-link{bw}',rows,cfg,S/'sweep.csv','交点标注为相邻采样 Q 的区间；没有交点时不制造交点。')
csvout(roots[1]/'data/crossings.csv',crossings)
csvout(roots[1]/'data/raw-sweep.csv',sweep)
for axis,key,values in [('link','link_GBps',[32,150,300,450]),('cache','cached',[0,1024,4096,8192])]:
    rows=[r for r in sweep if int(r['q']) in [8,32,128,512,1024] and num(r,key) in values and
        (int(r['cached'])==1024 if axis=='link' else num(r,'link_GBps')==300)]
    cfg=dict(kind='grouped',models=MODELS,case_keys=[key,'q'],case_prefix={key:'BW=' if axis=='link' else 'C=','q':'Q='},
        wide_series=[dict(name='MQ',key='mq_speedup_over_GPU')],series_order=['MQ'],title=f'MQ sensitivity to {axis}',ylabel='GPU / MQ PIM service time')
    package(2,axis+'-sensitivity',rows,cfg,S/'sweep.csv','固定未扫描参数，用横向分组长条形图表示；纵轴 >1 表示 MQ PIM 更快。')
csvout(roots[2]/'data/raw-sweep.csv',sweep)

for metric,label in [('TTFT_ms','TTFT'),('TBT_ms','TBT'),('E2E_ms','E2E latency'),('simultaneous_peak_KV_GiB','Peak KV capacity'),('decode_scan_ms_per_token','Decode scan')]:
    variants=['F0','F1','F2','F3','F4'] if metric!='decode_scan_ms_per_token' else ['F2','F3','F4']
    rows=[r for r in summary if r['variant'] in variants]
    base='F0' if metric!='decode_scan_ms_per_token' else 'F2'
    cfg=dict(kind='grouped',models=MODELS,case_keys=['workload_id'],series_key='variant',value_key=metric,
        series_order=variants,normalize_to=base,inverse_normalization=(metric!='simultaneous_peak_KV_GiB'),
        title=label+' by model and workload',ylabel=(label+' speedup over '+base if metric!='simultaneous_peak_KV_GiB' else label+' / '+base))
    package(3,metric,rows,cfg,W/'summary.csv','模型是外层分组，workload 是内层分组；性能图表示 baseline latency / 本方案 latency（越大越快），容量图表示本方案 / F0（越小占用越少）。所有绝对值保存在原始数据中。')
for p in W.glob('*.csv'):
    if p.name=='events.csv':
        dst=roots[3]/'data/events.csv.gz'
        with p.open('rb') as src,dst.open('wb') as out:
            with gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0) as gz:shutil.copyfileobj(src,gz)
        with gzip.open(dst,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==sha(p)
        save(roots[3]/'data/events-provenance.json',dict(source=str(p.relative_to(RUN)),source_sha256=sha(p),
            original_bytes=p.stat().st_size,compressed_sha256=sha(dst),compression='Lossless gzip, deterministic mtime=0; decompressed bytes verified identical.'))
        previous=roots[3]/'data/events.csv'
        if previous.exists():
            arc=RUN/'archived/paper-before-lossless-event-compression';arc.mkdir(parents=True,exist_ok=True)
            assert not (arc/'events.csv').exists();shutil.move(previous,arc/'events.csv')
    else:shutil.copy2(p,roots[3]/'data'/p.name)
shutil.copy2(RUN/'kvchime/workload.json',roots[3]/'data/workload.json')

for metric,label in [('scan_us','Scan (µs / layer)'),('TBT_ms','TBT (ms)')]:
    cfg=dict(kind='grouped',models=MODELS,case_keys=['agents','nominal_shared_fraction'],case_prefix={'agents':'A=','nominal_shared_fraction':'S='},
        series_key='mode',series_order=['single_query','MQ'],series_labels={'single_query':'Single query','MQ':'MQ'},value_key=metric,
        ylabel=label,title='Concurrent queries on shared KV')
    package(4,metric,shared,cfg,W/'shared-mq.csv','A 是就绪 agent 数，S 是按源 chunk 数选择的共享比例；具体共享 token 数保留在数据中。')
shutil.copy2(W/'shared-mq.csv',roots[4]/'data/shared-mq.csv')

selected=[r for r in selection if int(r['q']) in [32,128,512]]
cfg=dict(kind='grouped',models=MODELS,case_keys=['cached','q'],case_prefix={'cached':'C=','q':'Q='},
    wide_series=[dict(name=n,key=k) for n,k in [('GPU','fixed_GPU_us'),('PIM','fixed_PIM_us'),('Algorithm','algorithm_us'),('Oracle','oracle_us')]],
    series_order=['GPU','PIM','Algorithm','Oracle'],series_labels={'PIM':'MQ PIM'},normalize_to='Oracle',logy=True,
    ylabel='Attention service / oracle',title='Fixed devices, estimated-cost selection, and oracle')
package(5,'four-policies',selected,cfg,W/'selection.csv','Oracle 是相同服务范围内两个实际成本的逐点较小值；算法只使用两个独立校准点估计 PIM，不读取 oracle 答案。')
for name in ['selection.csv','selector-calibration.csv','decisions.csv']:shutil.copy2(W/name,roots[5]/'data'/name)

reductions=[]
for cid in dict.fromkeys(r['case_id'] for r in summary):
    by={r['variant']:r for r in summary if r['case_id']==cid}
    for a,b in [('F0','F1'),('F1','F2'),('F2','F3'),('F3','F4'),('F0','F4')]:
        for key in ['TTFT_ms','TBT_ms','E2E_ms','TTFT_energy_mJ','TBT_energy_mJ','E2E_energy_mJ','simultaneous_peak_KV_GiB']:
            reductions.append(dict(model=by[a]['model'],workload=by[a]['workload_id'],baseline=a,variant=b,metric=key,reduction_percent=100*(1-num(by[b],key)/num(by[a],key))))
csvout(roots[3]/'data/reductions.csv',reductions)
for i,root in roots.items():
    title=experiments[i][1];items=[r for r in packages if r['experiment']==i]
    (root/'README.md').write_text(f'# Experiment {i}：{title}\n\n统一模型：'+', '.join(MODELS)+'.\n\n每张图只保留一个纵轴指标；多模型/多 case 用 AttAcc Fig. 13 式横向分组条形图。每个 `paper/` 子文件夹均包含最终 PDF/PNG、原始数据与独立 `plot.py`。\n\n'+ '\n'.join(f"- [{r['name']}](paper/{r['name']}/README.md)" for r in items)+'\n\n完整采样和绝对值见 `data/`。默认 A100a、NVLink 3；模型几何与 GQA 假设见每图 `models.json` 及仓库 `docs/KVChime-multi-model.md`。\n')
energy_reductions=[r['reduction_percent'] for r in reductions if r['baseline']=='F0' and r['variant']=='F4' and r['metric']=='E2E_energy_mJ']
p=roots[3]/'README.md'
p.write_text(p.read_text()+f'\n能耗采用 AttAcc 原生动态能耗口径，F4 相对 F0 的 E2E 能耗降低范围为 {min(energy_reductions):.1f}%–{max(energy_reductions):.1f}%（负值为增加）；完整数值保留在 `data/summary.csv`，不单独绘图。\n')
(DEST/'README.md').write_text('# KVChime：多模型最终实验\n\n'+ '\n'.join(f"- [Experiment {i}：{experiments[i][1]}]({r.name}/README.md)" for i,r in roots.items())+'\n\nLLAMA-7B、GPT-13B、LLAMA-65B 使用 AttAcc 原生模型；LLAMA3.1-8B 是显式 GQA 扩展。频率和面积属于 kvpim-rtl 的独立硬件实验，不按模型重复同一硬件数据。\n')
def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def append(i,text):
    p=roots[i]/'README.md';p.write_text(p.read_text()+text)
ct=[]
for model in MODELS:
    for c,bw in [(0,300),(1024,300),(1024,32),(1024,450)]:
        values=[]
        for mode in ['plain','mq']:
            cr=[r for r in crossings if r['model']==model and r['cached']==c and r['link_GBps']==bw and r['mode']==mode]
            if cr:values.append(', '.join(str(r['q_low'])+'–'+str(r['q_high']) for r in cr))
            else:
                samples=[r for r in sweep if r['model']==model and int(r['cached'])==c and num(r,'link_GBps')==bw]
                differences=[num(r,mode+'_service_us')-num(r,'gpu_service_us') for r in samples]
                values.append('无交点；PIM 较快' if max(differences)<0 else '无交点；GPU 较快' if min(differences)>0 else '无严格交点；含相等样本')
        ct.append([model,c,bw,*values])
append(1,'\n## 交点与成本\n\n'+mdtable(['Model','C','GB/s 单向','PIM/GPU Q 区间','MQ/GPU Q 区间'],ct)+
    '\nGPU service = max(GPU QK + softmax + PV + cached KV 读回, 新 KV 写入)；PIM service = Q 输入 + QK/PV scan + softmax + 输出返回 + max(0, 新 KV 写入 − overlap 窗口)。窗口取第一组 scan 启动到首次消费新 K 的原生命令时间；C=0 时窗口为零。\n\n小 Q 的 GPU 成本可能主要是旧 KV 读回；普通 PIM 的逐 query 扫描随 Q 增长，MQ 让一列读取服务最多 8 个 query，推迟 PIM 算力成为限制的区间。Q 较大时 GPU 的矩阵吞吐可能占优，因此出现交点。缓存大小同时改变计算和读取，但 padding、固定传输项及设备带宽不同，交点不保证与 C 无关；C=0 没有旧 KV 回读项，也不保证存在 PIM 优势区。范围是实际相邻采样，不能解释为已逐整数测量。\n')
st=[]
for model in MODELS:
    rs=[r for r in sweep if r['model']==model]
    vals=[num(r,'mq_speedup_over_GPU') for r in rs]
    st.append([model,f'{min(vals):.3f}',f'{max(vals):.3f}'])
append(2,'\n## 解读\n\n'+mdtable(['Model','全网格最小 GPU/MQ','全网格最大 GPU/MQ'],st)+
    '\n>1 表示 MQ PIM 更快，<1 表示 GPU 更快。图中的所有样本来自同一个完整 grid；link 图固定 C=1024，cache 图固定单向 300 GB/s。Link 扫描仅重新计算传输及暴露时间，复用同一形状本次生成的 native scan；没有修改 scan 以制造交点。完整 latency 各组成项见 `data/raw-sweep.csv`。\n')
rr=[]
for model in MODELS:
    for a,b in [('F2','F3'),('F3','F4')]:
        line=[model,a+' → '+b]
        for metric in ['TTFT_ms','TBT_ms','E2E_ms','simultaneous_peak_KV_GiB']:
            vals=[r['reduction_percent'] for r in reductions if r['model']==model and r['baseline']==a and r['variant']==b and r['metric']==metric]
            line.append(f'{min(vals):.1f}%–{max(vals):.1f}%')
        rr.append(line)
append(3,'\n## Workload、对照与收益\n\n八种输入分别是 CacheBlend demo 2/3、EPIC LongContext 4K/8K/16K，以及同一 8K 文档被 1/2/4 个已就绪 reader 复用。准确 token 数、chunk 顺序、重算位置和输出截断见 `data/workload.json`；逐模型复用冻结形状，不重新分词，也不测量 LLM 输出精度。\n\nF0 全 GPU 重算；F1 软件 reuse + GPU attention/decode；F2 软件 reuse + GPU 拼出完整私有 KV 后写回原生 PIM decode；F3 只存私有重算/生成 KV、GPU prefill、共享视图单查询 PIM decode；F4 保持共享视图，增加 MQ decode 与估计成本 prefill 选边。\n\n性能柱状图使用加速比以展示各档位差距，容量图使用相对峰值占用；每张图只表示一个指标。下表给出每个模型八个 workload 的降低比例范围，负值就是衰退，没有删去；逐 case 的 F0/F1 等全部比较在 `data/reductions.csv`。\n\n'+mdtable(['Model','比较','TTFT 降低','TBT 降低','E2E 降低','KV 峰值降低'],rr)+
    '\nTTFT/TBT/E2E 包含正常 Transformer 层的 projection、FFN 和 attention 服务，因此 scan 下降不等于 TBT 按同样比例下降。容量取同时刻 GPU + remote pool + private 的总峰值；完整分项和 snapshot 在 `data/storage.csv`。完整事件表以无损 `data/events.csv.gz` 保存，解压字节哈希已核对原始 CSV；共享池预热单列 `data/warmup.csv`，F2 导出 overlap 对照单列 `data/F2-overlap-control.csv`。共享块中已失效的旧项仍可能被物理扫描，且共享视图带来分段和额外 Q/descriptor 成本，所以 F3 的 scan/TBT 不保证比 F2 小；它主要减少完整物化和私有副本。F1→F3 不是独立 RoPE kernel 消融，旋转算术沿用 native 未单列计时的范围。\n')
mr=[]
for model in MODELS:
    for agents in [1,2,4,8]:
        by={r['mode']:r for r in shared if r['model']==model and int(r['agents'])==agents and num(r,'nominal_shared_fraction')==1}
        a,b=by['single_query'],by['MQ']
        mr.append([model,agents,f"{100*(1-num(b,'scan_us')/num(a,'scan_us')):.1f}%",f"{100*(1-num(b,'TBT_ms')/num(a,'TBT_ms')):.1f}%"])
append(4,'\n## 完全共享文档的结果\n\n'+mdtable(['Model','Ready agents','Scan 降低','TBT 降低'],mr)+
    '\n取 EPIC 4K 文档作为共享池，每个 agent 独立保留重算项和新 token。MQ 复用共享列操作数，不合并 query 的累加器或 softmax。MHA 单 agent 只有一条 Q，MQ 必须退化为相同 decode；GQA 单 agent 的同一 KV head 已服务多个 Q heads，因此可能出现 MQ 收益。额外的 0/50% 共享控制保留在图和原始数据，实际共享 token 数单列，不能把按 chunk 选出的比例误称为精确 token 比例。这里假设 agent 已就绪，不含组批等待或线上排队。\n')
selrows=[]
for model in MODELS:
    rs=[r for r in selection if r['model']==model]
    wrong=sum(r['algorithm_choice']!=r['oracle_choice'] for r in rs)
    regrets=[num(r,'regret_percent') for r in rs]
    selrows.append([model,len(rs),wrong,f'{sum(regrets)/len(regrets):.2f}%',f'{max(regrets):.2f}%'])
append(5,'\n## 选择误差\n\n'+mdtable(['Model','样本','与 oracle 不同','平均 regret','最大 regret'],selrows)+
    '\nRegret = (算法实际 service / 两设备实际 service 的较小值 − 1)。朴素算法只用 N=1024/4096 的两条 8-query profile 拟合每对象 scan，然后加 Q/descriptor、输出、softmax 和全量新 KV 写入；不读当前样本的实际 PIM 时间来决策。Oracle 使用独立模拟所得的实际较小值，仅作上界。对象边界、padding 和保守的写入估计会引起错误选边；这里保留全部 96 个样本，图仅选 Q=32/128/512 展示。算法伪代码见仓库 `docs/KVChime-multi-model.md`。\n')

worst=max(selection,key=lambda r:num(r,'regret_percent'))
append(5,f"\n最差样本是 {worst['model']}、C={worst['cached']}、Q={worst['q']}：GPU {num(worst,'fixed_GPU_us'):.3f} µs，实际 MQ PIM {num(worst,'fixed_PIM_us'):.3f} µs，算法只估计 {num(worst,'estimated_PIM_us'):.3f} µs。两点校准来自 N=1024/4096，向很短对象外推会漏估行/列粒度和 query 移动命令的成本；GQA 又放大了 query 组数。这是朴素估计器的失效场景，实际仿真时间没有被删去，也没有按 oracle 重选后改写数据。\n")

save(DEST/'figure-index.json',packages)
for d in [DEST/r['path'] for r in packages]+[DEST]:
    files=[p for p in d.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt']
    (d/'SHA256SUMS.txt').write_text('\n'.join(sha(p)+'  '+str(p.relative_to(d)) for p in sorted(files))+'\n')
save(RUN/'model-paper-checks.json',dict(figures=len(packages),models=MODELS,all_figures_have_single_axis=True,
    all_figures_have_raw_data_and_standalone_plot=True,experiments=len(roots)))
print('PAPER PACKAGES',len(packages),flush=True)
