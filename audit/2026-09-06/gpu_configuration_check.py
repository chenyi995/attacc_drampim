#!/usr/bin/env python3
"""Inspect real CLI/config wiring, stopping before any hardware/DAG execution."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import main as cli
import src.workload_runner as runner
from src.config import make_xpu_config,SCALING_FACTOR
from src.devices import PIM,xPU
from src.system import System
from src.model import Layer
from src.type import GPUType,DeviceType,LayerType,DataType


class Captured(Exception):
    pass


def main():
    paths=['main.py','src/config.py','src/type.py','src/devices.py','src/system.py',
           'src/ablation.py','src/workload_runner.py','experiments/run_dag_ladder.sh','experiments/run_sweep.sh']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    source=(ROOT/'main.py').read_text()
    arguments={}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            if node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value in ('--gpu','--pim-link','--gmemcap'):
                arguments[node.args[0].value]={k.arg:ast.literal_eval(k.value) for k in node.keywords if k.arg in ('choices','default')}
    evidence={'scope':'Real parser, device constructors and ablation arguments. PIM initialization has no Ramulator; execution intercepted before DAG/warm. GPU transfer prices are analytical probes, not simulated workload results.',
              'source_sha256':hashes,'cli_arguments':arguments,'matrix':[],'gpu_defaults':{},'explicit_link_controls':[]}
    rung_names=('A1','A2','A3b','A4c','A4e','A5','A6')
    active={}

    def setup(system, modelinfos, device, config, ramulator_workers=1):
        assert device==DeviceType.PIM
        system.hetero_name=device
        system.devices['Acc']=PIM(config,system.scaling_factor,None)

    def capture(system,workload,plan,**kwargs):
        gpu=system.devices['GPU'];pim=system.devices['Acc']
        transfer=Layer('audit','kv_transfer',LayerType.X2G,False,DataType.W16A16,1,2**19,1,1)
        transfer.link_latency=False
        item=dict(active,gpu_received=gpu.gpu_type.name,fp16_dense_tflops=gpu.peak_flops/1e12,
                  near_hbm_tbps=gpu.peak_memory_bandwidth/1e12,gpu_nvlink_gbps=gpu.max_interface_bandwidth/1e9,
                  gpu_to_pim_link_gbps=gpu.pim_link_bandwidth/1e9,pim_to_gpu_link_gbps=pim.max_interface_bandwidth/1e9,
                  gpu_capacity_gib_per_device=gpu.aggregate_memory_capacity/gpu.num_xpu/2**30,
                  pim_bandwidth_per_stack_tbps=pim.peak_memory_bandwidth/pim.num_hbm/1e12,
                  far_hbm_stream_tbps=gpu.far_hbm_bandwidth/1e12,
                  transfer_1MiB_no_fixed_cost_us=gpu.get_time_and_energy(transfer)[0]*1e6,
                  prefill_attn=kwargs['pim_prefill_mode'],decode_attn=kwargs['decode_attn'],
                  pim_batch_command=kwargs['pim_batch_command'],pim_pe_freq_ghz=kwargs['pim_pe_freq_ghz'],
                  gpu_model=gpu.gpu_model,pipe=kwargs['pipe'],gemv_buffer_bytes=kwargs['gemv_buffer_bytes'])
        evidence['explicit_link_controls' if active.get('explicit_link') else 'matrix'].append(item)
        raise Captured()

    previous=os.getcwd()
    with tempfile.TemporaryDirectory(prefix='fugue-gpu-config-audit-') as tmp:
        os.chdir(tmp)  # main.py's legacy output.csv handling cannot touch user files.
        wl=Path(tmp)/'input.json'
        wl.write_text(json.dumps({'meta':{'format':'v2-dag','block_tokens':256},'agents':[
            {'id':'probe','tier':0,'parent':None,'history_len':0,'lout':2,
             'segs':[{'role':'user','sha':'gpu-configuration-probe','len':16}]}]}))
        try:
            with patch.object(System,'set_accelerator',setup),patch.object(runner,'run_reuse_prefill',capture):
                cases=[(gpu,rung,None) for gpu in ('A100a','H100','H200','B200') for rung in rung_names]
                cases += [(gpu,'A6','nvlink4') for gpu in ('H200','B200')]
                for gpu,rung,explicit in cases:
                    active.clear();active.update(gpu=gpu,rung=rung,explicit_link=explicit)
                    args=['main.py','--system','dgx-attacc','--gpu',gpu,'--model','LLAMA3-8B',
                          '--ngpu','1','--num-hbm','5','--engine','dag','--pipeopt','--gpu-model','flash',
                          '--word','2','--powerlimit','--ablation',rung,'--workload',str(wl),
                          '--workload-report',str(Path(tmp)/'must-not-be-written.json'),
                          '--reuse','no-reuse' if rung=='A1' else 'recompute']
                    if explicit:args.extend(['--pim-link',explicit])
                    with patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):
                        try:cli.main()
                        except Captured:pass
                        else:raise AssertionError('Execution was not intercepted')
                    assert not (Path(tmp)/'must-not-be-written.json').exists()
        finally:os.chdir(previous)

    for gpu in ('A100a','H100','H200','B200'):
        default=make_xpu_config(getattr(GPUType,gpu),num_gpu=2,gpu_model='flash')['GPU']
        device=xPU(DeviceType.GPU,default,SCALING_FACTOR)
        op=Layer('audit','allreduce',LayerType.G2G,False,DataType.W16A16,1,2**19,1,1)
        evidence['gpu_defaults'][gpu]={'capacity_gib':default['MEM_CAPACITY_PER_DEVICE']/2**30,
                                      'config_pim_link_bw':default['PIM_LINK_BW'],
                                      'fallback_link_gbps':device.pim_link_bandwidth/1e9,
                                      'allreduce_1MiB_two_gpus_us':device.get_time_and_energy(op)[0]*1e6}
        subset=[x for x in evidence['matrix'] if x['gpu']==gpu]
        assert len(subset)==len(rung_names)
        for key in ('gpu_received','fp16_dense_tflops','near_hbm_tbps','gpu_nvlink_gbps','gpu_to_pim_link_gbps','pim_to_gpu_link_gbps','gpu_capacity_gib_per_device','transfer_1MiB_no_fixed_cost_us','gpu_model','pipe'):
            assert len({x[key] for x in subset})==1,(gpu,key)
    assert len(evidence['matrix'])==28
    for p,digest in hashes.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    (HERE/'GPU_CONFIGURATION_CHECK.json').write_text(json.dumps(evidence,indent=2)+'\n')
    lines=['| GPU | dense FP16 TFLOPS | GPU HBM TB/s | GPU↔GPU GB/s | GPU↔PIM 默认 GB/s | 默认有效显存 GiB | 型号配置显存 GiB |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for gpu in evidence['gpu_defaults']:
        x=next(x for x in evidence['matrix'] if x['gpu']==gpu)
        lines.append('| {} | {:g} | {:g} | {:g} | {:g} | {:g} | {:g} |'.format(gpu,x['fp16_dense_tflops'],x['near_hbm_tbps'],x['gpu_nvlink_gbps'],x['gpu_to_pim_link_gbps'],x['gpu_capacity_gib_per_device'],evidence['gpu_defaults'][gpu]['capacity_gib']))
    lines += ['','| A档（所有GPU逐一检查） | prefill | decode | bank 命令 | PE GHz |', '|---|---|---|---|---:|']
    for x in evidence['matrix']:
        if x['gpu']=='H200':lines.append('| {} | {} | {} | {} | {} |'.format(x['rung'],x['prefill_attn'],x['decode_attn'],x['pim_batch_command'],x['pim_pe_freq_ghz']))
    (HERE/'GPU_CONFIGURATION_TABLES.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print('28 default GPU/rung combinations and 2 explicit-link controls captured; no simulation.')


if __name__=='__main__':main()
