#!/usr/bin/env python3
"""One CPU-only numerical KVChime check with real Qwen2.5-0.5B Q/K/V.

Python 3.10+; tested with torch==2.6.0+cpu, transformers==4.57.6,
numpy==2.2.6. Install the CPU torch wheel from https://download.pytorch.org/whl/cpu.
Run: python kvchime_correctness.py
Optional --cache-dir is the Hugging Face download cache. --tensor-cache saves
the one real forward's projections outside the deliverable folder, permitting
arithmetic diagnostics without another model forward. It never caches results.
Outputs: results.csv and correctness_report.md beside this script by default.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import traceback

MODEL_ID = 'Qwen/Qwen2.5-0.5B'
MODEL_REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'
TEXT = ('Shared documents can be reused by multiple language model agents. '
        'Each agent keeps its own updates and generated tokens.')
SEED, TOKEN_COUNT, MQ_CAPACITY = 42, 320, 8
FIELDS = ['precision', 'request', 'alignment_enabled', 'max_abs_error',
          'relative_l2_error', 'allclose', 'atol', 'rtol']


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def extract_once(args):
    """Only this function runs the language model; each hook must fire once."""
    if args.tensor_cache and args.tensor_cache.exists():
        data = torch.load(args.tensor_cache, map_location='cpu', weights_only=True)
        assert data['metadata']['model_id'] == MODEL_ID
        assert args.revision == 'main' or data['metadata']['model_revision'] == args.revision
        assert data['metadata']['text'] == TEXT and data['metadata']['seed'] == SEED
        assert data['metadata']['model_forward_calls'] == 1
        assert data['input_ids'].numel() == TOKEN_COUNT
        for name in ['q', 'k', 'v']:
            assert tensor_hash(data[name]) == data['metadata']['tensor_sha256'][name]
        print('Reusing the saved real single-forward projections; no model run.', flush=True)
        return data, 0

    options = dict(cache_dir=args.cache_dir, local_files_only=args.local_files_only,
                   revision=args.revision, trust_remote_code=False)
    print('Loading fixed model and tokenizer on CPU in FP32.', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
    repetitions = 1
    while True:
        ids = tokenizer(' '.join([TEXT] * repetitions), add_special_tokens=False)['input_ids']
        if len(ids) >= TOKEN_COUNT:
            break
        repetitions += 1
    input_ids = torch.tensor([ids[:TOKEN_COUNT]], dtype=torch.long, device='cpu')
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32,
                                    attn_implementation='eager', **options).to('cpu').eval()
    assert model.config.model_type == 'qwen2'
    assert model.config._attn_implementation == 'eager'
    last = model.layers[-1].self_attn
    captures, calls = {}, dict(model=0, q=0, k=0, v=0)

    def model_hook(module, inputs, output):
        calls['model'] += 1

    def projection_hook(name):
        def hook(module, inputs, output):
            calls[name] += 1
            captures[name] = output.detach().cpu().clone()
        return hook

    hooks = [model.register_forward_hook(model_hook)]
    for name in ['q', 'k', 'v']:
        hooks.append(getattr(last, name + '_proj').register_forward_hook(projection_hook(name)))
    print('Running the only model forward: 320 tokens, eval/no_grad, eager attention.', flush=True)
    try:
        with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                  position_ids=torch.arange(TOKEN_COUNT)[None, :], use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    assert calls == dict(model=1, q=1, k=1, v=1), calls
    config = model.config
    heads, kv_heads, dim = config.num_attention_heads, config.num_key_value_heads, last.head_dim
    assert heads % kv_heads == 0
    data = dict(input_ids=input_ids, config=config.to_dict(),
                inv_freq=model.rotary_emb.inv_freq.detach().cpu().clone())
    for name, count in [('q', heads), ('k', kv_heads), ('v', kv_heads)]:
        assert captures[name].shape == (1, TOKEN_COUNT, count * dim)
        data[name] = captures[name].reshape(TOKEN_COUNT, count, dim).contiguous()
        assert data[name].dtype == torch.float32 and torch.isfinite(data[name]).all()
    data['metadata'] = dict(
        model_id=MODEL_ID, model_revision=config._commit_hash, seed=SEED, text=TEXT,
        text_repetitions=repetitions, tokens_before_truncation=len(ids), tokens=TOKEN_COUNT,
        input_ids_sha256=tensor_hash(input_ids), model_forward_calls=calls['model'],
        hook_calls={name: calls[name] for name in ['q', 'k', 'v']},
        layer_index=len(model.layers)-1, model_dtype=str(next(model.parameters()).dtype),
        tensor_dtype=str(data['q'].dtype), device=str(next(model.parameters()).device),
        query_heads=heads, kv_heads=kv_heads, head_dim=dim,
        rope_theta=config.rope_theta, rope_type=model.rotary_emb.rope_type,
        attention_scaling=float(model.rotary_emb.attention_scaling),
        tensor_sha256={name: tensor_hash(data[name]) for name in ['q', 'k', 'v']},
        extracted_utc=datetime.now(timezone.utc).isoformat(),
        extraction_versions={name: importlib.metadata.version(name) for name in
                             ['torch', 'transformers', 'tokenizers', 'huggingface_hub', 'safetensors', 'numpy']},
    )
    if args.tensor_cache:
        args.tensor_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, args.tensor_cache)
    del model, captures
    print('Captured last-layer pre-RoPE Q/K/V; model forward count = 1.', flush=True)
    return data, 1


class CheckedRoPE:
    """Use the model's actual frequencies and Transformers rotate_half pairing.

    x is (..., heads, dim) with positions broadcasting over the leading axes,
    or (rows, dim) with one position per row. FP64 recomputes phases/trig in
    FP64 from the same model inv_freq values; it does not cast FP32 cos/sin.
    """
    def __init__(self, data):
        assert data['metadata']['rope_type'] == 'default'
        assert data['metadata']['attention_scaling'] == 1.0
        self.inv_freq = data['inv_freq']

    def __call__(self, x, positions):
        positions = torch.as_tensor(positions, dtype=x.dtype, device=x.device)
        phases = positions[..., None] * self.inv_freq.to(dtype=x.dtype, device=x.device)
        angles = torch.cat((phases, phases), dim=-1)
        while angles.ndim < x.ndim:
            angles = angles.unsqueeze(-2)
        return x * angles.cos() + rotate_half(x) * angles.sin()


def check_rope(data, rope):
    """Check pairing, native FP32 RoPE, FP64 arithmetic and rotation composition."""
    native = Qwen2RotaryEmbedding(Qwen2Config.from_dict(data['config']))
    assert torch.equal(native.inv_freq, data['inv_freq'])
    dim = data['metadata']['head_dim']
    basis = torch.arange(dim, dtype=torch.float64)
    assert torch.equal(rotate_half(basis), torch.cat((-basis[dim//2:], basis[:dim//2])))
    positions = torch.tensor([-192, -128, -64, 0, 17, 33, 127, 128, 143, 159, 192, 255, 271, 319, 335])
    q, k = data['q'][:len(positions)], data['k'][:len(positions)]
    cos, sin = native(q, positions[None, :])
    nq, nk = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin)
    native_error = max((rope(q, positions)-nq[0].transpose(0, 1)).abs().max().item(),
                       (rope(k, positions)-nk[0].transpose(0, 1)).abs().max().item())
    assert torch.allclose(rope(q, positions), nq[0].transpose(0, 1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(rope(k, positions), nk[0].transpose(0, 1), atol=1e-6, rtol=1e-6)
    q64 = q.double()
    phase64 = positions.double()[:, None] * data['inv_freq'].double()[None, :]
    angles64 = torch.cat((phase64, phase64), dim=-1)[None]
    native64, _ = apply_rotary_pos_emb(q64.transpose(0, 1)[None], q64.transpose(0, 1)[None],
                                      angles64.cos(), angles64.sin())
    assert torch.equal(rope(q64, positions), native64[0].transpose(0, 1))
    composed = rope(rope(q64, 335), -192)
    direct = rope(q64, 143)
    composition_error = (composed-direct).abs().max().item()
    assert torch.allclose(composed, direct, atol=1e-10, rtol=1e-10)
    return dict(native_FP32_max_abs=native_error, FP64_composition_max_abs=composition_error,
                pairing='rotate_half: [-x[d/2:], x[:d/2]]', passed=True)


def materialized_reference(q, k, v, rope):
    """Path A: independent full K/V construction and logical-position rotation."""
    results = {}
    query_heads, kv_heads, dim = q.shape[1], k.shape[1], q.shape[-1]
    for name, offset, tail_source, replacements, query_source, query_position in [
        ('A', 0, 256, [(17, 288), (143, 289)], 271, 271),
        ('B', 64, 272, [(33, 290), (159, 291)], 287, 335),
    ]:
        full_k = torch.cat((k[0:256].clone(), k[tail_source:tail_source+16].clone()), dim=0)
        full_v = torch.cat((v[0:256].clone(), v[tail_source:tail_source+16].clone()), dim=0)
        sources = torch.cat((torch.arange(256), torch.arange(tail_source, tail_source+16)))
        for logical_index, replacement_source in replacements:
            full_k[logical_index] = k[replacement_source]
            full_v[logical_index] = v[replacement_source]
            sources[logical_index] = replacement_source
        positions = torch.arange(272) + offset
        full_k = rope(full_k, positions)
        query = rope(q[query_source].clone(), query_position)
        # Reference GQA expands KV heads; Path B instead maps each scheduled Q head.
        keys = full_k.transpose(0, 1).repeat_interleave(query_heads // kv_heads, dim=0)
        values = full_v.transpose(0, 1).repeat_interleave(query_heads // kv_heads, dim=0)
        logits = torch.matmul(query[:, None, :], keys.transpose(-1, -2)).squeeze(1) / math.sqrt(dim)
        logits = logits.masked_fill(positions[None, :] > query_position, -torch.inf)
        probabilities = torch.softmax(logits, dim=-1)
        output = torch.matmul(probabilities[:, None, :], values).squeeze(1)
        results[name] = dict(output=output, logits=logits, probabilities=probabilities,
                             positions=positions, sources=sources)
    return results


def make_shared_state(q, k, v, rope):
    """Path B's independent chunk-local descriptors and private-only K/V storage."""
    shared = [dict(key=rope(k[start:start+128].clone(), torch.arange(128)),
                   value=v[start:start+128].clone(), source_start=start, reference_start=0)
              for start in (0, 128)]
    requests = {}
    # Independent descriptors use (chunk id, local offset, replacement source).
    for name, starts, patch_records, tail_source, tail_position, query_source, query_position in [
        ('A', (0, 128), [(0, 17, 288), (1, 15, 289)], 256, 256, 271, 271),
        ('B', (64, 192), [(0, 33, 290), (1, 31, 291)], 272, 320, 287, 335),
    ]:
        invalid = [set(), set()]
        private_sources, private_positions = [], []
        for chunk_id, local_offset, replacement_source in patch_records:
            assert local_offset not in invalid[chunk_id]
            invalid[chunk_id].add(local_offset)
            private_sources.append(replacement_source)
            private_positions.append(starts[chunk_id] + local_offset)
        private_sources.extend(range(tail_source, tail_source+16))
        private_positions.extend(range(tail_position, tail_position+16))
        source_ids = torch.tensor(private_sources)
        positions = torch.tensor(private_positions)
        requests[name] = dict(shared=shared, starts=starts, invalid=invalid,
            post_query=rope(q[query_source].clone(), query_position), query_position=query_position,
            private=dict(owner=name, key=rope(k[source_ids].clone(), positions),
                         value=v[source_ids].clone(), positions=positions, sources=source_ids))
    assert requests['A']['shared'] is requests['B']['shared'] is shared
    for chunk in shared:
        assert chunk['key'].shape[0] == chunk['value'].shape[0] == 128
    assert requests['A']['private']['value'].data_ptr() != requests['B']['private']['value'].data_ptr()
    return shared, requests


def kvchime_attention(shared, requests, rope, alignment_enabled):
    """Path B: MQ shared QK/PV; one global softmax per Q, private reads by owner."""
    heads, dim = requests['A']['post_query'].shape
    kv_heads = shared[0]['key'].shape[1]
    group_size = heads // kv_heads
    results = {name: dict(output=torch.empty_like(r['post_query']),
                         logits=torch.empty((heads, 272), dtype=r['post_query'].dtype),
                         probabilities=torch.empty((heads, 272), dtype=r['post_query'].dtype))
               for name, r in requests.items()}
    assigned, mq_groups = set(), []
    for kv_head in range(kv_heads):
        members = [(name, query_head) for name in requests
                   for query_head in range(heads) if query_head // group_size == kv_head]
        for begin in range(0, len(members), MQ_CAPACITY):
            batch = members[begin:begin+MQ_CAPACITY]
            assert 0 < len(batch) <= MQ_CAPACITY
            mq_groups.append(dict(kv_head=kv_head, queries=len(batch)))
            ordinary_q = torch.stack([requests[name]['post_query'][head] for name, head in batch])
            block_logits = []
            for chunk_id, chunk in enumerate(shared):
                delta = torch.tensor([requests[name]['starts'][chunk_id] - chunk['reference_start']
                                      for name, _ in batch])
                aligned_q = rope(ordinary_q, -delta) if alignment_enabled else ordinary_q
                # The same physical K operand is read by both requests in this MQ group.
                block_logits.append(aligned_q @ chunk['key'][:, kv_head, :].T / math.sqrt(dim))
            block_weights = [ordinary_q.new_zeros((len(batch), 128)) for _ in shared]
            private_weights = []
            for row, (name, head) in enumerate(batch):
                request = requests[name]
                score_parts, position_parts, source_parts, visible_indices = [], [], [], []
                for chunk_id, chunk in enumerate(shared):
                    positions = torch.arange(128) + request['starts'][chunk_id]
                    valid = positions <= request['query_position']  # actual logical causal mask
                    valid[list(request['invalid'][chunk_id])] = False
                    masked = block_logits[chunk_id][row].masked_fill(~valid, -torch.inf)
                    indices = torch.where(valid)[0]
                    visible_indices.append(indices)
                    score_parts.append(masked[indices])
                    position_parts.append(positions[indices])
                    source_parts.append(torch.arange(128)[indices] + chunk['source_start'])
                private = request['private']
                assert private['owner'] == name
                private_logits = ordinary_q[row] @ private['key'][:, kv_head, :].T / math.sqrt(dim)
                private_valid = private['positions'] <= request['query_position']
                private_logits = private_logits.masked_fill(~private_valid, -torch.inf)
                score_parts.append(private_logits[private_valid])
                position_parts.append(private['positions'][private_valid])
                source_parts.append(private['sources'][private_valid])
                logits = torch.cat(score_parts)
                positions, sources = torch.cat(position_parts), torch.cat(source_parts)
                assert logits.numel() == positions.unique().numel() == 272
                # No per-chunk normalization: exactly one full-context softmax per Q.
                probabilities = torch.softmax(logits, dim=-1)
                cursor = 0
                for chunk_id, indices in enumerate(visible_indices):
                    count = len(indices)
                    block_weights[chunk_id][row, indices] = probabilities[cursor:cursor+count]
                    cursor += count
                own_weights = ordinary_q.new_zeros(len(private['positions']))
                own_weights[private_valid] = probabilities[cursor:]
                private_weights.append(own_weights)
                # Canonical order is for diagnostics only; no shared K/V materialization.
                order = torch.argsort(positions)
                results[name]['logits'][head] = logits[order]
                results[name]['probabilities'][head] = probabilities[order]
                if 'positions' in results[name]:
                    assert torch.equal(results[name]['positions'], positions[order])
                    assert torch.equal(results[name]['sources'], sources[order])
                results[name]['positions'], results[name]['sources'] = positions[order], sources[order]
            # Shared V also remains a shared matrix operand in the MQ PV operation.
            output = ordinary_q.new_zeros(ordinary_q.shape)
            for chunk_id, chunk in enumerate(shared):
                output += block_weights[chunk_id] @ chunk['value'][:, kv_head, :]
            for row, (name, head) in enumerate(batch):
                private = requests[name]['private']
                assert private['owner'] == name
                output[row] += private_weights[row] @ private['value'][:, kv_head, :]
                assert (name, head) not in assigned
                assigned.add((name, head))
                results[name]['output'][head] = output[row]
    assert assigned == {(name, head) for name in requests for head in range(heads)}
    return results, mq_groups


def compare(ref, test, precision, request, aligned, tolerance):
    delta = test['output'] - ref['output']
    maximum = delta.abs().max().item()
    relative = (torch.linalg.vector_norm(delta) /
                torch.linalg.vector_norm(ref['output']).clamp_min(1e-12)).item()
    row = dict(precision=precision, request=request, alignment_enabled=aligned,
               max_abs_error=maximum, relative_l2_error=relative,
               allclose=bool(torch.allclose(test['output'], ref['output'], atol=tolerance, rtol=tolerance)),
               atol=tolerance, rtol=tolerance)
    order_matches = torch.equal(test['positions'], ref['positions'])
    sources_match = torch.equal(test['sources'], ref['sources'])
    diagnostic = dict(precision=precision, request=request, alignment_enabled=aligned,
        valid_token_order_matches=order_matches, replacement_source_order_matches=sources_match,
        QK_logits_max_abs=(test['logits']-ref['logits']).abs().max().item(),
        softmax_max_abs=(test['probabilities']-ref['probabilities']).abs().max().item())
    if not order_matches or not sources_match:
        diagnostic.update(reference_positions=ref['positions'].tolist(), test_positions=test['positions'].tolist(),
                          reference_sources=ref['sources'].tolist(), test_sources=test['sources'].tolist())
    return row, diagnostic


def write_results(folder, rows):
    with (folder / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(folder, data, fresh_calls, rope_checks, rows, diagnostics, groups, negative_ok):
    meta = data['metadata']
    positives_ok = all(r['allclose'] for r in rows if r['alignment_enabled'])
    semantics_ok = all(d['valid_token_order_matches'] and d['replacement_source_order_matches'] for d in diagnostics)
    passed = positives_ok and semantics_ok and negative_ok
    lines = [
        '# KVChime 最小数值正确性实验', '',
        f"结论：{'通过' if passed else '未通过'}。给定本实验同一构造缓存状态，"
        + ('共享视图与显式物化 attention 在 FP64/FP32 下均通过既定容差；关闭位置对齐的负对照被检出。' if passed else '存在未通过的检查，以下保留实际误差与诊断，不调整容差。'), '',
        f"模型为 `{MODEL_ID}`（revision `{meta['model_revision']}`）。只进行一次真实模型前向，在最后一层（从 0 计为 {meta['layer_index']}）的 q_proj/k_proj/v_proj 上分别触发一次 hook，提取 RoPE 前 Q/K/V；随后不再运行模型或生成。",
        f"提取设备 `{meta['device']}`，模型/张量 dtype 均为 `{meta['model_dtype']}`，eval、no_grad、eager attention；seed=42，CPU 线程数 {torch.get_num_threads()}。TF32 和 autocast 关闭，attention 显式计算，不调用 FlashAttention。",
        f"固定原文：{TEXT} 原文用一个空格连接重复 {meta['text_repetitions']} 次，不添加特殊 token；分词后 {meta['tokens_before_truncation']} 个 token，截取前 320 个。",
        f"Q/K/V 形状分别为 `{list(data['q'].shape)}`、`{list(data['k'].shape)}`、`{list(data['v'].shape)}`。GQA={meta['query_heads']}/{meta['kv_heads']}，head_dim={meta['head_dim']}。", '',
        'A/B 共享源 token 0–127、128–255 两块 KV。每块 reference start=0、reference positions=0–127；A 的 consumer starts=0/128，B=64/192。A 使用 tail 256–271、replacement 288/289→共享下标17/143、Q271@271；B 使用 tail 272–287@320–335、replacement 290/291→共享下标33/159、Q287@335。replacement 保留目标逻辑位置，私有数据只由所属请求读取；每请求恰有272个有效逻辑位置。', '',
        '物化参考从未旋转 K 独立构造完整 KV 并按实际位置旋转。KVChime 使用独立的块内 replacement 描述符，保留一份共享 post-RoPE K/V，屏蔽旧版本，以 R(-Δ)Q 访问共享 K；收集有效 logits 后只做一次全局 softmax。PV 分别访问共享 V 和私有 V，不构造完整私有共享-KV副本。',
        f"MQ 按 KV head 合并两个请求的 query heads，容量8；每个 KV head 的 {2 * meta['query_heads'] // meta['kv_heads']} 条 Q 分为 {[g['queries'] for g in groups if g['kv_head'] == 0]}，共 {len(groups)} 组，按请求和原 query head 恢复输出顺序。", '',
        f"RoPE 使用模型实际 inv_freq、theta={meta['rope_theta']:g}、default scaling 和 Transformers 的前后半维 `rotate_half` 配对。基础函数对原生 FP32 RoPE 的最大误差为 {rope_checks['native_FP32_max_abs']:.3e}，FP64 旋转组合检查误差为 {rope_checks['FP64_composition_max_abs']:.3e}。FP64 从同一实际 inv_freq 转为 double 后重算角度及三角函数；没有将已舍入的 FP32 cos/sin 冒充 FP64。", '',
        '工程检查阈值固定：FP64 atol=rtol=1e-8；FP32 atol=rtol=1e-4。相对 L2 = norm(test-ref)/max(norm(ref),1e-12)，比较对象是输出 projection 之前的 attention 输出。', '',
        '| 精度 | 请求 | Q 位置对齐 | 最大绝对误差 | 相对 L2 误差 | torch.allclose |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for r in rows:
        lines.append(f"| {r['precision']} | {r['request']} | {'开启' if r['alignment_enabled'] else '关闭（负对照）'} | {r['max_abs_error']:.8e} | {r['relative_l2_error']:.8e} | {r['allclose']} |")
    ratios = []
    for r in rows:
        if r['alignment_enabled']:
            bad = next(x for x in rows if x['precision'] == r['precision'] and x['request'] == r['request'] and not x['alignment_enabled'])
            ratios.append(bad['max_abs_error'] / max(r['max_abs_error'], 1e-30))
    lines += ['', f"负对照检查：{'通过' if negative_ok else '未通过'}。要求关闭对齐后的每个样本 allclose=False，且最大误差至少为 max(对齐版本误差, 该 dtype 的 epsilon) 的100倍；实测相对对齐版本的最小误差倍数为 {min(ratios):.3e}。",
              f"有效位置和 replacement 源 token 的规范顺序核对：{'全部一致' if semantics_ok else '存在不一致'}。"]
    for diagnostic, row in zip(diagnostics, rows):
        if row['alignment_enabled'] and (not row['allclose'] or not diagnostic['valid_token_order_matches'] or not diagnostic['replacement_source_order_matches']):
            lines += ['', '失败定位（QK logits、有效 token 顺序、softmax）：', '```json', json.dumps(diagnostic, indent=2), '```']
    aligned_diags = [d for d in diagnostics if d['alignment_enabled']]
    lines += ['', '对齐路径诊断：' + '；'.join(
        f"{d['precision']}/{d['request']} logits最大差={d['QK_logits_max_abs']:.3e}、softmax最大差={d['softmax_max_abs']:.3e}" for d in aligned_diags) + '。', '',
        '本实验只验证给定缓存状态的共享执行功能与数值一致性，不声称 KV 复用等价于完整重算；不验证 PIM 硬件算术、舍入、性能或复用后的任务质量，不训练、不做任务准确率。', '',
        '依赖：Python ' + platform.python_version() + '；' + '；'.join(f'{name} {version}' for name, version in meta['extraction_versions'].items()) + '。',
        f"本次调用新执行模型前向 {fresh_calls} 次；提取记录中的前向总数为1。输入 token ID SHA-256：`{meta['input_ids_sha256']}`。",
        '模型配置与原生 RoPE 配对实现：[Qwen 配置](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json)、[Transformers Qwen2 实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2/modeling_qwen2.py)。', '',
        '复现（装好上述依赖后，在本目录执行；固定 CPU）：', '',
        '```bash', 'python kvchime_correctness.py', '```', '',
        '模型下载缓存可通过 `--cache-dir` 指定；已下载时可加 `--local-files-only`。可选 `--tensor-cache` 只保存/重用该一次前向的原始张量，不缓存实验结果；FP64、FP32 和唯一的关闭对齐负对照始终使用同一批提取张量。', '',
    ]
    (folder / 'correctness_report.md').write_text('\n'.join(lines))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--tensor-cache', type=Path)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--revision', default=MODEL_REVISION)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage = 'dependency import'
    rows = []
    try:
        global torch, AutoModel, AutoTokenizer, Qwen2Config, Qwen2RotaryEmbedding
        global rotate_half, apply_rotary_pos_emb
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
        import torch
        import numpy as np
        from transformers import AutoModel, AutoTokenizer, Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2RotaryEmbedding, rotate_half, apply_rotary_pos_emb)
        if hasattr(os, 'sched_getaffinity'):
            available = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, available[:args.threads])
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        stage = 'model download / one CPU forward and QKV extraction'
        data, fresh_calls = extract_once(args)
        stage = 'independent basic RoPE checks'
        rope = CheckedRoPE(data)
        with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
            rope_checks = check_rope(data, rope)
            diagnostics = []
            groups = []
            for precision, dtype, tolerance in [('FP64', torch.float64, 1e-8), ('FP32', torch.float32, 1e-4)]:
                stage = precision + ' explicit attention comparison'
                q, k, v = (data[name].to(dtype=dtype) for name in ['q', 'k', 'v'])
                reference = materialized_reference(q, k, v, rope)
                shared, requests = make_shared_state(q, k, v, rope)
                for aligned in [True, False]:
                    actual, groups = kvchime_attention(shared, requests, rope, aligned)
                    for name in ['A', 'B']:
                        row, diagnostic = compare(reference[name], actual[name], precision, name, aligned, tolerance)
                        rows.append(row)
                        diagnostics.append(diagnostic)
                        print(json.dumps(row), flush=True)
            negative_ok = True
            for bad in (r for r in rows if not r['alignment_enabled']):
                good = next(r for r in rows if r['precision'] == bad['precision'] and r['request'] == bad['request'] and r['alignment_enabled'])
                epsilon = torch.finfo(torch.float64 if bad['precision'] == 'FP64' else torch.float32).eps
                negative_ok &= not bad['allclose'] and bad['max_abs_error'] >= 100 * max(good['max_abs_error'], epsilon)
        write_results(args.output_dir, rows)
        passed = write_report(args.output_dir, data, fresh_calls, rope_checks, rows, diagnostics, groups, negative_ok)
        print('PASS' if passed else 'FAIL: inspect measured errors and diagnostics; tolerances unchanged.', flush=True)
        return 0 if passed else 1
    except Exception as error:
        write_results(args.output_dir, rows)
        (args.output_dir / 'correctness_report.md').write_text(
            '# KVChime correctness experiment: incomplete\n\n'
            f'Stage: {stage}\n\n{type(error).__name__}: {error}\n\n'
            'No success claim is made. results.csv contains only measurements actually completed. '
            'Tolerances have not been changed. See stderr for the exact traceback.\n')
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
