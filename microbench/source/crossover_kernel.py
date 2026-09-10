"""Unchanged final crossover grouping and complete service equations."""

def keys_for(work, model, cap):
    keys = set()
    for p in work['panels']:
        for q in p['query_samples']:
            count = q * model.gqa
            residents = {1} if p['mode'] == 'plain' else {cap} | ({count % cap} if count % cap else set())
            keys.update((p['cached'] + q, model.h, r) for r in residents)
    return sorted(keys)


def service(p, q, model, profiles, windows, Model, cap, tck):
    c, bw, mode = p['cached'], p['link_GBps'], p['mode']; n = c + q
    count = q * model.gqa
    residents = [1] * count if mode == 'plain' else [cap] * (count // cap) + ([count % cap] if count % cap else [])
    scan = Model.combine([profiles[n, model.h, resident] for resident in residents])
    window = 0 if not c else min(clock for index, clock in windows[n, model.h, residents[0]] if index >= 2 * (c // 16)) * tck / 1000
    gt, _ = model.gpu_attention(q, n, 1); sm, _ = model.softmax(q, n, 1)
    qi = q * model.qbytes / bw / 1000
    kv = q * model.kvbytes / bw / 1000
    fetch = c * model.kvbytes / bw / 1000
    gpu = max(gt + fetch, kv)
    service = 2 * qi + scan['scan_us'] + sm + max(0, kv - window)
    row = dict(panel=p['panel'], model=model.name, GPUs=model.tp, Q_heads=model.m['num_heads'],
               KV_heads=model.m['num_kv_heads'], gqa_size=model.gqa, mode=mode, q=q, cached=c, total_kv=n, link_GBps=bw,
               gpu_attention_us=gt, pim_softmax_us=sm, q_input_us=qi, output_us=qi,
               q_input_bytes=q * model.qbytes, new_kv_bytes=q * model.kvbytes, cached_kv_bytes=c * model.kvbytes,
               cached_kv_readback_us=fetch, new_kv_transfer_us=kv, gpu_service_us=gpu,
               pim_scan_us=scan['scan_us'], pim_service_us=service, pim_new_kv_exposed_us=max(0, kv - window),
               pim_overlap_window_us=window, pim_speedup_over_GPU=gpu / service,
               pim_profile_repeats=scan['profile_repeats'], GPU_minus_PIM_us=gpu - service,
               GPU_query_tokens_per_us=q / gpu, PIM_query_tokens_per_us=q / service,
               source_kind='integer-refinement', numerical_sample=True)
    return row
