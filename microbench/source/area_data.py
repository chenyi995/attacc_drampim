import math
def physical_layers(components, totals, die, dies, bank_footprint, density):
    """Separate DRAM placement checks from selected logic/control components.

    The source's BANK_STORE_MM2 constant is retained byte-for-byte upstream,
    but denotes the adopted bank footprint rather than a pure cell-array
    measurement. PAPI's bank-area reference includes peripheral circuits.
    A footprint remainder is an accounting budget, not post-layout free area.
    """
    budget = die - bank_footprint
    assert budget > 0
    dram = dict(
        die_area_mm2=die, bank_footprint_mm2=bank_footprint,
        bank_footprint_pct_of_die=bank_footprint / die * 100,
        remaining_budget_mm2=budget, remaining_budget_pct_of_die=budget / die * 100,
        dram_dies_per_stack=dies, dram_density_factor=density,
        denominator_scope='Full DRAM die area; Bank and BG are additional PIM logic.',
        bank_footprint_scope='Adopted bank footprint includes arrays and bank peripheral circuits; not pure array area.',
        budget_scope='Full die minus adopted bank footprint; an accounting check, not verified post-layout spare area.',
        configurations={})
    for display, source in [('AttAcc', 'attacc'), ('KVChime', 'fugue')]:
        levels = totals['dram_equivalent', source]
        bank, bg = (levels[name] / dies / 1e6 for name in ['bank', 'bank_group'])
        total = bank + bg
        margin = budget - total
        assert total >= 0 and margin >= 0, (display, 'DRAM footprint budget')
        item = dict(bank_mm2=bank, bg_mm2=bg, pim_total_mm2=total,
                    bank_pct_of_die=bank / die * 100, bg_pct_of_die=bg / die * 100,
                    pim_pct_of_die=total / die * 100,
                    pim_pct_of_remaining_budget=total / budget * 100,
                    remaining_margin_mm2=margin, remaining_margin_pct_of_die=margin / die * 100,
                    accounted_die_pct=(bank_footprint + total) / die * 100,
                    within_remaining_budget=margin >= 0)
        assert math.isclose(item['bank_pct_of_die'] + item['bg_pct_of_die'], item['pim_pct_of_die'])
        assert math.isclose(bank_footprint + total + margin, die)
        dram['configurations'][display] = item
    base = dram['configurations']['AttAcc']['pim_total_mm2']
    final = dram['configurations']['KVChime']['pim_total_mm2']
    dram.update(increment_pim_mm2=final - base,
                increment_pim_pct_of_die=(final - base) / die * 100,
                relative_pim_growth_pct=(final / base - 1) * 100)

    groups = [
        ('softmax_and_buffers', 'Softmax and buffers', ['sfmarray_attacc_p769', 'sfmarray_fugue_p769']),
        ('channel_accumulation', 'Channel accumulation', ['acclogic_p1501']),
        ('replacement_decoder', 'Replacement decoder', ['diffdec_p1501']),
        ('causal_comparison', 'Causal comparison', ['causal_p1501']),
        ('controller', 'Controller', ['ctrl_attacc_p1501', 'ctrl_fugue_p1501']),
    ]
    by_tag = {row['tag']: row for row in components}
    selected_tags = {row['tag'] for row in components if row['level'] in ['logic_die', 'hbm_controller']}
    group_tags = [tag for _, _, tags in groups for tag in tags]
    assert len(group_tags) == len(set(group_tags)) and set(group_tags) == selected_tags
    # One integrated softmax array includes all of its SRAM buffers. Do not
    # add a standalone buffer/TLB reference or multiply by the DRAM-die count.
    buffer_totals = {
        source: (totals['asap7_raw', source]['logic_die'] +
                 totals['asap7_raw', source]['hbm_controller']) / 1e6
        for source in ['attacc', 'fugue']}
    native, current = buffer_totals['attacc'], buffer_totals['fugue']
    parts = []
    for identifier, label, tags in groups:
        values = {}
        for display, source in [('attacc', 'attacc'), ('kvchime', 'fugue')]:
            values[display + '_mm2'] = sum(
                float(by_tag[tag]['area_um2']) * int(by_tag[tag]['count_per_stack']) / 1e6
                for tag in tags if source in by_tag[tag]['used_by'].split('+'))
        before, after = values['attacc_mm2'], values['kvchime_mm2']
        parts.append(dict(id=identifier, label=label, source_tags=tags, **values,
                          added_mm2=after - before,
                          growth_pct=(after / before - 1) * 100 if before else None,
                          newly_added=before == 0,
                          pct_of_current_total=after / current * 100,
                          added_pct_of_current_total=(after - before) / current * 100,
                          attacc_pct_of_attacc_total=before / native * 100))
    assert math.isclose(sum(part['attacc_mm2'] for part in parts), native)
    assert math.isclose(sum(part['kvchime_mm2'] for part in parts), current)
    assert math.isclose(sum(part['pct_of_current_total'] for part in parts), 100)
    assert math.isclose(sum(part['added_mm2'] for part in parts), current - native)
    buffer = dict(
        physical_layer='HBM buffer die; logic die is the same physical layer in the retained source accounting.',
        display_scope='Selected logic and control components',
        denominator_scope='Sum of selected KVChime logic_die and hbm_controller components after additions; not the complete physical buffer die.',
        attacc_total_mm2=native, kvchime_total_mm2=current,
        increment_mm2=current - native, growth_pct=(current / native - 1) * 100,
        kvchime_over_attacc_ratio=current / native, components=parts,
        logic_subtotal_mm2=totals['asap7_raw', 'fugue']['logic_die'] / 1e6,
        controller_mm2=totals['asap7_raw', 'fugue']['hbm_controller'] / 1e6,
        controller_counted_separately_once=True, softmax_buffers_already_included=True,
        excludes='Unmodeled PHY, whole-die infrastructure, routing and packaging.',
        physical_mapping_source='fig/plots/overhead/data_overhead.csv header',
        density_conversion_applied=False, divide_by_dram_die_count=False)
    buffer.update(logic_attacc_mm2=totals['asap7_raw', 'attacc']['logic_die']/1e6,logic_kvchime_mm2=totals['asap7_raw', 'fugue']['logic_die']/1e6,logic_increment_mm2=(totals['asap7_raw', 'fugue']['logic_die']-totals['asap7_raw', 'attacc']['logic_die'])/1e6,controller_attacc_mm2=totals['asap7_raw', 'attacc']['hbm_controller']/1e6,controller_increment_mm2=(totals['asap7_raw', 'fugue']['hbm_controller']-totals['asap7_raw', 'attacc']['hbm_controller'])/1e6)
    assert math.isclose(buffer['logic_increment_mm2']+buffer['controller_increment_mm2'],buffer['increment_mm2'])
    return dict(dram_die=dram, buffer_die=buffer)

