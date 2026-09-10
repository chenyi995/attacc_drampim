"""Exact original source-based frequency derivation, with caller-provided roots."""
import ast
from fractions import Fraction
import math
FILES = ('fugue/attention.py', 'src/config.py')
def expression(node, context, source):
    """Evaluate the limited arithmetic/dictionary AST used by these inputs."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return Fraction(ast.get_source_segment(source, node))
        return node.value
    if isinstance(node, ast.Name):
        return context[node.id]
    if isinstance(node, ast.Attribute):
        return expression(node.value, context, source) + '.' + node.attr
    if isinstance(node, ast.Dict):
        return {expression(k, context, source): expression(v, context, source)
                for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [expression(v, context, source) for v in node.elts]
    if isinstance(node, ast.Subscript):
        return expression(node.value, context, source)[expression(node.slice, context, source)]
    if isinstance(node, ast.BinOp):
        left, right = (expression(x, context, source) for x in (node.left, node.right))
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if isinstance(node, ast.Call):
        assert not node.keywords
        func = expression(node.func, context, source)
        args = [expression(x, context, source) for x in node.args]
        if func == 'max':
            return max(args)
        if func == 'math.ceil':
            assert len(args) == 1
            return math.ceil(args[0])
    raise ValueError('Unsupported source expression: ' + ast.dump(node))


def assign(target, value, context, source):
    if isinstance(target, ast.Name):
        context[target.id] = value
    elif isinstance(target, ast.Subscript):
        expression(target.value, context, source)[expression(target.slice, context, source)] = value
    else:
        raise ValueError('Unsupported source assignment: ' + ast.dump(target))


def root_name(node):
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def calculate():
    """Return exact-input derivation plus a staircase curve ready for plotting.

    Each query MAC means one vector-MAC issue for a resident query per bank.
    ``relative_throughput`` is normalized to this model's own plateau. It is
    not a TTFT, TBT, attention-service, or native-AttAcc speedup.
    """
    config = (SNAPSHOT / FILES[1]).read_text()
    attention = (SNAPSHOT / FILES[0]).read_text()
    context = {'PIMType': 'PIMType', 'max': 'max', 'math': 'math'}
    for node in ast.parse(config).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if root_name(target) == 'ENERGY_TABLE':
                assign(target, expression(node.value, context, config), context, config)
    tree = ast.parse(attention)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ('TCK', 'CAP', 'E_COL', 'E_OP'):
                assign(target, expression(node.value, context, attention), context, attention)
    interval = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'interval')
    result = next(n for n in interval.body if isinstance(n, ast.Return)).value
    # Fail on a semantic change to the supported timing/energy equation.
    expected = ast.parse('max(6, r, math.ceil(6*(E_COL+r*E_OP)/(E_COL+E_OP)))', mode='eval').body
    assert isinstance(result, ast.Call) and len(result.args) == len(expected.args)
    floor = expression(result.args[0], context, attention)
    expected.args[0] = result.args[0]
    expected.args[2].args[0].left.left = result.args[0]
    assert ast.dump(result) == ast.dump(expected), 'MQ interval source equation changed'
    tck, count, col, op = [context[k] for k in ('TCK', 'CAP', 'E_COL', 'E_OP')]
    assert count.denominator == 1 and floor.denominator == 1
    count, floor = int(count), int(floor)
    command_energy = col + count * op
    budget_energy = col + op
    power_interval = math.ceil(floor * command_energy / budget_energy)
    plateau_interval = max(floor, power_interval)
    frequency = Fraction(count) / (plateau_interval * tck)
    throughput = Fraction(count) / (plateau_interval * tck)
    intervals = [int(expression(result, dict(context, r=Fraction(r)), attention))
                 for r in range(1, count + 1)]
    assert intervals[-1] == plateau_interval
    assert math.ceil(Fraction(count) / (frequency * tck)) == plateau_interval
    # Plot range and sampling are rendering choices, not experimental data.
    low, high = frequency / 4, frequency * 2
    points = {low + (high - low) * Fraction(i, 600) for i in range(601)}
    points.add(frequency)
    for cycles in range(floor, math.ceil(Fraction(count) / (low * tck)) + 1):
        crossing = Fraction(count) / (cycles * tck)
        if low <= crossing <= high:
            points.add(crossing)
    curve = []
    for f in sorted(points):
        compute = math.ceil(Fraction(count) / (f * tck))
        effective = max(floor, power_interval, compute)
        rate = Fraction(count) / (effective * tck)
        curve.append(dict(frequency_ghz=float(f), compute_interval_tck=compute,
                          effective_interval_tck=effective,
                          effective_query_mac_per_ns=float(rate),
                          relative_throughput=float(rate / throughput)))
    assert all(a['effective_query_mac_per_ns'] <= b['effective_query_mac_per_ns']
               for a, b in zip(curve, curve[1:]))
    assert all(p['relative_throughput'] == 1.0 for p in curve
               if p['frequency_ghz'] >= float(frequency))
    return dict(
        evidence_kind='Model-derived balance under frequency-independent operation energy; not an RTL sweep.',
        inputs=dict(tck_ns=float(tck), query_capacity=count, dram_floor_tck=floor,
                    column_energy_pj=float(col), operation_energy_pj=float(op),
                    exact=dict(tck_ns=str(tck), column_energy_pj=str(col), operation_energy_pj=str(op)),
                    configured_query_intervals_tck=intervals),
        balance=dict(frequency_ghz=float(frequency), exact_frequency_ghz=str(frequency),
                     power_interval_tck=power_interval, effective_interval_tck=plateau_interval,
                     effective_query_mac_per_ns=float(throughput),
                     power_budget_mw_per_bank=float(budget_energy / (floor * tck)),
                     modeled_power_mw_per_bank=float(command_energy / (plateau_interval * tck))),
        curve=curve, sources=source_records(),
        limitations=[
            'Frequency is varied analytically; no synthesis or simulator runs are launched.',
            'Column and MAC/operand-read energy are fixed at the source energy-table values.',
            'One vector-MAC issue per PE cycle is the throughput assumption of the source design point.',
            'No frequency-dependent capacitance, voltage, leakage, timing closure, or PVT margin is inferred.',
            'Command throughput is not complete attention service, TTFT, TBT, E2E, or workload makespan.',
        ])

