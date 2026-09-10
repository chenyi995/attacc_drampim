#!/usr/bin/env python3
"""Plot Readers 4 with a proportion-preserving Full GPU schematic or 100% stacks."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent


def load_rows(spec):
    schema = json.loads((HERE / "columns.json").read_text(encoding="utf-8"))
    with (HERE / "data.csv").open(newline="", encoding="utf-8") as stream:
        source = [{key: None if val == "" else json.loads(val)
                   if schema[key] in ("int", "float", "dict", "list", "bool") else val
                   for key, val in row.items()} for row in csv.DictReader(stream)]
    selected = [row for row in source if row["model"] == spec["model"]
                and row["phase"] == spec["phase"] and row["display"] == spec["workload"]
                and row["variant"] in spec["variants"]]
    lookup = {row["variant"]: row for row in selected}
    assert len(selected) == len(spec["variants"]) == len(lookup)
    assert set(lookup) == set(spec["variants"])
    for row in selected:
        assert row["eligible"] and not row["invalid_reasons"]
        parts = [row[s["field"]] for s in spec["segments"]]
        assert all(math.isfinite(value) and value >= 0 for value in parts)
        assert row["source_field"] == "TTFT_ms" and row["total_ms"] > 0
        assert math.isclose(math.fsum(parts), row["total_ms"], rel_tol=1e-10)
        assert math.isclose(row["memory_ms"], row["memory_only_ms"] + row["memory_compute_overlap_ms"], rel_tol=1e-10)
    return lookup, len(source)


def load_annotation_evidence(spec, lookup):
    source = HERE / "event-blocks.csv"
    reuse = lookup["F1"]
    with source.open(newline="", encoding="utf-8") as stream:
        blocks = [row for row in csv.DictReader(stream) if row["model"] == spec["model"]
                  and row["phase"] == spec["phase"] and row["variant"] == "F1"
                  and row["workload_id"] == reuse["workload_id"]]
    assert blocks
    block_ids = [int(row["id"]) for row in blocks]
    assert set(block_ids) == set(reuse["source_refs"]["block_ids"])
    def phase_ms(field):
        return math.fsum(float(row[field]) * int(row["repetitions"]) / 1000 for row in blocks)
    memory_ms = phase_ms("memory_only_us_per_layer")
    assert math.isclose(memory_ms, reuse["memory_only_ms"], rel_tol=1e-10)
    readback_ms = phase_ms("external_exposed_us_per_layer.KV_readback")
    write_ms = phase_ms("external_exposed_us_per_layer.exposed_KV_write")
    movement_ms = readback_ms + write_ms
    assert 0 <= movement_ms <= memory_ms
    share = movement_ms / reuse["total_ms"] * 100
    assert share > 50
    return dict(source_file=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                block_ids=block_ids, variant="F1", workload=spec["workload"],
                exposed_KV_readback_ms=readback_ms, exposed_KV_write_ms=write_ms,
                exposed_KV_movement_ms=movement_ms, total_TTFT_ms=reuse["total_ms"],
                share_of_total_TTFT_pct=share, uncovered_memory_ms=memory_ms,
                formula="sum(external_exposed KV_readback + exposed_KV_write, weighted by repetitions) / TTFT")


def draw(spec, lookup, evidence, mode, path):
    from figure_style import configure, style_axis
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch, Rectangle, FancyArrowPatch
    from matplotlib.text import Text

    configure(spec)
    normalized = mode == "normalized"
    fonts = spec["fonts_pt"]
    fig = plt.figure(figsize=(spec["width_inches"], spec["height_inches"]))
    fig_width, fig_height = spec["width_inches"] * 72, spec["height_inches"] * 72
    labels = []
    width = spec["bar_width_pt"]
    step = width + spec["bar_gap_pt"]
    xs = [i * step for i in range(len(spec["variants"]))]
    xmin = -width / 2 - spec["axis_inner_margin_pt"]
    xmax = xs[-1] + width / 2 + spec["axis_inner_margin_pt"]
    axis_width = xmax - xmin
    ax = fig.add_axes([spec["axis_left_pt"] / fig_width, spec["axis_bottom_pt"] / fig_height,
                       axis_width / fig_width, spec["axis_height_pt"] / fig_height])
    style_axis(ax)
    cap = 100 if normalized else spec["capped_ylim_ms"]
    ax.set_ylim(0, cap)
    ax.set_xlim(xmin, xmax)
    ax.set_yticks([i * cap / 4 for i in range(5)])
    ax.set_ylabel("Share of prefill TTFT (%)" if normalized else "Prefill latency (ms)", labelpad=5)
    tick_labels = list(spec["variant_tick_labels"])
    ax.set_xticks(xs, tick_labels)
    ax.tick_params(axis="x", pad=4)
    from matplotlib.transforms import ScaledTranslation
    for label, offset in zip(ax.get_xticklabels(), spec.get("tick_label_offsets_pt", [0] * len(xs))):
        label.set_transform(label.get_transform() + ScaledTranslation(offset / 72, 0, fig.dpi_scale_trans))
    labels.extend(ax.get_xticklabels())
    labels.append(ax.yaxis.label)
    by_field = {s["field"]: s for s in spec["segments"]}
    handles = [Patch(facecolor=by_field[k]["color"], edgecolor="#222222", hatch=by_field[k]["hatch"],
                     linewidth=0.5, label=by_field[k]["label"]) for k in spec["legend_order"]]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, spec["legend_top_fraction"]),
                        ncol=spec.get("legend_columns", 3), fontsize=fonts["legend"], frameon=False, borderaxespad=0,
                        handlelength=0.9, handletextpad=0.3, labelspacing=spec.get("legend_labelspacing", 0.45), columnspacing=0.6)
    labels.extend(legend.get_texts())

    geometry, totals = [], []
    for variant, x in zip(spec["variants"], xs):
        row = lookup[variant]
        total = row["total_ms"]
        schematic = not normalized and variant == "F0" and total > cap
        assert normalized or schematic or total <= cap
        height = 100 if normalized else cap if schematic else total
        scale = height / total
        raw_bottom = 0.0
        for segment in spec["segments"]:
            raw_ms = row[segment["field"]]
            raw_top = raw_bottom + raw_ms
            percentage = raw_ms / total * 100
            shown_bottom, shown_top = raw_bottom * scale, raw_top * scale
            shown = shown_top - shown_bottom
            if shown > 0:
                ax.add_patch(Rectangle((x - width / 2, shown_bottom), width, shown,
                    facecolor=segment["color"], edgecolor="#202020", hatch=segment["hatch"], linewidth=0.45, zorder=3))
            shown_height_pt = shown / cap * spec["axis_height_pt"]
            if percentage >= spec["percent_label_min"] and shown_height_pt >= spec["percent_label_min_height_pt"]:
                labels.append(ax.text(x, shown_bottom + shown / 2, f"{percentage:.1f}%", ha="center", va="center",
                    color=segment["text_color"], fontsize=fonts["note"], zorder=5,
                    bbox={"facecolor": segment["color"], "edgecolor": "none", "pad": 0.25}))
            geometry.append(dict(variant=variant, field=segment["field"], raw_ms=raw_ms,
                percentage=percentage, denominator_ms=total, raw_start_ms=raw_bottom, raw_end_ms=raw_top,
                shown_start=shown_bottom, shown_height=shown,
                shown_unit="percent" if normalized else "schematic display units" if schematic else "ms",
                schematic=schematic, display_scale_per_ms=scale, omitted_above_cap_ms=0.0))
            raw_bottom = raw_top
        assert math.isclose(raw_bottom, total, rel_tol=1e-10)
        label = f"{total:,.1f} ms"
        labels.append(ax.annotate(label, (x, height), xytext=(0, 4), textcoords="offset points",
                                  ha="center", va="bottom", fontsize=fonts["note"], annotation_clip=False,
                                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4}))
        totals.append(dict(variant=variant, total_ms=total, label=label, clipped=False,
                           schematic=schematic, display_scale_per_ms=scale, shown_total=height))

    # Only the over-height Full GPU bar has an omitted span; the other two
    # bars and the axis retain their existing linear millisecond scale.
    from matplotlib.lines import Line2D
    bar_breaks = []
    for total, x in zip(totals, xs):
        if not total["schematic"]:
            continue
        centre = cap * spec["bar_break_fraction"]
        amp = cap * spec["bar_break_amplitude_fraction"]
        break_x = [x - width / 2 + width * k / 6 for k in range(7)]
        break_y = [centre + (amp if k % 2 else -amp) for k in range(7)]
        ax.add_line(Line2D(break_x, break_y, color="white", linewidth=3.2, zorder=8))
        ax.add_line(Line2D(break_x, break_y, color="#202020", linewidth=.7, zorder=9))
        bar_breaks.append(dict(variant=total["variant"], actual_total_ms=total["total_ms"],
                               x=break_x, y=break_y))

    callouts = []
    if not normalized:
        compute_color = spec["annotation_colors"]["compute"]
        memory_color = spec["annotation_colors"]["memory"]
        memory_text = spec["annotations"]["memory"].format(kv_pct=evidence["share_of_total_TTFT_pct"])
        reuse_compute = next(r for r in geometry if r["variant"] == "F1" and r["field"] == "compute_ms")
        full_compute = next(r for r in geometry if r["variant"] == "F0" and r["field"] == "compute_ms")
        full_compute_top = full_compute["shown_start"] + full_compute["shown_height"]
        reuse_compute_tip = reuse_compute["shown_start"] + reuse_compute["shown_height"] * 0.70
        arrow_start = (xs[0] + width / 2, full_compute_top - cap * 0.015)
        arrow_end = (xs[1] - width / 2, reuse_compute_tip)
        ax.add_patch(FancyArrowPatch(arrow_start, arrow_end, arrowstyle="-|>", mutation_scale=8,
                     connectionstyle="arc3,rad=-0.08", color=compute_color, linewidth=0.8,
                     shrinkA=0, shrinkB=0, zorder=7))
        compute_text_position = tuple(spec["annotation_positions"]["recomputation"])
        compute_callout = ax.text(*compute_text_position, spec["annotations"]["recomputation"],
                                 ha="center", va="bottom", fontsize=fonts["note"],
                                 color=compute_color, linespacing=1.05, zorder=7, clip_on=False)
        reuse_memory = next(r for r in geometry if r["variant"] == "F1" and r["field"] == "memory_only_ms")
        memory_target = reuse_memory["shown_start"] + reuse_memory["shown_height"] * 0.75
        memory_callout = ax.annotate(memory_text,
            xy=(xs[1] + width / 2, memory_target), xytext=tuple(spec["annotation_positions"]["memory"]),
            xycoords="data", textcoords="data", ha="left", va="center",
            fontsize=fonts["note"], color=memory_color, linespacing=1.05,
            arrowprops={"arrowstyle": "->", "color": memory_color, "lw": 0.8,
                        "connectionstyle": "arc3,rad=0.15", "shrinkA": 3, "shrinkB": 0},
            annotation_clip=False, zorder=7)
        labels.extend([compute_callout, memory_callout])
        callouts = [dict(text=spec["annotations"]["recomputation"], source_variant="F0", source_field="compute_ms",
                         target_variant="F1", target_field="compute_ms", arrow_start=arrow_start, arrow_end=arrow_end,
                         text_position=compute_text_position),
                    dict(text=memory_text, target_variant="F1", target_field="memory_only_ms",
                         evidence_field="exposed_KV_movement_ms")]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    physical_width = ax.get_window_extent(renderer).width / fig.dpi * 72 * width / (xmax - xmin)
    assert math.isclose(physical_width, width, rel_tol=1e-12)
    texts = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, (artist.get_text(), box)
        texts.append(dict(text=artist.get_text(), bbox_px=list(box.bounds)))
    boxes = [Text.get_window_extent(artist, renderer) for artist in labels]
    collisions = [(labels[i].get_text(), labels[j].get_text()) for i, a in enumerate(boxes)
                  for j, b in enumerate(boxes) if i < j and a.overlaps(b)]
    assert not collisions, ("Labels overlap", collisions)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Title": spec["title"] + " / " + mode, "Creator": "CSV / Matplotlib",
                               "CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return dict(mode=mode, workload=spec["workload"], phase=spec["phase"], geometry=geometry, totals=totals,
                y_limit=cap, y_unit="percent" if normalized else "ms (Full GPU uses schematic height)",
                axis_count=1, displayed_bars=len(xs), legend_position="above axes",
                legend_columns=spec.get("legend_columns", 3),
                width_inches=spec["width_inches"], height_inches=spec["height_inches"],
                stack_order=[s["field"] for s in spec["segments"]],
                legend_labels=[by_field[k]["label"] for k in spec["legend_order"]],
                footer_present=False, tick_configuration_labels=tick_labels, total_label_unit="ms",
                panel_label="", bar_breaks=bar_breaks, axis_break=False,
                title_present=False, subtitle_present=False,
                palette={s["label"]:s["color"] for s in spec["segments"]},
                x_axis_title="", callouts=callouts, callout_evidence=evidence,
                normalization="Each segment / its own bar's total TTFT * 100" if normalized else
                              "Full GPU components / its own TTFT * display cap; other bars in milliseconds",
                all_segments_preserved=True, percent_labels_use_own_TTFT=True,
                overlap_counted_once=True, raw_data_unchanged=True, new_experiments=False,
                bar_width_pt=physical_width, bar_gap_pt=spec["bar_gap_pt"], texts=texts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--mode", choices=("capped-100ms", "normalized"), default="capped-100ms",
                        help="Default: the Readers 4 100 ms capped display; normalized is an optional comparison.")
    args = parser.parse_args()
    spec = json.loads((HERE / "template.json").read_text(encoding="utf-8"))
    lookup, source_count = load_rows(spec)
    evidence = load_annotation_evidence(spec, lookup)
    source_hash = hashlib.sha256((HERE / "data.csv").read_bytes()).hexdigest()
    outputs = []
    for mode in (args.mode,):
        path = args.output_dir / f"figure1-readers4-{mode}.pdf"
        audit = draw(spec, lookup, evidence, mode, path)
        audit.update(source_csv_sha256=source_hash, original_row_count=source_count, plotted_row_count=len(lookup))
        path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        outputs.append(str(path.resolve()))
    fields = ["workload", "variant", "TTFT_ms"] + [s["field"] for s in spec["segments"]] + [s["field"] + "_pct" for s in spec["segments"]]
    with (args.output_dir / "figure1-readers4.values.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant in spec["variants"]:
            row = lookup[variant]
            writer.writerow(dict(workload=spec["workload"], variant=variant, TTFT_ms=row["total_ms"],
                **{s["field"]: row[s["field"]] for s in spec["segments"]},
                **{s["field"] + "_pct": row[s["field"]] / row["total_ms"] * 100 for s in spec["segments"]}))
    assert source_hash == hashlib.sha256((HERE / "data.csv").read_bytes()).hexdigest()
    print(json.dumps(dict(outputs=outputs, source_data_unchanged=True, checks="passed")))


if __name__ == "__main__":
    main()
