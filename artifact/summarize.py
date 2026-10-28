#!/usr/bin/env python3
"""Summarize MCCL artifact runs produced by artifact/run.sh.

Usage:  python3 artifact/summarize.py <results dir> [--plots]

Looks for E1-equivalence/, E2-single/ and E3-e2e/results.csv under the given directory, prints
Markdown tables, compares simulated times with artifact/expected/, and writes <dir>/summary.md.
With --plots (requires matplotlib) it also writes PNG bar charts next to summary.md.
Only the Python standard library is needed otherwise.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

FAILURE_ORDER = ["no_failure", "failure_1", "failure_2", "failure_3"]


def forder(f):
    return FAILURE_ORDER.index(f) if f in FAILURE_ORDER else len(FAILURE_ORDER)


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def ns(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def size_of(workload):
    m = re.search(r"(\d+)_bytes", workload)
    return int(m.group(1)) if m else None


def human(nbytes):
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if nbytes >= div:
            return f"{nbytes / div:g} {unit}"
    return f"{nbytes} B"


def summarize_e1(path):
    rows = read_csv(path)
    table = [[r["workload"], r["failure_type"], r["nccl_msccl_ns"], r["simai_native_ns"], r["status"]]
             for r in rows]
    passed = sum(r["status"] == "PASS" for r in rows)
    text = ["## E1 - Custom-schedule path vs SimAI native NCCL model",
            "",
            "NCCL-equivalent MSCCL schedules executed by `mscclCustomFlow` must finish at exactly the "
            "same simulated time as SimAI's native `ncclFlowModel` (paper Sec. 2.3: custom ring "
            "schedule reproduces native ring completion time).",
            "",
            md_table(["workload", "failure", "MSCCL path (ns)", "native (ns)", "verdict"], table),
            "",
            f"**{passed}/{len(rows)} pairs identical.**", ""]
    return "\n".join(text), passed == len(rows)


def summarize_e2(path, plots, outdir):
    rows = read_csv(path)
    by = defaultdict(dict)
    for r in rows:
        by[(r["workload"], r["algorithm"])][r["failure_type"]] = r
    table = []
    ok = True
    for (wl, alg), fs in sorted(by.items()):
        base = ns(fs.get("no_failure", {}).get("sim_time_ns"))
        size = size_of(wl)
        for ft in sorted(fs, key=forder):
            r = fs[ft]
            t = ns(r["sim_time_ns"])
            ok &= r["status"] == "SUCCESS"
            algbw = f"{size / t:.2f}" if (t and size) else "-"  # bytes/ns == GB/s
            slow = f"{t / base:.2f}x" if (t and base) else "-"
            table.append([wl.split("-")[0].upper(), human(size) if size else "?", alg, ft,
                          f"{t / 1e6:.3f}" if t else "FAILED", algbw, slow, r["schedule"]])
    text = ["## E2 - Single collectives on 16 GPUs (2 hosts x 8, rail-optimized) under gray failures",
            "",
            "`nccl` = NCCL-equivalent MSCCL schedules through the custom-schedule path "
            "(the configuration used for all MCCL experiments in the paper); `simai_native` = SimAI's "
            "built-in NCCL flow model. AlgBW = collective size / completion time (GB/s). "
            "Slowdown is relative to the same workload and algorithm with no failure: static "
            "schedules collapse to the speed of the degraded link (the \"Orig\"/\"No reopt\" "
            "baselines of Fig. 1 and Fig. 8).",
            "",
            md_table(["coll", "size", "algorithm", "failure", "time (ms)", "AlgBW (GB/s)",
                      "slowdown vs no failure", "schedule used"], table), ""]
    if plots:
        _plot_e2(by, outdir)
        text += ["![E2](e2_single.png)", ""]
    return "\n".join(text), ok


def _plot_e2(by, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(by)
    fails = sorted({f for fs in by.values() for f in fs}, key=forder)
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / max(1, len(fails))
    for i, ft in enumerate(fails):
        vals = []
        for k in keys:
            t = ns(by[k].get(ft, {}).get("sim_time_ns"))
            s = size_of(k[0])
            vals.append(s / t if (t and s) else 0)
        ax.bar([x + i * w for x in range(len(keys))], vals, w, label=ft)
    ax.set_xticks([x + w * (len(fails) - 1) / 2 for x in range(len(keys))])
    ax.set_xticklabels([f"{k[0].split('-')[0]} {human(size_of(k[0]))}\n{k[1]}" for k in keys], fontsize=8)
    ax.set_ylabel("Algorithm bandwidth (GB/s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "e2_single.png", dpi=150)


# Full-batch reconstruction, identical to benchmark/analyze_results.py (paper Sec. 6.3):
#   T_batch = T_transformer * TRANSFORMERS_PER_STAGE * BATCH_SIZE / (num_gpus / TP) + T_grad
E3_TP, E3_TRANSFORMERS_PER_STAGE, E3_BATCH_SIZE, E3_GPUS = 1, 3, 16384, 128
E3_SCALE = E3_TRANSFORMERS_PER_STAGE * E3_BATCH_SIZE / (E3_GPUS / E3_TP)


def summarize_e3(path, plots, outdir):
    rows = read_csv(path)
    by = defaultdict(dict)
    ok = True
    for r in rows:
        by[r["failure_type"]][r["workload"]] = r
        ok &= r["status"] == "SUCCESS"
    fails = sorted(by, key=forder)
    totals, table = {}, []
    for ft in fails:
        tr = by[ft].get("DeepSeek671B-transformer", {})
        gr = by[ft].get("DeepSeek671B-grad", {})
        tt, tg = ns(tr.get("sim_time_ns")), ns(gr.get("sim_time_ns"))
        totals[ft] = int(tt * E3_SCALE) + tg if (tt and tg) else None
        table.append([ft,
                      f"{tt / 1e6:.3f}" if tt else tr.get("status", "-"),
                      f"{tg / 1e6:.3f}" if tg else gr.get("status", "-"),
                      f"{totals[ft] / 1e9:.2f}" if totals[ft] else "-"])
    base = totals.get("no_failure")
    for row, ft in zip(table, fails):
        row.append(f"{totals[ft] / base:.2f}x" if (totals[ft] and base) else "-")
    text = ["## E3 - DeepSeek-671B training under gray failures (one pipeline stage, 128 GPUs)",
            "",
            "One of the 16 pipeline stages (1 segment x 16 hosts x 8 GPUs, HPN; equivalent to a "
            "2048-GPU job), with the NCCL-equivalent schedules through the custom-schedule path "
            "(NCCL baseline of Fig. 11 and Fig. 17). The training time on one 16K-token batch is "
            "reconstructed from the two simulated phases as in `benchmark/analyze_results.py`: "
            f"transformer pass x {E3_TRANSFORMERS_PER_STAGE} transformer blocks per stage x "
            f"{E3_BATCH_SIZE} / ({E3_GPUS} GPUs / TP {E3_TP}) + gradient phase.",
            "",
            md_table(["failure", "transformer pass (ms)", "gradient phase (ms)",
                      "training time on one batch (s)", "vs no failure"], table),
            ""]
    if plots and any(totals.values()):
        _plot_e3(totals, outdir)
        text += ["![E3](e3_e2e.png)", ""]
    return "\n".join(text), ok


def _plot_e3(totals, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fails = sorted(totals, key=forder)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(fails, [(totals[f] or 0) / 1e9 for f in fails])
    ax.set_ylabel("Training time on one batch (s)")
    ax.set_title("DeepSeek-671B, 128 GPUs, NCCL schedules")
    fig.tight_layout()
    fig.savefig(outdir / "e3_e2e.png", dpi=150)


EXPECTED_DIR = Path(__file__).resolve().parent / "expected"
TIME_COLS = ("sim_time_ns", "nccl_msccl_ns", "simai_native_ns")


def compare_with_reference(path, ref_path):
    """Compare simulated times with the authors' reference run (the simulator is deterministic)."""
    key_cols = ("workload", "failure_type", "algorithm")
    def index(rows):
        return {tuple(r.get(k, "") for k in key_cols): r for r in rows}
    got, ref = index(read_csv(path)), index(read_csv(ref_path))
    same, diff, missing = 0, [], 0
    for k, r in got.items():
        if k not in ref:
            missing += 1
            continue
        cols = [c for c in TIME_COLS if c in r and c in ref[k]]
        if all(r[c] == ref[k][c] for c in cols):
            same += 1
        else:
            diff.append(" / ".join(x for x in k if x) + ": " +
                        ", ".join(f"{c} {r[c] or '-'} vs ref {ref[k][c]}" for c in cols if r[c] != ref[k][c]))
    lines = [f"**Reference check** (`artifact/expected/{ref_path.name}`): "
             f"{same}/{len(got)} runs identical to the authors' reference simulated times"
             + (f", {missing} not in reference" if missing else "") + "."]
    if diff:
        lines += ["", "Differences (expected only if inputs, commits or the build profile changed):", ""]
        lines += [f"- {d}" for d in diff]
    return "\n".join(lines) + "\n"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    plots = "--plots" in sys.argv
    if len(args) != 1:
        print(__doc__)
        return 2
    root = Path(args[0])
    sections, all_ok, found = [f"# MCCL artifact results: `{root.name}`", ""], True, False
    for name, fn in (("E1-equivalence", None), ("E2-single", summarize_e2), ("E3-e2e", summarize_e3)):
        p = root / name / "results.csv"
        if not p.exists():
            continue
        found = True
        text, ok = summarize_e1(p) if fn is None else fn(p, plots, root)
        sections.append(text)
        all_ok &= ok
        ref = EXPECTED_DIR / f"{name}.csv"
        if ref.exists():
            sections.append(compare_with_reference(p, ref))
    if not found:
        print(f"No E1-equivalence/E2-single/E3-e2e results.csv under {root}", file=sys.stderr)
        return 2
    report = "\n".join(sections)
    (root / "summary.md").write_text(report + "\n")
    print(report)
    print(f"\nWritten: {root / 'summary.md'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
