"""Cross-study synthesis of MathMage's results (read-only; uses saved summaries only).

    python analysis/synthesis.py            # publication figures
    python analysis/synthesis.py --draft    # fast previews

Reads results/tables/summary.json (v3), v4_summary.json (pre-registered v4) and v5_summary.json
(pre-registered v5) and writes:
  tables/synthesis.md          gains per evaluation set and baseline family, cost dominance, per-benchmark
                               consistency, selection/computation/coverage decomposition
  figures/synthesis_forest     MathMage at the frozen operating point (dev mu*) against every baseline
                               family on every evaluation set, with 95% CIs
The operating point is mu*, the deployed setting fixed on development data; where a pre-registered
contrast used the maximal run instead, it is labelled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mathmage.analysis import style  # noqa: E402

TABLES, FIGURES = REPO / "results/tables", REPO / "results/figures"
LABEL = {"aime_2024": "AIME 2024", "aime_2025": "AIME 2025", "aime_2026": "AIME 2026", "hmmt_feb_2026": "HMMT Feb 2026",
         "tqa_phys": "TheoremQA phys.", "math500": "MATH-500", "hmmt_nov_2025": "HMMT Nov 2025", "brumo_2025": "BRUMO 2025",
         "cmimc_2025": "CMIMC 2025", "smt_2025": "SMT 2025"}
BASELINES = ["majority vote", "self-consistency (matched cost)", "universal self-consistency", "debate", "mixture of agents"]
POCOCK = 0.0294


def load():
    read = lambda name: json.loads((TABLES / name).read_text(encoding="utf-8"))  # noqa: E731
    return read("summary.json"), read("v4_summary.json"), read("v5_summary.json")


def tagged(c, kind):
    """Attach how a contrast's significance is judged: 'holm' (primary), 'pocock' (pooled debate) or 'raw'."""
    return None if not c else {**c, "kind": kind}


def sets(v3, v4, v5):
    """(evaluation set, roster, n, wave, {baseline: contrast}, {method: summary}).

    Contamination: Claude Haiku 4.5's training data runs to July 2025, so for the SAT-style team only AIME 2026,
    HMMT Feb 2026 and HMMT Nov 2025 postdate every member (93 problems; analysis/strict_cutoff.py). The April 2025
    contests postdate o3-mini and DeepSeek-V3 only. For the open team (gpt-oss-120b, June 2024) every 2025-26
    contest is strictly post-cutoff.
    """
    e1c, e2c = v3["e1"]["contrast"], v3["e2"]["contrast"]
    strict = json.loads((TABLES / "strict_sat.json").read_text(encoding="utf-8"))
    s1s, s1o = v5["S1"]["SAT roster"], v5["S1"]["open roster"]
    h6, h7 = v5["H6"], v5["H7"]
    return [
        ("Original suite (E1)", "SAT", 237, "v3 · v4 · v5",
         {"majority vote": tagged(e1c["mu* vs vote"], "raw"),
          "self-consistency (matched cost)": tagged(v4["H1"]["contrasts"]["mu* vs SC@10 (DeepSeek-V3)"], "holm"),
          "universal self-consistency": tagged(v4["H1"]["contrasts"]["mu* vs USC"], "holm"),
          "debate": tagged(v4["H1"]["contrasts"]["mu* vs debate"], "holm"),
          "mixture of agents": tagged(v5["H8"]["E1"]["contrasts"]["mu* vs MoA"], "holm")},
         {**v4["H1"]["methods"], "MoA": v5["H8"]["E1"]["methods"]["MoA"]}),
        ("April 2025 contests (primary)", "SAT", 123, "v5 H6",
         {"self-consistency (matched cost)": tagged(h6["contrasts"]["mu* vs SC@10 (DeepSeek-V3)"], "holm"),
          "universal self-consistency": tagged(h6["contrasts"]["mu* vs USC"], "holm"),
          "debate": tagged(h6["contrasts"]["mu* vs debate"], "holm"),
          "mixture of agents": tagged(h6["contrasts"]["mu* vs MoA"], "holm")}, h6["methods"]),
        ("Strictly post-cutoff for every member", "SAT", 93, "v4 H3 · v5",
         {"majority vote": tagged(v4["H3"]["e1"]["contrasts"]["mu* vs vote"], "holm"),
          "self-consistency (matched cost)": tagged(v4["H3"]["e1"]["contrasts"]["mu* vs SC@10 (DeepSeek-V3)"], "holm"),
          "universal self-consistency": tagged(v4["H3"]["e1"]["contrasts"]["mu* vs USC"], "holm"),
          "debate": tagged(v4["H3"]["e1"]["contrasts"]["mu* vs debate"], "holm"),
          "mixture of agents": tagged(strict["contrasts"]["mu* vs MoA"], "raw")}, strict["methods"]),
        ("All 2025-26 contests", "SAT", 216, "v4 + v5 S1",
         {"majority vote": tagged(s1s["contrasts"].get("mu* vs vote"), "raw"),
          "self-consistency (matched cost)": tagged(s1s["contrasts"]["mu* vs SC@10 (DeepSeek-V3)"], "raw"),
          "universal self-consistency": tagged(s1s["contrasts"]["mu* vs USC"], "raw"),
          "debate": tagged(s1s["contrasts"]["mu* vs debate"], "pocock"),
          "mixture of agents": tagged(s1s["contrasts"]["mu* vs MoA"], "raw")}, s1s["methods"]),
        ("Original suite (E2)", "open", 580, "v3 · v4 · v5",
         {"majority vote": tagged(e2c["mu* vs vote"], "raw"),
          "self-consistency (matched cost)": None,
          "universal self-consistency": tagged(v4["H2"]["contrasts"]["mu* vs USC"], "holm"),
          "debate": tagged(v4["H2"]["contrasts"]["mu* vs debate"], "holm"),
          "mixture of agents": tagged(v5["H8"]["E2"]["contrasts"]["mu* vs MoA"], "holm")},
         {**v4["H2"]["methods"], "MoA": v5["H8"]["E2"]["methods"]["MoA"]}),
        ("April 2025 + 2026 contests (primary)", "open", 1488, "v5 H7",
         {"universal self-consistency": tagged(h7["contrasts"]["mu* vs USC"], "holm"),
          "debate": tagged(h7["contrasts"]["mu* vs debate"], "holm"),
          "mixture of agents": tagged(h7["contrasts"]["mu* vs MoA"], "holm")}, h7["methods"]),
        ("All 2025-26 contests (strictly post-cutoff)", "open", 1968, "v4 + v5 S1",
         {"majority vote": tagged(v4["H3"]["e2"]["contrasts"]["mu* vs vote"], "holm"),
          "self-consistency (matched cost)": tagged(v4["H3"]["e2"]["contrasts"]["mu* vs SC@5 (pooled seeds)"], "holm"),
          "universal self-consistency": tagged(s1o["contrasts"]["mu* vs USC"], "raw"),
          "debate": tagged(s1o["contrasts"]["mu* vs debate"], "pocock"),
          "mixture of agents": tagged(s1o["contrasts"]["mu* vs MoA"], "raw")}, s1o["methods"]),
    ]


def significant(c):
    if c["kind"] == "holm" and "holm_p" in c:
        return c["holm_p"] < .05
    if c["kind"] == "pocock":
        return c["p"] < POCOCK
    return c["low"] > 0 or c["high"] < 0


def pval(p):
    return "< 0.001" if p < .001 else f"= {p:.2g}"


def fmt(c):
    raw = c.get("p", c.get("mcnemar_exact_p"))
    if c["kind"] == "holm" and "holm_p" in c:
        tail = "Holm p " + pval(c["holm_p"])
    elif c["kind"] == "pocock":
        tail = "p " + pval(raw) + " (Pocock 0.0294)"
    else:
        tail = "p " + pval(raw) if raw is not None else ""
    star = "**" if significant(c) else ""
    return f"{star}{100 * c['estimate']:+.1f}{star} [{100 * c['low']:+.1f}, {100 * c['high']:+.1f}]" + (f", {tail}" if tail else "")


COST_KEYS = {"majority vote": "vote", "universal self-consistency": "USC", "debate": "debate", "mixture of agents": "MoA"}


def sc_key(methods):
    for key in ("SC@10 (DeepSeek-V3)", "SC@5 (pooled seeds)"):
        if key in methods:
            return key
    return None


def write_tables(v3, v4, v5):
    rows = sets(v3, v4, v5)
    lines = ["# MathMage: cross-study synthesis (v3, v4, v5)", "",
             "MathMage at its frozen operating point (dev mu\\*) minus each baseline family, in macro accuracy points with "
             "95% family-clustered CIs. **Bold** = significant under the contrast's own pre-registered rule: Holm across the "
             "primary family (v4 H1–H3 or v5 H6–H8), the two-look Pocock boundary for the pooled debate contrasts (v5 S1), "
             "or an interval excluding zero for secondary and descriptive rows.", "",
             "| Evaluation set | Roster | n | Wave | vs majority vote | vs self-consistency (matched cost) | vs universal SC | "
             "vs debate | vs mixture of agents |", "|---|---|---|---|---|---|---|---|---|"]
    positive = total = sig = 0
    for name, roster, n, wave, contrasts, _ in rows:
        cells = []
        for b in BASELINES:
            c = contrasts.get(b)
            if c is None:
                cells.append("—")
                continue
            cells.append(fmt(c))
            total += 1
            positive += c["estimate"] > 0
            sig += significant(c)
        lines.append(f"| {name} | {roster} | {n:,} | {wave} | " + " | ".join(cells) + " |")
    e2_sc = v4["H2"]["contrasts"].get("maximal vs SC@5 (matched cost)")
    lines += ["", f"Point estimates favour MathMage in **{positive} of {total}** comparisons; **{sig}** are significant under "
                  f"their pre-registered rule. On E2 the pre-registered matched-cost self-consistency contrast used the maximal run: "
                  f"+{100 * e2_sc['estimate']:.1f} [{100 * e2_sc['low']:+.1f}, {100 * e2_sc['high']:+.1f}] (AIME, 480 evaluations). "
                  "Rows overlap: the 2025-26 sets contain the April primary sets and the strict set. Contamination: Claude Haiku 4.5 trained on data to July 2025, so for the SAT-style team only the strict row postdates every member.", ""]
    # cost dominance
    lines += ["## Accuracy and cost: who is dominated", "",
              "| Evaluation set | Roster | MathMage mu\\*: acc / $ | Baseline: acc / $ (cost ratio) | Verdict |", "|---|---|---|---|---|"]
    dominated = compared = 0
    sc_e2 = v3["self_consistency_e2"]["mathmage"]["mu*"]
    for name, roster, n, wave, _, methods in rows:
        mm = methods.get("mu*")
        for b in BASELINES:
            key = sc_key(methods) if b.startswith("self-consistency") else COST_KEYS[b]
            t = methods.get(key) if key else None
            if not (mm and t):
                continue
            acc, cost, note = mm["macro"], mm["cost"], ""
            if t["n"] != mm["n"]:
                shared = [bb for bb in t["by_benchmark"] if bb in mm["by_benchmark"]]
                acc = sum(mm["by_benchmark"][bb] for bb in shared) / len(shared)
                note = " ‡"
                if name.endswith("(E2)") and b.startswith("self-consistency"):
                    acc, cost = sc_e2["accuracy"], sc_e2["cost"]
            better, cheaper = acc >= t["macro"], cost <= t["cost"]
            compared += 1
            dominated += better and cheaper
            lines.append(f"| {name}{note} | {roster} | {100 * acc:.1f} / {cost:.4f} | {b}: {100 * t['macro']:.1f} / "
                         f"{t['cost']:.4f} ({t['cost'] / cost:.2f}×) | "
                         + ("more accurate **and** cheaper" if better and cheaper else
                            "more accurate, costlier" if better else "cheaper, less accurate" if cheaper else "dominated") + " |")
    lines += ["", f"MathMage is more accurate **and** cheaper in **{dominated} of {compared}** pairings. ‡ Baseline run on a "
                  "subset of the set's benchmarks; MathMage restricted to the same benchmarks.", ""]
    # per-benchmark consistency against every baseline on the contamination-free sets
    lines += ["## Per-benchmark consistency on the 2025-26 contests", "",
              "| Roster | Baseline | MathMage wins | ties | losses | benchmarks where it trails |", "|---|---|---|---|---|---|"]
    for roster, block in (("SAT", v5["S1"]["SAT roster"]), ("open", v5["S1"]["open roster"])):
        mm = block["methods"]["mu*"]["by_benchmark"]
        for b in [k for k in block["methods"] if k != "mu*"]:
            other = block["methods"][b]["by_benchmark"]
            gains = {bb: mm[bb] - other[bb] for bb in mm if bb in other}
            w = sum(g > 1e-9 for g in gains.values())
            t_ = sum(abs(g) <= 1e-9 for g in gains.values())
            lose = [LABEL.get(bb, bb) for bb, g in gains.items() if g < -1e-9]
            lines.append(f"| {roster} | {b} | {w} | {t_} | {len(lose)} | {', '.join(lose) or '—'} |")
    lines.append("")
    # per-benchmark consistency against majority vote (all sets)
    lines += ["## Per-benchmark consistency against majority vote (every benchmark and roster)", "",
              "| Roster | Benchmark | majority vote | MathMage mu\\* | gain |", "|---|---|---|---|---|"]
    wins = ties = losses = 0
    per = [("SAT", v4["H1"]["methods"]), ("SAT", v4["H4"]["e1"]["methods"]), ("open", v4["H2"]["methods"]),
           ("open", v4["H4"]["e2"]["methods"])]
    seen = set()
    for roster, methods in per:
        vote, mm = methods["vote"]["by_benchmark"], methods["mu*"]["by_benchmark"]
        for b in vote:
            if (roster, b) in seen or b not in mm:
                continue
            seen.add((roster, b))
            gain = mm[b] - vote[b]
            wins += gain > 1e-9
            ties += abs(gain) <= 1e-9
            losses += gain < -1e-9
            lines.append(f"| {roster} | {LABEL.get(b, b)} | {100 * vote[b]:.1f} | {100 * mm[b]:.1f} | {100 * gain:+.1f} |")
    lines += ["", f"MathMage at mu\\* beats majority vote on **{wins}** benchmark-roster pairs, ties on {ties}, trails on {losses}.", ""]
    # decomposition on the new contests
    lines += ["## Where the gain comes from on the 153 new contest problems", "",
              "| Roster | best single model | majority vote | + reliability-weighted selection (program voter) | "
              "+ value-of-computation actions (mu\\* / maximal) | coverage ceiling (final pool) | share of ceiling reached |",
              "|---|---|---|---|---|---|---|"]
    for roster, key, single in [("SAT", "e1", "o3-mini (low)"), ("open", "e2", "single sample (mean of 3)")]:
        m = v4["H4"][key]["methods"]
        best = max([m[single]["macro"]] + [m[k]["macro"] for k in ("Claude Haiku 4.5", "DeepSeek-V3") if k in m])
        top = max(m["mu*"]["macro"], m["maximal"]["macro"])
        lines.append(f"| {roster} | {100 * best:.1f} | {100 * m['vote']['macro']:.1f} | {100 * m['weighted vote + program']['macro']:.1f} | "
                     f"{100 * m['mu*']['macro']:.1f} / {100 * m['maximal']['macro']:.1f} | {100 * m['oracle (final pool)']['macro']:.1f} | "
                     f"{100 * top / m['oracle (final pool)']['macro']:.0f}% |")
    (TABLES / "synthesis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines, rows, {"positive": positive, "total": total, "significant": sig, "dominated": dominated, "compared": compared}


def fig_forest(rows, out, v4, rename=None):
    """rename: optional display names for evaluation sets (the paper's wording); matching uses the originals."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    P, M = style.P, style.METHOD
    rename = rename or {}
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 3.0), sharey=True)
    names = [f"{'SAT' if roster == 'SAT' else 'Open'} · {rename.get(name, name).replace(' (primary)', '*')} ({n:,})"
             for name, roster, n, *_ in rows]
    vs = r"\textit{vs.}\ "                     # the Latin abbreviation, italic as in the text
    titles = {"majority vote": vs + "majority\nvote", "self-consistency (matched cost)": vs + "self-consistency\n(matched cost)",
              "universal self-consistency": vs + "universal\nself-consistency", "debate": vs + "debate\n(one round)",
              "mixture of agents": vs + "mixture\nof agents"}
    y = np.arange(len(rows))[::-1]
    for ax, b in zip(axes, BASELINES):
        for yi, (name, roster, _, _, contrasts, _) in zip(y, rows):
            c, marker = contrasts.get(b), "o"
            if c is None and b.startswith("self-consistency") and name.endswith("(E2)"):
                c, marker = tagged(v4["H2"]["contrasts"]["maximal vs SC@5 (matched cost)"], "holm"), "D"
            if not c:
                ax.text(0.02, yi, "—", ha="left", va="center", fontsize=6, color=P["faint"],
                        transform=ax.get_yaxis_transform())
                continue
            color = M["mathmage"] if significant(c) else P["lav3"]
            ax.errorbar(100 * c["estimate"], yi, xerr=[[100 * (c["estimate"] - c["low"])], [100 * (c["high"] - c["estimate"])]],
                        fmt=marker, ms=3.4 if marker == "D" else 3.6, color=color, ecolor=color, elinewidth=1, capsize=1.6,
                        zorder=3)
        ax.axvline(0, color=P["ink"], lw=.6)
        ax.set_title(titles[b], fontsize=6.8, loc="left")
        ax.grid(True, axis="x", color=P["grid"], lw=.5)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("gain (points)", fontsize=6.2)
    axes[0].set_yticks(y, names, fontsize=6.1)
    handles = [Line2D([], [], marker="o", ls="", color=M["mathmage"], ms=4, label="passes recorded rule"),
               Line2D([], [], marker="o", ls="", color=P["lav3"], ms=4, label="does not pass"),
               Line2D([], [], marker="D", ls="", color=M["mathmage"], ms=3.6, label="maximal run (specified for this contrast)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.tight_layout(w_pad=.5)
    fig.subplots_adjust(bottom=.22)
    style.save(fig, "synthesis_forest", out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    style.use(publication=not args.draft)
    v3, v4, v5 = load()
    lines, rows, counts = write_tables(v3, v4, v5)
    (TABLES / "synthesis_counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    fig_forest(rows, FIGURES, v4)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
