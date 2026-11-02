"""Training-only reasoning-effort pilot for gpt-oss-120b (rule in PROTOCOL.md).

Solves each frozen training item once per effort with the stage-0 solve prompt.
Paid, tiny (about 32 calls). Writes pilot_effort.json next to this file.
"""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

LANE = Path(__file__).resolve().parent
sys.path.insert(0, str(LANE))

from mathmage import paths  # noqa: E402
from mathmage.data.registry import load_registry  # noqa: E402
from mathmage.experiments.offline import _sample_split  # noqa: E402
from mathmage.grading.equiv import normalize  # noqa: E402
from mathmage.grading.extract import extract_final  # noqa: E402
from mathmage.grading.verify import grade_answer  # noqa: E402
from mathmage.llm.client import LLMClient  # noqa: E402
from mathmage.llm.types import LLMRequest  # noqa: E402
from mathmage.prompts.templates import SOLVE  # noqa: E402

EFFORTS = ["gpt-oss-120b-low", "gpt-oss-120b-medium"]


def choose(accuracy: dict[str, float]) -> tuple[str, str]:
    qualifies = [key for key in EFFORTS if 0.40 <= accuracy[key] <= 0.75]
    if len(qualifies) == 2:
        return EFFORTS[0], "both efforts in [40%, 75%]: choose low"
    if len(qualifies) == 1:
        return qualifies[0], "only qualifying effort"
    best = min(EFFORTS, key=lambda key: (abs(accuracy[key] - 0.575), EFFORTS.index(key)))
    return best, "neither in band: closest to 57.5%"


def main():
    plan = json.loads((paths.RUNS / "live_20261018/program_v3/program_plan.json").read_text(encoding="utf-8"))
    registry = load_registry()
    train = _sample_split(registry, "train", plan["calibration"]["train_limit"])
    if [i.item_id for i in train] != plan["training_ids"]:
        raise ValueError("training split differs from the frozen plan")
    client = LLMClient(experiment="pilot", live=True)
    rows = []
    try:
        requests = []
        for key in EFFORTS:
            for item in train:
                view = item.view()
                requests.append((key, item, LLMRequest(
                    model_key=key, template_id=SOLVE.id, kind=item.kind, sample_tag="open_lane_effort_pilot",
                    messages=[{"role": "user", "content": SOLVE.render(problem=item.problem,
                                                                        format_instruction=view.format_instruction,
                                                                        cards="")}],
                    meta={"item_id": item.item_id, "family_id": item.family_id, "condition": "effort_pilot",
                          "role": key, "phase": "pilot"})))
        responses = client.complete_many([r for _, _, r in requests])
        for (key, item, _), response in zip(requests, responses):
            raw = extract_final(response.text, lenient=False) if response.ok and not response.truncated else None
            answer = normalize(raw, item.answer_format) if raw else None
            correct = bool(answer is not None and grade_answer(item.item_id, answer).correct)
            rows.append({"model": key, "item_id": item.item_id, "correct": correct, "error": response.error,
                         "truncated": response.truncated, "tokens_out": response.tokens_out,
                         "tokens_reasoning": response.tokens_reasoning, "latency_s": response.latency_s,
                         "cost_usd": response.cost_usd, "provider": response.provider})
    finally:
        client.close()
    summary = {}
    for key in EFFORTS:
        mine = [r for r in rows if r["model"] == key]
        summary[key] = {"n": len(mine), "accuracy": statistics.mean(r["correct"] for r in mine),
                        "errors": sum(bool(r["error"]) for r in mine), "truncated": sum(bool(r["truncated"]) for r in mine),
                        "median_tokens_out": statistics.median(r["tokens_out"] for r in mine),
                        "median_latency_s": statistics.median(r["latency_s"] for r in mine),
                        "cost_usd": sum(r["cost_usd"] for r in mine),
                        "providers": sorted({r["provider"] for r in mine if r["provider"]})}
    chosen, reason = choose({key: summary[key]["accuracy"] for key in EFFORTS})
    result = {"rule": "lanes/open_weight/PROTOCOL.md", "split": "training only", "summary": summary,
              "chosen": chosen, "reason": reason, "rows": rows}
    (LANE / "pilot_effort.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "chosen": chosen, "reason": reason}, indent=2))


if __name__ == "__main__":
    main()
