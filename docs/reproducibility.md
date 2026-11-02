# Reproducing and inspecting MathMage

This release contains the research implementation, the three model-team configurations, frozen dataset registries, statistical analysis code, aggregate result snapshots and the source hashes that the study recorded. Commands below run from the repository root in an activated Python environment. Python 3.11 or later is required.

## What can run immediately

Install the package with `python -m pip install -e '.[dev]'`, then run `python -m pytest`. The tests exercise inference, calibration, policy decisions, answer grading, separation of reference answers from prompts, checkpoint integrity, and resource accounting. The mock walkthrough in the README uses the bundled registry; `--no-public-logs` keeps calibration from downloading external model responses. Synthetic outcomes check execution and are not measurements of language-model performance.

`python scripts/verify_provenance.py` checks the release against the hashes recorded by the study's frozen plans and manifests (see [Source provenance and release edits](#source-provenance-and-release-edits)).

`python analysis/prop2_check.py` compares the repeated-author evidence calculation with its closed form. It requires no recorded model responses.

Three figure commands use only the included aggregate JSON snapshots:

```bash
python analysis/fig_frontier.py --draft
python analysis/fig_cost.py --draft
python analysis/fig_v6.py --draft
```

They write PNG and PDF previews to `results/figures/`. These commands redraw existing estimates; they do not recompute confidence intervals from individual responses. Draft mode uses Matplotlib and needs no TeX installation. Publication typography additionally requires LuaLaTeX, installed Latin Modern OpenType fonts, and `python -m pip install -e '.[figures]'` followed by `python scripts/build_fonts.py`. The font builder creates local derivatives; generated fonts are not distributed here.

These checks were run with Python 3.13 on Windows: the 400 tests pass, the four-command mock walkthrough completes, the three figures redraw, and all 26 recorded bindings reproduce.

## Code and experiment map

| Component | Implementation |
|:---|:---|
| Answer posterior, author reliability, response dependence, and zone estimates | `mathmage/team/posterior.py` |
| Value-of-computation policy and stopping | `mathmage/team/policy.py` |
| Agent orchestration and evidence updates | `mathmage/team/engine.py` |
| Calibration and development-selected artifacts | `mathmage/experiments/offline.py`, `mathmage/psychometrics/` |
| Prompts and hint checks | `mathmage/prompts/`, `mathmage/scaffolds/` |
| Answer extraction, equivalence, and grader-only vault | `mathmage/grading/` |
| Executed program checks | `mathmage/tools/` |
| Paired estimates and clustered bootstrap statistics | `mathmage/analysis/`, `analysis/` |
| First paid wave of the SAT-style team: calibration, E1, ablations, ZPD continuations | `scripts/run_paid_program.py`, with `run_e1_sat.py`, `run_ablations.py`, `run_zpd.py` |
| Open-weight team: launcher, effort-selection protocol, training-only pilot | `lanes/open_weight/run_lane.py`, `PROTOCOL.md`, `pilot_effort.py` |
| Additional contests and matched baseline comparisons | `scripts/v4_followup.py`, `v5_followup.py` |
| Fresh-set replication, response banks, hint trial, and refitting | `scripts/v6_followup.py`, `v6_fit.py` |
| Program-augmented baseline comparisons | `scripts/v7_parity.py` |
| Source hashes recorded by the plans and manifests | `results/provenance.json`, `scripts/verify_provenance.py` |

The `v4`–`v7` filenames retain the study-wave identifiers used by saved manifests. They share helpers, so the earlier drivers remain dependencies of the later ones. The copies of `mathmage/` in `lanes/open_weight/` and `lanes/sat26/` preserve each team's implementation and configuration. They share data and run locations with the root package. Run the lane-selecting drivers in separate Python processes to avoid mixing imported packages. `lanes/open_weight/PROTOCOL.md` records why that lane replaced the Qwen3-32B roster, the effort-selection rule declared before its pilot, and the lane's four operational code differences from the root package.

## Data and provenance

| Directory | Contents |
|:---|:---|
| `data/registry/` | Original problem registry, family/split assignments, named subsets, data card, and answer vault |
| `data/registry_ext/` | Additional MathArena contest problems and source/hash manifest |
| `data/registry_v6/` | 187-problem fresh-set registry, source transcriptions, exclusions, curation overrides, and hash manifest |
| `results/tables/` | Aggregate estimates, contrasts, calibration diagnostics, and frontier summaries |

Each registry keeps `items.parquet` separate from `answers.parquet`. Agents receive restricted item views; reference answers and source solutions are used by the grader. Keep these boundaries when extending an experiment.

The bundled registry is the reference for this release. `scripts/build_registry.py` rebuilds the original suite from public sources and can download datasets and an optional embedding model. `scripts/v4_data.py` builds the extension. `scripts/v6_data.py` builds the fresh set from the included transcriptions. Inspect each script before rebuilding: upstream revisions and optional similarity backends can change families, split assignments, and file hashes. The original builder uses a 0.92 similarity threshold; the extension and fresh-set builders use their manifest-recorded 0.85 rule. These are distinct construction stages.

Source attribution is retained in registry records, transcriptions, manifests, and loader code. The project MIT license covers the implementation; benchmark problems, solutions, and source materials retain their upstream terms. See the MathArena dataset identifiers and official competition URLs recorded with the data, TheoremQA provenance in `mathmage/data/loaders.py`, and the MATH-500 source identifier there.

## Statistical recomputation and recorded runs

The release does not include raw live response archives, provider caches, account records, or original execution logs. Aggregate JSON files preserve the reported estimates but are insufficient for rerunning the full statistical analysis. For that, the analysis scripts require the corresponding agent records, graded rows, manifests, calibration observations, and fitted artifacts in the historical layout:

```text
runs/live_20261018/
├── program_v3/       # Original SAT-style study and offline_sat/
├── program_open/     # Open-weight study and offline_open/
├── program_v4/       # Additional contests and baselines
├── program_v5/       # Completed baseline matrix
├── program_v6/       # Fresh replication, mechanisms, offline_sat26/, refitted artifacts
└── program_v7/       # Program-augmented baselines
```

The core dependency order is `analysis/results.py`, `v4_results.py`, `v5_results.py`, `v6_results.py`, then `v7_results.py`. Auxiliary scripts inspect stopping, calibration, agreement, hint strata, and cutoff-sensitive subsets. `fig_zpd.py` and `fig_zpd_v6.py` also require fitted artifacts and training evidence from the recorded runs. The historical `v6`/`v7` driver `mocktest` subcommands require saved source states and plans, so they are not part of the self-contained walkthrough.

## Source provenance and release edits

Before their first model call, the study's plans and manifests recorded SHA-256 hashes of what they would run: each team package's protocol digest (every `.py` file of the package except five presentation-only modules, `analysis/style.py`, `diagram.py`, `concepts.py`, `figures.py` and `atlas.py`), the drivers, the model configurations and the registries. `results/provenance.json` lists the 26 bindings that this release reproduces, and `python scripts/verify_provenance.py` recomputes them; each package digest comes from that package's own `_protocol_digest()`. Every frozen source therefore ships byte for byte, original development comments included, and `.gitattributes` stops Git from rewriting line endings. The manifest's `not_reproduced` entries name what the release cannot match: the pilots and the first paid program ran earlier code, and `configs/budget.yaml` gained spending lines between waves.

Only files outside every recorded binding differ from the development tree:

- **Presentation-only modules** in all three packages: comments on the palette and the drawing primitives. The root package's `analysis/style.py` also falls back to DejaVu fonts in draft mode when the Latin Modern TrueType faces have not been built.
- **Analysis scripts**, moved from the development tree's `paper/` directory to `analysis/`. They write to `results/` instead of `paper/out/`, `fig_zpd.py` labels models without the manuscript's logo files, and `fig_frontier.py` is new: it redraws the frontier from its bundled snapshot.
- **Three test fixtures**, which stub the optional public-log download and pin the budget that a test asserts, so the suite runs offline.

The frozen drivers' docstrings and the v7 analysis plan still name the development paths `paper/v4_results.py` and `paper/v7_results.py`; they correspond to `analysis/v4_results.py` and `analysis/v7_results.py`. `scripts/v6_worker_guarded.py` starts its grading child with the Windows-only flag `subprocess.CREATE_NO_WINDOW`, as it ran; on Linux or macOS, drop that argument before rerunning the fourth-wave workers. A new live study must freeze new plans and artifacts; no historical hash check is bypassed.

## New live experiments

Mock mode is the default in the general-purpose runners. For a new live study, set `OPENROUTER_API_KEY` in the environment and inspect `python scripts/preflight.py --help`. Model identifiers, provider pins, prices, and budget lines describe the recorded study; they are historical configuration, not a guarantee of current endpoint availability or pricing. The root configuration also retains the earlier Qwen comparison setup; the paper's open-weight team is configured in `lanes/open_weight/configs/models.yaml`.

The general-purpose live runners use `--live --estimate` to create an itemized plan and require its `--approved-plan` identifier before execution. Calibration and evaluation must use compatible runtime and source metadata. Historical follow-up drivers are orchestration code for the saved study layout, rather than a one-command fresh reproduction.

Generated-code checks use the selected Docker sandbox or an explicitly chosen local-process backend. `auto` does not silently fall back to local execution. The local backend is a limited process-isolation option. Use `--help` on the runner to inspect sandbox and resource settings; the mock walkthrough does not require a paid provider or a prepared Docker image.
