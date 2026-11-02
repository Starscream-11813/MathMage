# Open-weight lane (v4-open): gpt-oss-120b replaces the blocked Qwen3-32B roster

Decided with the user on 2026-10-19, before any open-roster test item was run.

## Why the roster changed

- OpenRouter's only long-output Qwen3-32B host (SiliconFlow) returned "temporarily
  rate-limited upstream" (HTTP 429, unbilled) from 11:13 UTC to at least 17:22 UTC on
  2026-10-18, through six bounded recovery waves.
- The other Qwen3-32B hosts cap output near 16k tokens. In the 68 successful Qwen3-32B
  thinking responses we collected, 22 (32%) exceeded 16k tokens (p75 19.2k, max 24.6k),
  so those hosts would truncate about a third of answers.
- gpt-oss-120b is open-weight (Apache-2.0), served by more than 20 OpenRouter hosts with
  about 118k-token outputs, at roughly $0.03 / $0.17 per million tokens.

No held-out test result informed this change. The decision rests on provider
availability, output limits and cost only.

## Roster

- Three instances of `openai/gpt-oss-120b` (homogeneous team, like the Qwen roster).
- Auxiliary roles (program checker, card writer, default judge) use the same model and effort.
- Hosts: CoreWeave, then Near AI, then Nebius. All serve the native MXFP4 weights and
  support every sent parameter. OpenRouter never routes outside this list
  (`allow_fallbacks: false`, `require_parameters: true`). The serving host of every
  call is logged.
- Sampling: temperature 1.0, top_p 1.0 (OpenAI's gpt-oss recommendation); 40k max tokens.

## Effort-selection rule (declared before the pilot)

The pilot solves each of the 16 frozen **training** items once at reasoning effort
`low` and once at `medium`, with the stage-0 solve prompt. No development or test item
is used.

1. An effort qualifies if its training accuracy lies in [40%, 75%]: enough headroom for
   team methods, comparable to the SAT roster's members (about 50% on AIME).
2. If both qualify, choose `low` (cheaper and faster, mirroring o3-mini at low effort).
3. If exactly one qualifies, choose it.
4. If neither qualifies, choose the effort whose accuracy is closest to 57.5%, the
   midpoint of the band; break ties toward `low`.

## What stays the same

The v3 method code (posterior model `sequential_cluster_v3`, VOC policy, four-action
horizon, 20-call limit, $0.20 estimated item cap, `mu = 0` maximal evaluation runs),
the training/development/mechanism splits, the E2 item set (60 AIME 2024/2025 items at
eight seeds plus MATH-100 once) and the single $75 authorization with the $73 soft stop.

## Code differences from the main v3 package

Only four, all operational:

- `mathmage/paths.py`: configs come from this lane; data, ledger, cache, logs and runs
  are the main repository's.
- `mathmage/llm/types.py`, `mathmage/llm/openrouter.py`: an ordered list of same-weight hosts.
- `mathmage/experiments/offline.py`: public MathArena "GPT OSS 120B (high)" logs on
  training items serve as a linked anchor agent (as o3-mini and DeepSeek do for SAT).
- CLI roster choices include `open`.
