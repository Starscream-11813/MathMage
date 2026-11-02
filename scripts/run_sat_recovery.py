"""Resume only the frozen SAT calibration while the Qwen provider is blocked.

No scope, prompts, models, or budget changes. This operational wrapper retains
the original Qwen attention record and shares the main driver's process lock.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathmage import paths
from mathmage.config import load_models
from mathmage.experiments.artifacts import atomic_json, digest, file_digest
from mathmage.experiments.offline import offline_phase
from mathmage.experiments.runner import _protocol_digest
from mathmage.llm.client import LLMClient
from mathmage.llm.ledger import BudgetExceeded, BudgetLedger
from mathmage.llm.locking import file_lock
from mathmage.llm.types import LLMResponse
from scripts.run_paid_program import guarded_completion, safe_unbilled_429


def now():
    return datetime.now(timezone.utc).isoformat()


def reviewed_qwen_attention(root):
    """Require a matching zero-charge call-log record, not merely an error label."""
    attention = json.loads((root / 'provider_attention.json').read_text(encoding='utf-8'))
    if (attention.get('model') not in {'qwen3-32b', 'qwen3-32b-nothink'}
            or attention.get('billing_uncertain') is not False or attention.get('generation_id') is not None):
        raise ValueError('only a reviewed, explicitly unbilled Qwen rate-limit block can be isolated')
    attention_time = datetime.fromisoformat(attention['time_utc'])
    for line in reversed((paths.LOGS / 'calls' / 'offline.jsonl').read_text(encoding='utf-8').splitlines()):
        row = json.loads(line)
        if row.get('model_key') != attention['model'] or row.get('error') != attention['error']:
            continue
        if abs((datetime.fromisoformat(row['timestamp']) - attention_time).total_seconds()) > 10:
            continue
        response = LLMResponse(text='', error=row['error'], attempts=row['attempts'],
            billing_uncertain=row['billing_uncertain'], cost_usd=row['cost_usd'],
            tokens_in=row['tokens_in'], tokens_out=row['tokens_out'],
            tokens_reasoning=row['tokens_reasoning'], generation_id=row['generation_id'])
        if safe_unbilled_429(response):
            return attention
    raise ValueError('Qwen attention lacks a matching definitively zero-charge 429 call record')


def run(root):
    """Caller holds the shared paid-program lock before any run-state guards."""
    root = root.resolve()
    if (root / 'STOP').exists():
        raise ValueError('global STOP must be reviewed before restarting')
    health = root / 'sat_recovery'
    if (health / 'provider_attention.json').exists() or (health / 'STOP').exists():
        raise ValueError('SAT recovery attention/stop needs review before restarting')
    plan = json.loads((root / 'program_plan.json').read_text(encoding='utf-8'))
    if plan.get('mode') != 'live' or plan.get('authorization', {}).get('cap_usd') != 75:
        raise ValueError('explicit $75 live authorization required')
    if plan.get('sha256') != digest({k: v for k, v in plan.items() if k != 'sha256'}):
        raise ValueError('program plan hash changed')
    if plan.get('soft_stop_usd') != 73:
        raise ValueError('unchanged $73 dispatch guard required')
    wrapper_hash = file_digest(Path(__file__))

    def check_sources():
        for key, value in [('protocol_sha256', _protocol_digest()),
                           ('models_sha256', file_digest(paths.CONFIGS / 'models.yaml')),
                           ('budget_sha256', file_digest(paths.CONFIGS / 'budget.yaml')),
                           ('driver_sha256', file_digest(Path(__file__).with_name('run_paid_program.py')))]:
            if plan[key] != value:
                raise ValueError(f'{key} changed; SAT recovery cannot alter the frozen study')
        if file_digest(Path(__file__)) != wrapper_hash:
            raise ValueError('SAT recovery wrapper changed during collection')

    check_sources()
    ledger = BudgetLedger()
    balance = ledger.summary()
    if balance['pending_calls'] or balance['unreconciled_calls']:
        raise ValueError('reconcile pending/unreconciled calls before SAT recovery')
    if balance['global_cap'] != 75 or balance['spent_total'] + balance['pending_total'] >= 73:
        raise ValueError('unchanged $75 cap and available $73 dispatch capacity required')
    attention = reviewed_qwen_attention(root)
    options = {**plan['calibration'], 'roster': 'sat', 'live': True, 'sandbox': 'local',
               'output': root / 'offline_sat'}
    manifest = json.loads((root / 'offline_sat' / 'manifest.json').read_text(encoding='utf-8'))
    approved = json.loads((root / 'offline_sat_review.json').read_text(encoding='utf-8'))['plan']['plan_id']
    with contextlib.redirect_stdout(io.StringIO()):
        review = offline_phase(**options, estimate=True)
    if review['plan_id'] != approved or {k: v for k, v in review.items() if k != 'budget_review'} != manifest:
        raise ValueError('SAT recovery must match the exact previously approved offline manifest')
    roster = load_models().rosters['sat']
    allowed_models = set(roster.members) | {roster.checker, roster.card_writer, roster.default_judge}
    if any(key.startswith('qwen') for key in allowed_models):
        raise ValueError('SAT-only recovery cannot dispatch a Qwen model')
    health.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    atomic_json(health / f'review_{stamp}.json', {
        'reviewed_at_utc': now(), 'authorization': plan['authorization'], 'wrapper_sha256': wrapper_hash,
        'program_plan_sha256': plan['sha256'], 'program_plan': plan, 'approved_sat_plan_id': approved,
        'sat_output': str(options['output']), 'blocked_qwen_attention': attention,
        'scope': 'Only existing SAT calibration; no E1 or Qwen dispatch; no budget or protocol changes.',
        'starting_budget': balance})
    state = json.loads((root / 'status.json').read_text(encoding='utf-8'))
    state.update(status='partial collection running; Qwen paused', active_phases=['offline_sat'],
                 program_plan_sha256=plan['sha256'], transport_waits={},
                 classification='LIVE COLLECTION; NOT SIMULATED',
                 recovery={'wrapper': str(Path(__file__)), 'wrapper_sha256': wrapper_hash,
                           'scope': 'SAT calibration only', 'qwen_attention': attention})
    state.setdefault('phases', {}).setdefault('offline_sat', {}).update(
        status='running', plan_id=approved, resumed_at_utc=now())
    state['phases'].setdefault('offline_xolver', {}).update(
        status='blocked by Qwen provider rate limit', error=attention['error'])
    lock = threading.RLock()
    stopped, monitor_done = threading.Event(), threading.Event()

    def publish():
        with lock:
            state.update(updated_at_utc=now(), budget=ledger.summary())
            atomic_json(root / 'status.json', state)
            atomic_json(paths.OUTPUTS / 'live_status.json', state)
            lines = ['# MathMage live collection', '', f"Status: **{state['status']}**.",
                     f"Updated (UTC): {state['updated_at_utc']}", '',
                     f"Recorded spend: **${state['budget']['spent_total']:.4f}** / $75.00.",
                     f"Reserved for in-flight calls: ${state['budget']['pending_total']:.4f}.", '',
                     'Active: ' + (', '.join(state['active_phases']) or 'none'), '',
                     '| Phase | State |', '|---|---|']
            lines.extend(f"| {name} | {value['status']} |" for name, value in state['phases'].items())
            lines.extend(['', 'Qwen/Xolver remains paused; its provider-attention record is retained.',
                          'Only the existing SAT calibration resumes; E1 has not started.',
                          f'Create `{root / "STOP"}` to stop new dispatch; in-flight calls settle.',
                          '', 'This page reports execution and spending, not performance results.'])
            temporary = paths.OUTPUTS / 'LIVE_STATUS.md.tmp'
            temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            temporary.replace(paths.OUTPUTS / 'LIVE_STATUS.md')

    def monitor():
        while not monitor_done.wait(10):
            publish()

    original_complete = LLMClient.complete

    def complete(client, request):
        if request.model_key not in allowed_models:
            stopped.set()
            raise BudgetExceeded('SAT-only recovery refused a model outside its frozen roster')
        check_sources()
        if (root / 'STOP').exists():
            stopped.set()
        return guarded_completion(original_complete, client, request, root=health, stopped=stopped,
            ledger=ledger, soft_stop_usd=73, fault_lock=lock, recovery_waves=0)

    LLMClient.complete = complete
    thread = threading.Thread(target=monitor, daemon=True)
    try:
        publish()
        thread.start()
        result = offline_phase(**options, approved_plan=approved)
        with lock:
            state['phases']['offline_sat'].update(status='complete', completed_at_utc=now(), output=str(result))
            state.update(status='waiting for Qwen provider recovery', active_phases=[])
        return result
    except BaseException as exc:
        stopped.set()
        with lock:
            state['phases']['offline_sat'].update(status='stopped', error_type=type(exc).__name__, ended_at_utc=now())
            state.update(status='SAT recovery stopped; Qwen remains paused', active_phases=[])
        raise
    finally:
        monitor_done.set()
        if thread.ident is not None:
            thread.join(timeout=15)
        LLMClient.complete = original_complete
        publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    with file_lock(paths.RUNS / '.paid_program.lock'):
        run(args.root)


if __name__ == '__main__':
    main()
