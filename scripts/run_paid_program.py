"""Run the locally frozen, explicitly authorized $75 study in resumable stages.

No test-accuracy-dependent scope decisions. Provider faults stop new dispatch;
completed responses remain in the global cache. Create ROOT/STOP to stop after
in-flight requests settle. This script does not change prompts, models or budgets.

The two rosters run as independent paths: SAT (calibration, E1, ablations, Z6/Z3,
optional panels) and Xolver (calibration, E2). A provider fault stops only the
roster whose model failed and records ``provider_attention_<roster>.json``. The
STOP file, the spending guard, and unresolved billing stop both paths.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathmage import paths
from mathmage.config import load_models
from mathmage.experiments.artifacts import atomic_json, digest, file_digest, read_frozen, read_record
from mathmage.experiments.offline import offline_phase
from mathmage.experiments.mechanisms import run_mechanism
from mathmage.experiments.runner import _protocol_digest, run_study
from mathmage.data.registry import load_registry
from mathmage.experiments.offline import _sample_split
from mathmage.llm.client import LLMClient
from mathmage.llm.ledger import BudgetExceeded, BudgetLedger
from mathmage.llm.locking import file_lock
from mathmage.llm.openrouter import OpenRouterSync
from mathmage.llm.types import LLMResponse
from scripts.provider_diagnostics import ProviderDiagnostics
from scripts.provider_retry import ProviderRetry, recoverable_attention


QUEUE_CANCELLED = 'operational stop before provider dispatch; no request sent'


def safe_unbilled_429(response):
    """Only the transport's explicit, definitively unbilled 429 exhaustion."""
    match = re.fullmatch(r'failed after ([1-9][0-9]*) attempts: HTTP 429 rate limited', response.error or '')
    return (bool(match) and response.attempts == int(match.group(1))
            and not response.billing_uncertain and response.cost_usd == 0
            and response.tokens_in == 0 and response.tokens_out == 0
            and response.tokens_reasoning == 0 and not response.generation_id)


class SiliconFlowPacer:
    """Driver-only scheduling; model requests and cached responses stay unchanged."""
    def __init__(self, root, stopped, *, interval_s=10, cooldown_s=180,
                 clock=time.monotonic, wait=None, on_wait=None, dispatch_blocked=None):
        self.root, self.stopped = Path(root), stopped
        self.interval_s, self.cooldown_s = interval_s, cooldown_s
        self.clock, self.wait = clock, wait or stopped.wait
        self.on_wait = on_wait or (lambda reason: None)
        self.dispatch_blocked = dispatch_blocked or self.cancelled
        self._slot = threading.Lock()
        self._next_at = 0.0

    def cancelled(self):
        return self.stopped.is_set() or (self.root / 'STOP').exists()

    @staticmethod
    def matches(settings):
        return settings.provider == 'SiliconFlow' and settings.slug == 'qwen/qwen3-32b'

    def complete(self, original, backend, settings, messages, response_format, spec):
        if not self.matches(settings):
            return original(backend, settings, messages, response_format, spec)
        acquired = False
        try:
            self.on_wait('waiting for the shared SiliconFlow Qwen dispatch slot')
            while not self.cancelled() and not self.dispatch_blocked():
                if self._slot.acquire(blocking=False):
                    acquired = True
                    break
                self.wait(.5)
            if acquired:
                remaining = self._next_at - self.clock()
                if remaining > 0:
                    self.on_wait(f'SiliconFlow Qwen pacing/cooldown (up to {remaining:.0f}s)')
                while self.clock() < self._next_at and not self.cancelled() and not self.dispatch_blocked():
                    self.wait(min(.5, self._next_at - self.clock()))
            if not acquired or self.cancelled() or self.dispatch_blocked():
                # Raising here would make LLMClient conservatively mark billing
                # uncertain even though no HTTP request was dispatched.
                return LLMResponse(text='', error=QUEUE_CANCELLED, attempts=0,
                                   model_slug=settings.slug, provider=settings.provider), 0
            self.on_wait(None)
            delay = self.interval_s
            try:
                result = original(backend, settings, messages, response_format, spec)
                response = result[0] if isinstance(result, tuple) else result
                delay = self.cooldown_s if safe_unbilled_429(response) else self.interval_s
                return result
            finally:
                self._next_at = max(self._next_at, self.clock() + delay)
        finally:
            self.on_wait(None)
            if acquired:
                self._slot.release()


def guarded_completion(original, client, request, *, root, stopped, ledger, soft_stop_usd,
                       fault_lock, recovery_waves=2, cooldown_s=180, global_stopped=None,
                       attention_name='provider_attention.json'):
    """Retain fault stops; only replay explicit zero-charge rate-limit failures.

    ``stopped`` belongs to the request's roster path; ``global_stopped`` (if given)
    stops every path and is set by the spending and billing guards.
    """
    everything = global_stopped or stopped
    for wave in range(recovery_waves + 1):
        if client.live:
            balance = ledger.summary()
            if (root / 'STOP').exists() or stopped.is_set() or everything.is_set():
                raise BudgetExceeded('collection stop requested; completed responses retained')
            if balance['unreconciled_calls'] or balance['spent_total'] + balance['pending_total'] >= soft_stop_usd:
                everything.set()
                stopped.set()
                raise BudgetExceeded('collection stopped at resource/billing guard')
        started = time.monotonic()
        response = original(client, request)
        if not client.live or not (response.error or response.billing_uncertain):
            return response
        if response.error == QUEUE_CANCELLED:
            raise BudgetExceeded('collection stopped before queued provider dispatch; no request sent')
        if request.model_key in {'qwen3-32b', 'qwen3-32b-nothink'} and safe_unbilled_429(response):
            event = {'time_utc': now(), 'model': request.model_key, 'sample_tag': request.sample_tag,
                     'item_id': request.meta.get('item_id'), 'error': response.error,
                     'wave': wave, 'will_retry': wave < recovery_waves,
                     'cooldown_s': cooldown_s, 'attempts': response.attempts,
                     'transport_latency_s': response.latency_s,
                     'driver_elapsed_s': time.monotonic() - started, 'cost_usd': response.cost_usd,
                     'billing_uncertain': False,
                     'note': 'Operational retry overhead; retain alongside final-response runtime analyses.'}
            with fault_lock:
                with (root / 'rate_limit_recoveries.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(event) + '\n')
                    stream.flush()
            if wave < recovery_waves:
                continue
        with fault_lock:
            stopped.set()
            if response.billing_uncertain:
                everything.set()           # uncertain charges stop every path until reconciled
            # A queued cancellation or secondary fault must not replace the first
            # provider record needed for billing and recovery review.
            if not (root / attention_name).exists():
                atomic_json(root / attention_name, {'time_utc': now(), 'model': request.model_key,
                            'error': response.error, 'billing_uncertain': response.billing_uncertain,
                            'generation_id': response.generation_id,
                            'attempts': response.attempts, 'cost_usd': response.cost_usd,
                            'tokens_in': response.tokens_in, 'tokens_out': response.tokens_out,
                            'tokens_reasoning': response.tokens_reasoning,
                            'safe_unbilled_429': safe_unbilled_429(response),
                            'note': 'Only a frozen bounded provider-retry policy may replay a proven zero-charge 429. Other faults require review.'})
        raise BudgetExceeded(f'provider health gate stopped collection; see {attention_name}')


def now():
    return datetime.now(timezone.utc).isoformat()


def immutable_manifest(review):
    return {key: value for key, value in review.items() if key != 'budget_review'}


def completed_phase(target, review, registry):
    """Skip a finished phase only after validating its frozen records and runtime.

    Incomplete phases are handed back to their normal immutable checkpoint runner.
    An execution marker alone never authorizes a skip or a different manifest.
    """
    target = Path(target)
    manifest_path, execution_path = target / 'manifest.json', target / 'execution.json'
    manifest = immutable_manifest(review)
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('saved phase manifest differs from the reviewed immutable plan')
    if not execution_path.exists():
        return None
    execution = json.loads(execution_path.read_text(encoding='utf-8'))
    if execution.get('status') != 'complete':
        return None
    if not manifest_path.exists():
        raise ValueError('completed phase has no immutable manifest')
    plan_id, mode = manifest['plan_id'], manifest['mode']
    if manifest.get('experiment') == 'offline':
        result = target / 'artifacts.json'
        artifact = read_frozen(result, live=True)
        provenance = artifact['provenance']
        if (execution.get('artifact_sha256') != artifact['sha256']
                or provenance.get('manifest_plan_id') != plan_id
                or artifact.get('roster') != manifest['roster']
                or artifact.get('policy_config') != manifest['policy_config']):
            raise ValueError('completed calibration artifact does not match its manifest')
        for key in ('train_ids', 'dev_ids', 'registry_sha256', 'answer_vault_sha256',
                    'models_sha256', 'budget_sha256', 'protocol_sha256', 'tools_runtime',
                    'prompt_hashes', 'public_anchor_snapshot', 'public_kappa_snapshot'):
            if provenance.get(key) != manifest.get(key):
                raise ValueError(f'completed calibration provenance changed: {key}')
        for name in ('training_evidence.json', 'development_evidence.json'):
            read_record(target / name, plan_id=plan_id, mode=mode)
        return result
    if execution.get('tools_runtime') != manifest.get('tools_runtime'):
        raise ValueError('completed phase tools runtime changed')
    expected_count = 0
    if manifest.get('study') in {'z6', 'z3'}:
        assignments = manifest['assignments']
        if read_record(target / 'assignments.json', plan_id=plan_id, mode=mode) != {'assignments': assignments}:
            raise ValueError('completed mechanism assignments changed')
        for assignment in assignments:
            identity = {'item_id': assignment['item_id'], 'seed': assignment['seed']}
            ident = digest([identity['item_id'], identity['seed']])[:24]
            checkpoint = read_record(target / 'checkpoints' / f'{ident}.json',
                                     plan_id=plan_id, mode=mode, identity=identity)['checkpoint']
            for arm in assignment['executed_arms']:
                arm_identity = {**identity, 'arm': arm}
                raw = read_record(target / 'agent' / f'{ident}-{arm}.json',
                                  plan_id=plan_id, mode=mode, identity=arm_identity)
                if raw['checkpoint_sha256'] != checkpoint['sha256']:
                    raise ValueError('completed mechanism continuation checkpoint changed')
                read_record(target / 'graded' / f'{ident}-{arm}.json',
                            plan_id=plan_id, mode=mode, identity=arm_identity)
                expected_count += 1
    else:
        for seed in manifest['seeds']:
            for item_id in manifest['item_ids']:
                if manifest['study'] == 'e2' and registry[item_id].benchmark == 'math500' and seed != manifest['seeds'][0]:
                    continue
                for condition in manifest['conditions']:
                    ident = digest([item_id, seed, condition])[:24]
                    read_record(target / 'agent' / f'{ident}.json', plan_id=plan_id, mode=mode,
                                identity={'item_id': item_id, 'seed': seed, 'condition': condition})
                    if not (target / 'graded' / f'{ident}.json').exists():
                        raise ValueError('completed phase is missing a graded observation')
                    expected_count += 1
    if execution.get('observations') != expected_count:
        raise ValueError('completed phase observation count does not match its manifest')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / 'provider_attention.json').exists() or (root / 'STOP').exists():
        raise ValueError('explicit stop/provider attention needs review before restarting')
    plan = json.loads((root / 'program_plan.json').read_text(encoding='utf-8'))
    if plan['authorization']['cap_usd'] != 75 or plan['mode'] != 'live':
        raise ValueError('explicit $75 live authorization required')
    expected = {k: v for k, v in plan.items() if k != 'sha256'}
    if plan['sha256'] != digest(expected):
        raise ValueError('program plan hash changed')
    dependencies = plan.get('operational_dependencies_sha256', {})
    if set(dependencies) - {'provider_diagnostics.py', 'provider_retry.py'}:
        raise ValueError('unknown operational dependency in the frozen program plan')

    def check_dependencies():
        for name, expected_hash in dependencies.items():
            if file_digest(Path(__file__).with_name(name)) != expected_hash:
                raise ValueError(f'operational dependency changed: {name}')

    check_dependencies()
    for key, current in [('protocol_sha256', _protocol_digest()),
                         ('models_sha256', file_digest(paths.CONFIGS / 'models.yaml')),
                         ('budget_sha256', file_digest(paths.CONFIGS / 'budget.yaml')),
                         ('driver_sha256', file_digest(Path(__file__)))]:
        if plan[key] != current:
            raise ValueError(f'{key} changed; review a new program plan')
    if plan.get('soft_stop_usd') != 73:
        raise ValueError('unchanged $73 dispatch guard required')
    overrides = plan.get('study_policy_overrides', {})
    if (set(overrides) - {'e1', 'e2', 'ablations'}
            or any(value != {'mu': 0.0} for value in overrides.values())):
        raise ValueError('only explicitly frozen mu=0 study overrides are supported')
    ledger = BudgetLedger()
    initial_balance = ledger.summary()
    if initial_balance['pending_calls'] or initial_balance['unreconciled_calls']:
        raise ValueError('reconcile outstanding reservations/billing before restarting')
    registry = load_registry()
    learned = _sample_split(registry, 'train', plan['calibration']['train_limit']) + _sample_split(
        registry, 'dev', plan['calibration']['dev_limit'])
    heldout = [registry[i] for i in plan['mechanism_holdout_ids']]
    if (len(heldout) != 10 or len({i.item_id for i in heldout}) != 10 or any(i.split != 'dev' for i in heldout)
            or {i.family_id for i in learned} & {i.family_id for i in heldout}):
        raise ValueError('mechanism holdout must contain ten untouched development families')
    stopped = threading.Event()             # global resource, billing and integrity stop
    roster_stopped = {roster: threading.Event() for roster in ('sat', 'xolver')}
    retry_options = plan.get('transport_policy', {})
    retry_keys = {'provider_retry_interval_s', 'provider_retry_max_waves'}
    if retry_keys & set(retry_options) and not retry_keys <= set(retry_options):
        raise ValueError('provider retry needs both frozen interval and program-wide bound')
    provider_retry = (ProviderRetry(root, interval_s=retry_options['provider_retry_interval_s'],
                                  max_waves=retry_options['provider_retry_max_waves'])
                      if retry_keys <= set(retry_options) else None)
    models = load_models()
    model_rosters = {}
    for roster in roster_stopped:
        spec = models.rosters[roster]
        for key in set(spec.members) | {spec.checker, spec.card_writer, spec.default_judge}:
            if key in model_rosters and model_rosters[key] != roster:
                raise ValueError('independent roster health requires disjoint model keys')
            model_rosters[key] = roster
    attentions = {}
    for roster in roster_stopped:
        attention_path = root / f'provider_attention_{roster}.json'
        if attention_path.exists():
            attention = json.loads(attention_path.read_text(encoding='utf-8'))
            if attention.get('billing_uncertain') is not False:
                raise ValueError('uncertain provider billing needs reconciliation before restarting any roster')
            if model_rosters.get(attention.get('model')) != roster:
                raise ValueError('provider attention does not match its roster')
            attentions[roster] = attention
            roster_stopped[roster].set()
        resource_path = root / f'resource_attention_{roster}.json'
        if resource_path.exists():
            resource = json.loads(resource_path.read_text(encoding='utf-8'))
            if resource.get('roster') != roster or resource.get('billing_uncertain') is not False:
                raise ValueError('resource attention requires review before restarting')
            attentions.setdefault(roster, resource)
            roster_stopped[roster].set()
    if provider_retry and 'xolver' not in attentions:
        crash_attention = provider_retry.attention_for_resume()
        if crash_attention:
            attentions['xolver'] = crash_attention
            roster_stopped['xolver'].set()
    monitor_done = threading.Event()
    lock = threading.RLock()
    state_path = root / 'status.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'phases': {}}
    state.setdefault('phases', {})
    state.setdefault('run_history', []).append({'started_at_utc': now(), 'previous_status': state.get('status')})
    state.update(status='running', started_at_utc=state.get('started_at_utc', now()),
                 resumed_at_utc=now(), active_phases=[], transport_waits={},
                 execution_mode='independent_rosters',
                 classification='LIVE COLLECTION; NOT SIMULATED', program_plan_sha256=plan['sha256'],
                 program_root=str(root), lanes={roster: {'status': (
                      'blocked by experiment budget' if attentions[roster].get('kind') == 'experiment_budget'
                      else 'blocked by provider') if roster in attentions else 'ready',
                      **({'attention': attentions[roster]} if roster in attentions else {})}
                     for roster in roster_stopped})
    for key in ('error_type', 'ended_at_utc', 'completed_at_utc'):
        state.pop(key, None)
    transport = plan.get('transport_policy', {})
    if transport.get('siliconflow_max_concurrent', 1) != 1:
        raise ValueError('reviewed SiliconFlow transport requires one dispatch at a time')
    interval_s = transport.get('siliconflow_min_interval_s', 10)
    cooldown_s = transport.get('rate_limit_cooldown_s', 180)
    recovery_waves = transport.get('safe_429_recovery_waves', 2)
    if interval_s < 10 or cooldown_s < 180 or type(recovery_waves) is not int or not 0 <= recovery_waves <= 2:
        raise ValueError('transport pacing must retain conservative spacing and bounded recovery')
    if provider_retry and recovery_waves != 0:
        raise ValueError('durable provider retries require zero immediate recovery waves')
    qwen_max_attempts = transport.get('qwen_max_attempts', 1)
    if type(qwen_max_attempts) is not int or qwen_max_attempts != 1:
        raise ValueError('reviewed Qwen recovery requires a single transport attempt per dispatch')
    diagnostics = ProviderDiagnostics(root, qwen_max_attempts=qwen_max_attempts)

    def publish():
        with lock:
            state['updated_at_utc'] = now()
            state['lane_status'] = {roster: value['status'] for roster, value in state['lanes'].items()}
            state['budget'] = ledger.summary()
            atomic_json(root / 'status.json', state)
            atomic_json(paths.OUTPUTS / 'live_status.json', state)
            budget = state['budget']
            lines = ['# MathMage live collection', '', f"Status: **{state['status']}**.",
                     f"Updated (UTC): {state['updated_at_utc']}", '',
                     f"Recorded spend: **${budget['spent_total']:.4f}** / $75.00.",
                     f"Reserved for in-flight calls: ${budget['pending_total']:.4f}.", '',
                     'Active: ' + (', '.join(state['active_phases']) or 'none'), '',
                     '| Phase | State |', '|---|---|']
            lines.extend(f"| {name} | {value['status']} |" for name, value in state['phases'].items())
            if state['transport_waits']:
                lines.extend(['', 'Transport: ' + '; '.join(sorted(set(state['transport_waits'].values())))])
            lines.extend(['', 'Collection scope depends only on declared costs/runtime health, not test accuracy.',
                          'Provider errors pause the affected roster; billing uncertainty and STOP pause both.',
                          f'Create `{root / "STOP"}` to request a stop. In-flight calls finish and reconcile.',
                          '', 'This page reports execution and spending, not performance results.'])
            temporary = paths.OUTPUTS / 'LIVE_STATUS.md.tmp'
            temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            temporary.replace(paths.OUTPUTS / 'LIVE_STATUS.md')

    def monitor():
        while not monitor_done.wait(10):
            publish()

    original_complete = LLMClient.complete
    original_backend_complete = OpenRouterSync.complete

    def transport_wait(reason):
        with lock:
            key = str(threading.get_ident())
            if reason is None:
                state['transport_waits'].pop(key, None)
            else:
                state['transport_waits'][key] = reason

    def dispatch_blocked():
        if (root / 'STOP').exists() or stopped.is_set() or roster_stopped['xolver'].is_set():
            return True
        balance = ledger.summary()
        if balance['unreconciled_calls'] or balance['spent_total'] + balance['pending_total'] >= plan['soft_stop_usd']:
            stopped.set()
            return True
        return False

    pacer = SiliconFlowPacer(root, stopped, interval_s=interval_s, cooldown_s=cooldown_s,
                            on_wait=transport_wait, dispatch_blocked=dispatch_blocked)

    def paced_backend(backend, settings, messages, response_format, spec):
        def observed_backend(backend, settings, messages, response_format, spec):
            result = diagnostics.complete(original_backend_complete, backend, settings, messages, response_format, spec)
            if pacer.matches(settings):
                pacer._next_at = max(pacer._next_at, pacer.clock() + diagnostics.last_provider_backoff(settings))
            return result
        return pacer.complete(observed_backend, backend, settings, messages, response_format, spec)

    def guarded_complete(client, request):
        roster = model_rosters.get(request.model_key)
        if roster is None:
            stopped.set()
            raise ValueError('paid dispatch used a model outside the frozen rosters')
        response = guarded_completion(original_complete, client, request, root=root, stopped=roster_stopped[roster],
                                  global_stopped=stopped, attention_name=f'provider_attention_{roster}.json',
                                  ledger=ledger, soft_stop_usd=plan['soft_stop_usd'], fault_lock=lock,
                                  recovery_waves=recovery_waves, cooldown_s=cooldown_s)
        if (provider_retry and roster == 'xolver' and client.live and not response.cached
                and not response.error and not response.billing_uncertain
                and not stopped.is_set() and not roster_stopped[roster].is_set()):
            provider_retry.mark_recovered()
        return response

    LLMClient.complete = guarded_complete
    OpenRouterSync.complete = paced_backend

    def health_guard(roster):
        if (root / 'STOP').exists() or stopped.is_set() or roster_stopped[roster].is_set():
            raise BudgetExceeded('collection stop requested; completed responses retained')
        balance = ledger.summary()
        if balance['unreconciled_calls'] or balance['spent_total'] + balance['pending_total'] >= plan['soft_stop_usd']:
            stopped.set()
            raise BudgetExceeded('collection stopped at resource/billing guard')

    def execute(name, function, options, roster):
        health_guard(roster)
        check_dependencies()
        if (_protocol_digest() != plan['protocol_sha256']
                or file_digest(paths.CONFIGS / 'budget.yaml') != plan['budget_sha256']
                or file_digest(paths.CONFIGS / 'models.yaml') != plan['models_sha256']
                or file_digest(Path(__file__)) != plan['driver_sha256']):
            raise ValueError('source, models or budget changed during collection; stop before the next phase')
        # Never redirect process-wide stdout while the other roster is working.
        # The estimate contains public plan metadata and does not dispatch requests.
        review = function(**options, estimate=True)
        atomic_json(root / f'{name}_review.json', {'reviewed_at_utc': now(),
            'authorization': plan['authorization'], 'program_plan_sha256': plan['sha256'], 'plan': review})
        prior_manifest = root / name / 'manifest.json'
        if prior_manifest.exists() and json.loads(prior_manifest.read_text(encoding='utf-8')) != immutable_manifest(review):
            raise ValueError('saved phase manifest differs from the reviewed immutable plan')
        complete = completed_phase(root / name, review, registry)
        with lock:
            previous = state['phases'].get(name, {})
            history = list(previous.get('history', []))
            if previous:
                history.append({key: value for key, value in previous.items() if key != 'history'})
            state['phases'][name] = {'status': 'complete' if complete else 'running',
                'started_at_utc': previous.get('started_at_utc', now()), 'resumed_at_utc': now(),
                'plan_id': review['plan_id'], 'roster': roster, 'history': history}
            if complete:
                state['phases'][name].update(output=str(complete), validated_completed_resume=True,
                                           completed_at_utc=previous.get('completed_at_utc', now()))
                publish()
                return complete
            state['active_phases'].append(name)
            publish()
        try:
            result = function(**options, approved_plan=review['plan_id'])
            with lock:
                state['phases'][name].update(status='complete', completed_at_utc=now(), output=str(result))
            return result
        except BaseException as exc:
            with lock:
                state['phases'][name].update(status='stopped', error_type=type(exc).__name__, ended_at_utc=now())
            raise
        finally:
            with lock:
                state['active_phases'].remove(name)
                publish()

    def affordable(name, estimate, experiment):
        balance = ledger.summary()
        if (root / name / 'execution.json').exists():
            previous = json.loads((root / name / 'execution.json').read_text(encoding='utf-8'))
            if previous.get('status') == 'complete':
                return True                 # immutable resume, not a new spend allowance
        capacity = plan['soft_stop_usd'] - balance['spent_total'] - balance['pending_total']
        line_capacity = ledger.line(experiment) - balance['by_experiment'].get(experiment, 0)
        if capacity < estimate * 1.35 + plan['optional_reserve_usd'] or line_capacity < estimate * 1.35:
            with lock:
                state['phases'][name] = {'status': 'skipped by predeclared budget rule',
                                        'required_allowance_usd': estimate * 1.35, 'available_usd': capacity,
                                        'experiment_available_usd': line_capacity}
                publish()
            return False
        return True

    thread = threading.Thread(target=monitor, daemon=True)
    try:
        thread.start()
        publish()
        common = {**plan['calibration'], 'live': True, 'sandbox': 'local'}
        offline_options = {roster: {**common, 'roster': roster, 'output': root / f'offline_{roster}'}
                           for roster in ['sat', 'xolver']}
        artifacts = {}

        def validate_calibration(roster, path):
            artifact = read_frozen(path, live=True)
            if artifact['policy_config'] != plan['deployment_policy']:
                raise ValueError('calibration/deployment policy mismatch')
            learned_ids = set(artifact['provenance']['train_ids']) | set(artifact['provenance']['dev_ids'])
            if {registry[i].family_id for i in learned_ids} & {i.family_id for i in heldout}:
                raise ValueError('mechanism holdout overlaps frozen learning provenance')

        def study(name, roster, **options):
            selected = options['study']
            return execute(name, run_study, {'live': True, 'sandbox': 'local',
                'artifacts_path': artifacts[roster], 'output': root / name,
                **options, **overrides.get(selected, {})}, roster)

        def resource_fault(exc, roster):
            if not isinstance(exc, BudgetExceeded) or stopped.is_set() or (root / 'STOP').exists():
                return None
            match = re.fullmatch(r'([a-z0-9_]+): reserving \$[0-9]+\.[0-9]{4} exceeds its budget line', str(exc))
            allowed = {'sat': {'offline', 'e1_sat', 'e4_ablations', 'e5_zpd'}, 'xolver': {'offline', 'e2_xolver'}}
            if not match or match.group(1) not in allowed[roster]:
                return None
            balance = ledger.summary()
            if balance['unreconciled_calls'] or balance['spent_total'] + balance['pending_total'] >= plan['soft_stop_usd']:
                stopped.set()
                return None
            record = {'kind': 'experiment_budget', 'time_utc': now(), 'roster': roster,
                      'experiment': match.group(1), 'error': str(exc), 'billing_uncertain': False,
                      'note': 'Reservation refused before dispatch; unchanged experiment cap. No automatic replay or budget reallocation.'}
            roster_stopped[roster].set()
            atomic_json(root / f'resource_attention_{roster}.json', record)
            return record

        def lane_once(roster):
            if roster_stopped[roster].is_set():
                with lock:
                    prior = state['phases'].setdefault(f'offline_{roster}', {})
                    if prior.get('status') != 'complete':
                        prior.update(status=state['lanes'][roster]['status'] + '; completed records retained', roster=roster)
                    publish()
                return
            try:
                with lock:
                    state['lanes'][roster].update(status='running', started_at_utc=now())
                    publish()
                path = execute(f'offline_{roster}', offline_phase, offline_options[roster], roster)
                validate_calibration(roster, path)
                artifacts[roster] = path
                # Scope is declared before collecting test outcomes. Neither lane
                # depends on the other lane's provider or observed correctness.
                if roster == 'xolver':
                    study('e2', roster, study='e2', seeds=list(range(8)))
                else:
                    study('e1_primary', roster, study='e1', seeds=[0])
                    study('ablations_primary', roster, study='ablations', conditions=plan['primary_ablations'])
                    for mechanism in ['z6', 'z3']:
                        execute(mechanism, run_mechanism, {'study': mechanism, 'live': True, 'sandbox': 'local',
                            'artifacts_path': path, 'item_ids': plan['mechanism_holdout_ids'],
                            'paired': mechanism == 'z6', 'max_calls': 12, 'stage0_max_calls': 20,
                            'output': root / mechanism}, roster)
                with lock:
                    state['lanes'][roster].update(status='primary complete', completed_at_utc=now())
                    publish()
            except BaseException as exc:
                attention_path = root / f'provider_attention_{roster}.json'
                resource = resource_fault(exc, roster)
                scoped_fault = (isinstance(exc, BudgetExceeded) and roster_stopped[roster].is_set()
                                and attention_path.exists() and not stopped.is_set())
                with lock:
                    if resource:
                        state['lanes'][roster].update(status='blocked by experiment budget',
                            attention=resource, ended_at_utc=now())
                    elif scoped_fault:
                        state['lanes'][roster].update(status='blocked by provider',
                            attention=json.loads(attention_path.read_text(encoding='utf-8')),
                            ended_at_utc=now())
                    else:
                        stopped.set()
                        state['lanes'][roster].update(status='stopped; review required',
                            error_type=type(exc).__name__, ended_at_utc=now())
                    publish()
                if not scoped_fault and not resource:
                    raise

        def wait_for_provider(roster):
            if not provider_retry or roster != 'xolver' or stopped.is_set():
                return False
            attention = state['lanes'][roster].get('attention')
            if not attention or not recoverable_attention(attention):
                return False

            def stop_requested():
                if (root / 'STOP').exists() or stopped.is_set():
                    stopped.set()
                    return True
                balance = ledger.summary()
                if balance['unreconciled_calls'] or balance['spent_total'] + balance['pending_total'] >= plan['soft_stop_usd']:
                    stopped.set()
                    return True
                return False

            def waiting(snapshot):
                with lock:
                    state['lanes'][roster].update(status='waiting for scheduled provider recovery',
                        next_retry_at_utc=snapshot['next_retry_at_utc'],
                        provider_retry_waves_used=snapshot['waves_used'],
                        provider_retry_max_waves=snapshot['max_waves'])
                    state.update(status='running; Xolver waiting for scheduled provider recovery')
                    publish()

            delay = diagnostics.last_provider_backoff('SiliconFlow')
            rearmed = provider_retry.wait_and_rearm(attention, stop_requested=stop_requested,
                                                   on_wait=waiting, provider_delay_s=delay)
            snapshot = provider_retry.snapshot()
            with lock:
                state['lanes'][roster].update(provider_retry_waves_used=snapshot['waves_used'],
                    provider_retry_max_waves=snapshot['max_waves'])
                if rearmed:
                    # The helper persisted its counter and archived the exact
                    # zero-charge proof before this event permits dispatch.
                    roster_stopped[roster].clear()
                    state['lanes'][roster].update(status='retrying provider',
                        last_retry_at_utc=snapshot['last_retry_at_utc'])
                    state['lanes'][roster].pop('next_retry_at_utc', None)
                    state['lanes'][roster].pop('attention', None)
                    state.update(status='running')
                elif snapshot['status'] == 'exhausted':
                    state['lanes'][roster].update(status='blocked by provider; scheduled retries exhausted')
                    state['lanes'][roster].pop('next_retry_at_utc', None)
                publish()
            if stopped.is_set():
                raise BudgetExceeded('scheduled provider recovery stopped by global resource/billing/STOP guard')
            return rearmed

        def lane(roster):
            while True:
                lane_once(roster)
                if state['lanes'][roster]['status'] == 'primary complete':
                    return
                try:
                    if not wait_for_provider(roster):
                        return
                except BaseException:
                    stopped.set()
                    raise

        errors = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {roster: pool.submit(lane, roster) for roster in ('sat', 'xolver')}
            for roster, future in futures.items():
                try:
                    future.result()
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]
        primary_complete = all(value['status'] == 'primary complete' for value in state['lanes'].values())
        if primary_complete:
            # Optional work waits for both primary lanes so a blocked Qwen provider
            # cannot surrender E2's funding to optional SAT panels.
            try:
                if affordable('ablations_secondary', plan['secondary_ablations_allowance_usd'], 'e4_ablations'):
                    study('ablations_secondary', 'sat', study='ablations', conditions=plan['secondary_ablations'])
                seed_allowance = sum(json.loads(p.read_text(encoding='utf-8'))['cost_usd']
                                     for p in (root / 'e1_primary' / 'graded').glob('*.json'))
                for seed in [1, 2]:
                    name = f'e1_seed{seed}'
                    if not affordable(name, seed_allowance, 'e1_sat'):
                        break
                    study(name, 'sat', study='e1', seeds=[seed])
            except BaseException as exc:
                attention_path = root / 'provider_attention_sat.json'
                resource = resource_fault(exc, 'sat')
                if resource:
                    with lock:
                        state['lanes']['sat'].update(status='optional work blocked by experiment budget', attention=resource)
                    primary_complete = False
                elif (isinstance(exc, BudgetExceeded) and roster_stopped['sat'].is_set()
                        and attention_path.exists() and not stopped.is_set()):
                    with lock:
                        state['lanes']['sat'].update(status='optional work blocked by provider',
                            attention=json.loads(attention_path.read_text(encoding='utf-8')))
                    primary_complete = False
                else:
                    stopped.set()
                    raise
        else:
            with lock:
                for name in ('ablations_secondary', 'e1_seed1', 'e1_seed2'):
                    if state['phases'].get(name, {}).get('status') != 'complete':
                        state['phases'][name] = {'status': 'deferred until both primary lanes finish', 'roster': 'sat'}
        with lock:
            if primary_complete:
                state.update(status='complete', completed_at_utc=now())
            else:
                state.update(status='partial collection stopped; review required', ended_at_utc=now())
    except BaseException as exc:
        with lock:
            state.update(status='stopped; review required', error_type=type(exc).__name__, ended_at_utc=now())
        raise
    finally:
        monitor_done.set()
        if thread.ident is not None:
            thread.join(timeout=15)
        try:
            publish()
        finally:
            LLMClient.complete = original_complete
            OpenRouterSync.complete = original_backend_complete


if __name__ == '__main__':
    # A process-wide lock prevents a second background driver from racing the
    # first driver's status/review files. Per-request locks still prevent billing
    # duplicates if a separate approved command is run concurrently.
    with file_lock(Path(__file__).resolve().parents[1] / 'runs' / '.paid_program.lock'):
        main()
