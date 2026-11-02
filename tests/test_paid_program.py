"""Mock-only orchestration tests: no backend, credentials, or network are used."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathmage.experiments.artifacts import atomic_json, digest, freeze, seal_record


@pytest.fixture
def program(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts" / "run_paid_program.py"
    spec = importlib.util.spec_from_file_location("paid_program_under_test", source)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    root = tmp_path / "program"
    root.mkdir()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(driver.paths, "OUTPUTS", outputs)
    monkeypatch.setattr(driver, "_protocol_digest", lambda: "source")
    monkeypatch.setattr(driver, "file_digest", lambda p: {
        "models.yaml": "models", "budget.yaml": "budget",
        "run_paid_program.py": "driver"}[Path(p).name])
    train = [SimpleNamespace(item_id=f"train/{i}", family_id=f"train-family/{i}", split="train")
             for i in range(16)]
    tuning = [SimpleNamespace(item_id=f"tuning/{i}", family_id=f"tuning-family/{i}", split="dev")
              for i in range(10)]
    heldout = [SimpleNamespace(item_id=f"heldout/{i}", family_id=f"heldout-family/{i}", split="dev")
               for i in range(10)]
    registry = {it.item_id: it for it in train + tuning + heldout}
    for item in registry.values():
        item.benchmark = 'aime_2023'
    monkeypatch.setattr(driver, "load_registry", lambda: registry)
    monkeypatch.setattr(driver, "_sample_split", lambda registry, split, limit:
                        (train if split == "train" else tuning)[:limit])

    class Ledger:
        spent = 0.0
        pending = 0
        unreconciled = 0
        line_caps = {"e1_sat": 100.0, "e4_ablations": 100.0}
        by_experiment = {}

        def summary(self):
            return {"spent_total": self.spent, "pending_total": 0.0,
                    "pending_calls": self.pending, "unreconciled_calls": self.unreconciled,
                    "by_experiment": dict(self.by_experiment)}

        def line(self, experiment):
            return self.line_caps[experiment]

    ledger = Ledger()
    monkeypatch.setattr(driver, "BudgetLedger", lambda: ledger)
    rosters = {'sat': SimpleNamespace(members=['o3', 'haiku'], checker='haiku', card_writer='haiku', default_judge='o3'),
               'xolver': SimpleNamespace(members=['qwen3-32b'], checker='qwen3-32b-nothink',
                   card_writer='qwen3-32b-nothink', default_judge='qwen3-32b')}
    monkeypatch.setattr(driver, 'load_models', lambda: SimpleNamespace(rosters=rosters))

    class Client:
        def __init__(self, *args, **kwargs):
            raise AssertionError("orchestration tests must never create an LLM client")

        @staticmethod
        def complete(client, request):
            if request.model_key.startswith('qwen') and behavior.qwen_rate_limit_failures:
                behavior.qwen_rate_limit_failures -= 1
                return driver.LLMResponse(text='', error='failed after 1 attempts: HTTP 429 rate limited', attempts=1)
            return driver.LLMResponse(text='', error="mock provider unavailable")

    monkeypatch.setattr(driver, "LLMClient", Client)
    policy = {"max_policy_actions": 4, "max_calls": 20, "max_item_cost_usd": .20,
              "voc_epsilon": .002, "hint_audit": "structural"}
    plan = {"authorization": {"cap_usd": 75}, "mode": "live",
            "protocol_sha256": "source", "models_sha256": "models",
            "budget_sha256": "budget", "driver_sha256": "driver",
            "soft_stop_usd": 73.0, "optional_reserve_usd": 3.0,
            "calibration": {"train_limit": 16, "dev_limit": 10, "policy_actions": 4,
                            "max_calls": 20, "max_item_cost_usd": .20},
            "deployment_policy": policy,
            "mechanism_holdout_ids": [it.item_id for it in heldout],
            "primary_ablations": ["full", "A1", "A2", "A3", "A4", "A7", "R1"],
            "secondary_ablations": ["full", "A5", "A6", "A8", "C1", "C2"],
            "secondary_ablations_allowance_usd": 1.0}
    calls = []
    behavior = SimpleNamespace(provider_fault=False, seed_resource_cost=2.0,
                               artifact_overlap=False, offline_fault=None, phase_fault=None,
                               unexpected_phase=None, budget_line_fault=None, qwen_rate_limit_failures=0)

    def save_plan():
        plan.pop("sha256", None)
        plan["sha256"] = digest(plan)
        atomic_json(root / "program_plan.json", plan)

    def fake_offline(**options):
        manifest = {'plan_id': 'offline-' + options['roster'], 'mode': 'live', 'experiment': 'offline',
                    'roster': options['roster'], 'policy_config': policy,
                    'train_ids': [it.item_id for it in train], 'dev_ids': [it.item_id for it in tuning],
                    'tools_runtime': {'backend': 'mock-runtime'}, 'protocol_sha256': 'source',
                    'models_sha256': 'models', 'budget_sha256': 'budget'}
        if options.get("estimate"):
            return manifest
        calls.append(("offline_" + options["roster"], options))
        if (behavior.offline_fault == options['roster']
                or options['roster'] == 'xolver' and behavior.qwen_rate_limit_failures):
            key = 'qwen3-32b' if options['roster'] == 'xolver' else 'o3'
            driver.LLMClient.complete(SimpleNamespace(live=True), SimpleNamespace(
                model_key=key, sample_tag='mock-offline-probe', meta={'item_id': train[0].item_id}))
        output = options["output"]
        artifact = {"mode": "live", "policy_config": policy,
                    'roster': options['roster'],
                    "provenance": {**{k: v for k, v in manifest.items() if k not in
                        {'plan_id', 'mode', 'experiment', 'roster', 'policy_config'}},
                        'manifest_plan_id': manifest['plan_id']}}
        if behavior.artifact_overlap:
            artifact["provenance"]["dev_ids"].append(heldout[0].item_id)
        path = output / "artifacts.json"
        artifact = freeze(artifact, path)
        atomic_json(output / 'manifest.json', manifest)
        atomic_json(output / 'execution.json', {'status': 'complete', 'artifact_sha256': artifact['sha256']})
        for name in ('training_evidence.json', 'development_evidence.json'):
            atomic_json(output / name, seal_record({}, plan_id=manifest['plan_id'], mode='live'))
        return path

    def fake_phase(**options):
        name = options["output"].name
        manifest = {'plan_id': 'phase-' + name, 'mode': 'live', 'study': options['study'],
                    'tools_runtime': {'backend': 'mock-runtime'}, 'item_ids': [train[0].item_id],
                    'seeds': options.get('seeds', [0]), 'conditions': options.get('conditions', ['full'])}
        if options['study'] in {'z6', 'z3'}:
            manifest['assignments'] = [{'item_id': iid, 'seed': 0, 'assigned_arm': 'mock-arm',
                                       'executed_arms': ['mock-arm']} for iid in options['item_ids']]
        if options.get("estimate"):
            return manifest
        calls.append((name, options))
        if behavior.budget_line_fault == name:
            experiment = 'e2_xolver' if name == 'e2' else 'e1_sat'
            raise driver.BudgetExceeded(f'{experiment}: reserving $0.0500 exceeds its budget line')
        if behavior.unexpected_phase == name:
            raise RuntimeError('unexpected orchestration fault')
        if (behavior.provider_fault and name == 'e1_primary') or behavior.phase_fault == name:
            key = 'qwen3-32b' if name == 'e2' else 'o3'
            driver.LLMClient.complete(SimpleNamespace(live=True), SimpleNamespace(model_key=key))
        output = options["output"]
        # A resumed runner reports zero newly incurred client spend. The driver
        # must use original logical item costs to budget additional seeds.
        atomic_json(output / 'manifest.json', manifest)
        count = 0
        if options['study'] in {'z6', 'z3'}:
            atomic_json(output / 'assignments.json', seal_record({'assignments': manifest['assignments']},
                        plan_id=manifest['plan_id'], mode='live'))
            for assignment in manifest['assignments']:
                identity = {'item_id': assignment['item_id'], 'seed': assignment['seed']}
                ident = digest([identity['item_id'], identity['seed']])[:24]
                atomic_json(output / 'checkpoints' / f'{ident}.json', seal_record(
                    {**identity, 'checkpoint': {'sha256': 'checkpoint'}}, plan_id=manifest['plan_id'], mode='live'))
                for arm in assignment['executed_arms']:
                    raw = {**identity, 'arm': arm, 'checkpoint_sha256': 'checkpoint'}
                    for directory in ('agent', 'graded'):
                        atomic_json(output / directory / f'{ident}-{arm}.json',
                                    seal_record(raw, plan_id=manifest['plan_id'], mode='live'))
                    count += 1
        else:
            for seed in manifest['seeds']:
                for item_id in manifest['item_ids']:
                    for condition in manifest['conditions']:
                        ident = digest([item_id, seed, condition])[:24]
                        atomic_json(output / 'agent' / f'{ident}.json', seal_record(
                            {'item_id': item_id, 'seed': seed, 'condition': condition}, plan_id=manifest['plan_id'], mode='live'))
                        atomic_json(output / 'graded' / f'{ident}.json', {'cost_usd': behavior.seed_resource_cost})
                        count += 1
        atomic_json(output / "execution.json", {"status": "complete", 'tools_runtime': manifest['tools_runtime'],
                    'observations': count, "client": {"cost_usd": 0.0}})
        return output

    monkeypatch.setattr(driver, "offline_phase", fake_offline)
    monkeypatch.setattr(driver, "run_study", fake_phase)
    monkeypatch.setattr(driver, "run_mechanism", fake_phase)
    monkeypatch.setattr(sys, "argv", [str(source), "--root", str(root)])
    save_plan()
    return SimpleNamespace(driver=driver, root=root, plan=plan, save_plan=save_plan,
                           registry=registry, ledger=ledger, calls=calls, behavior=behavior)


def test_paid_stage_order_complete_seeds_and_exact_mechanism_holdout(program):
    original_complete = program.driver.LLMClient.complete
    program.driver.main()
    names = [name for name, _ in program.calls]
    assert [n for n in names if n not in {'offline_xolver', 'e2'}] == [
        'offline_sat', 'e1_primary', 'ablations_primary', 'z6', 'z3', 'ablations_secondary', 'e1_seed1', 'e1_seed2']
    assert names.index('offline_xolver') < names.index('e2') < names.index('ablations_secondary')
    options = dict(program.calls)
    assert options["e2"]["seeds"] == list(range(8))
    assert options["e1_primary"]["seeds"] == [0]
    assert options["e1_seed1"]["seeds"] == [1]
    assert options["e1_seed2"]["seeds"] == [2]
    for study in ("z6", "z3"):
        assert options[study]["item_ids"] == program.plan["mechanism_holdout_ids"]
        assert options[study]["paired"] is (study == "z6")
    assert options["ablations_primary"]["conditions"] == program.plan["primary_ablations"]
    assert json.loads((program.root / "status.json").read_text())["status"] == "complete"
    assert program.driver.LLMClient.complete is original_complete


def test_provider_fault_persists_and_rejects_restart_before_any_phase(program):
    program.behavior.provider_fault = True
    program.driver.main()
    assert (program.root / "provider_attention_sat.json").exists()
    assert 'e2' in dict(program.calls)
    before = list(program.calls)
    program.driver.main()
    assert program.calls == before
    status = json.loads((program.root / 'status.json').read_text())
    assert status['lanes']['sat']['status'] == 'blocked by provider'
    assert status['lanes']['xolver']['status'] == 'primary complete'
    assert status['phases']['offline_sat']['status'] == 'complete'


def test_qwen_calibration_failure_does_not_delay_sat_and_defers_optional_panels(program, monkeypatch):
    import threading
    reached_e1 = threading.Event()
    original_offline, original_study = program.driver.offline_phase, program.driver.run_study
    program.behavior.offline_fault = 'xolver'
    def offline(**options):
        if options['roster'] == 'xolver' and not options.get('estimate'):
            assert reached_e1.wait(5), 'SAT E1 must start independently of Qwen calibration'
        return original_offline(**options)
    def study(**options):
        if options['output'].name == 'e1_primary' and not options.get('estimate'):
            reached_e1.set()
        return original_study(**options)
    monkeypatch.setattr(program.driver, 'offline_phase', offline)
    monkeypatch.setattr(program.driver, 'run_study', study)
    program.driver.main()
    names = [name for name, _ in program.calls]
    assert all(name in names for name in ('e1_primary', 'ablations_primary', 'z6', 'z3'))
    assert all(name not in names for name in ('e2', 'ablations_secondary', 'e1_seed1', 'e1_seed2'))
    state = json.loads((program.root / 'status.json').read_text())
    assert state['status'] == 'partial collection stopped; review required'
    assert state['execution_mode'] == 'independent_rosters'
    assert state['lane_status'] == {'sat': 'primary complete', 'xolver': 'blocked by provider'}
    assert state['phases']['e1_seed1']['status'] == 'deferred until both primary lanes finish'
    assert (program.root / 'provider_attention_xolver.json').exists()
    # Restart validates all complete SAT artifacts/records without executing a
    # client or retrying any blocked Qwen request. It preserves the first epoch.
    started, previous_calls = state['started_at_utc'], list(program.calls)
    program.driver.main()
    assert program.calls == previous_calls
    resumed = json.loads((program.root / 'status.json').read_text())
    assert resumed['started_at_utc'] == started
    assert resumed['phases']['offline_sat']['validated_completed_resume']
    assert resumed['phases']['e1_primary']['history']


def test_complete_program_resume_validates_all_steps_without_recollecting(program):
    program.driver.main()
    before = list(program.calls)
    program.driver.main()
    assert program.calls == before
    state = json.loads((program.root / 'status.json').read_text())
    assert state['status'] == 'complete'
    assert all(phase['validated_completed_resume'] for phase in state['phases'].values())


def _fake_durable_retry(program, monkeypatch, wait=None):
    from datetime import datetime, timedelta, timezone
    from scripts.provider_retry import ProviderRetry
    clock = SimpleNamespace(value=datetime.now(timezone.utc), waits=[])
    def advance(seconds):
        clock.waits.append(seconds)
        clock.value += timedelta(seconds=seconds)
        if wait:
            wait()
    monkeypatch.setattr(program.driver, 'ProviderRetry', lambda root, **options:
        ProviderRetry(root, **options, clock=lambda: clock.value, wait=advance))
    program.plan['transport_policy'] = {'safe_429_recovery_waves': 0, 'qwen_max_attempts': 1,
        'provider_retry_interval_s': 1800, 'provider_retry_max_waves': 6}
    program.save_plan()
    return clock


def test_durable_qwen_retry_recovers_same_lane_after_cooldown_and_preserves_sat(program, monkeypatch):
    clock = _fake_durable_retry(program, monkeypatch)
    program.behavior.qwen_rate_limit_failures = 1
    program.driver.main()
    names = [name for name, _ in program.calls]
    assert names.count('offline_xolver') == 2
    assert names.count('offline_sat') == names.count('e1_primary') == names.count('z3') == 1
    assert 'e2' in names and 'ablations_secondary' in names
    assert sum(clock.waits) >= 1800
    retry = json.loads((program.root / 'provider_retry_xolver.json').read_text())
    assert retry['waves_used'] == 1
    assert not (program.root / 'provider_attention_xolver.json').exists()
    archived = next((program.root / 'provider_attention_history').glob('*.json'))
    proof = json.loads(archived.read_text())
    assert proof['safe_unbilled_429'] and proof['attempts'] == 1 and proof['cost_usd'] == 0
    assert json.loads((program.root / 'status.json').read_text())['status'] == 'complete'


def test_durable_qwen_retry_bound_survives_restart_without_extra_probes(program, monkeypatch):
    _fake_durable_retry(program, monkeypatch)
    program.behavior.qwen_rate_limit_failures = 10
    program.driver.main()
    names = [name for name, _ in program.calls]
    assert names.count('offline_xolver') == 7  # initial request, then six saved resumption waves
    assert names.count('e1_primary') == names.count('z3') == 1
    state = json.loads((program.root / 'status.json').read_text())
    assert state['lanes']['xolver']['status'] == 'blocked by provider; scheduled retries exhausted'
    assert state['lanes']['xolver']['provider_retry_waves_used'] == 6
    assert 'e2' not in names and 'ablations_secondary' not in names
    before = list(program.calls)
    program.driver.main()
    assert program.calls == before


def test_stop_interrupts_scheduled_retry_without_consuming_a_wave(program, monkeypatch):
    _fake_durable_retry(program, monkeypatch, wait=lambda: (program.root / 'STOP').touch())
    program.behavior.qwen_rate_limit_failures = 1
    with pytest.raises(program.driver.BudgetExceeded):
        program.driver.main()
    assert json.loads((program.root / 'provider_retry_xolver.json').read_text())['waves_used'] == 0
    assert (program.root / 'provider_attention_xolver.json').exists()
    state = json.loads((program.root / 'status.json').read_text())
    assert state['status'] == 'stopped; review required'


def test_ambiguous_provider_error_is_not_given_a_durable_retry(program, monkeypatch):
    clock = _fake_durable_retry(program, monkeypatch)
    program.behavior.offline_fault = 'xolver'
    program.driver.main()
    assert [name for name, _ in program.calls].count('offline_xolver') == 1
    assert not clock.waits and not (program.root / 'provider_retry_xolver.json').exists()


def test_durable_policy_rejects_immediate_retry_waves(program, monkeypatch):
    _fake_durable_retry(program, monkeypatch)
    program.plan['transport_policy']['safe_429_recovery_waves'] = 1
    program.save_plan()
    with pytest.raises(ValueError, match='zero immediate recovery waves'):
        program.driver.main()
    assert not program.calls


def test_completed_record_corruption_fails_before_skipping_that_phase(program):
    program.driver.main()
    saved = next((program.root / 'e1_primary' / 'agent').glob('*.json'))
    value = json.loads(saved.read_text())
    value['item_id'] = 'tampered'
    atomic_json(saved, value)
    before = list(program.calls)
    with pytest.raises(ValueError, match='provenance or content hash changed'):
        program.driver.main()
    assert program.calls == before
    assert json.loads((program.root / 'status.json').read_text())['status'] == 'stopped; review required'


def test_frozen_study_overrides_apply_only_to_declared_studies(program):
    program.plan['study_policy_overrides'] = {study: {'mu': 0.0} for study in ('e1', 'e2', 'ablations')}
    program.save_plan()
    program.driver.main()
    for name, options in program.calls:
        if options.get('study') in {'e1', 'e2', 'ablations'}:
            assert options['mu'] == 0.0
        else:
            assert 'mu' not in options


def test_provider_attention_with_uncertain_billing_blocks_both_lanes(program):
    atomic_json(program.root / 'provider_attention_xolver.json',
                {'model': 'qwen3-32b', 'billing_uncertain': True})
    with pytest.raises(ValueError, match='uncertain provider billing'):
        program.driver.main()
    assert program.calls == []


def test_unexpected_lane_failure_stops_the_program(program):
    program.behavior.unexpected_phase = 'e1_primary'
    with pytest.raises(RuntimeError, match='unexpected orchestration fault'):
        program.driver.main()
    state = json.loads((program.root / 'status.json').read_text())
    assert state['status'] == 'stopped; review required'
    assert state['lanes']['sat']['status'] == 'stopped; review required'
    assert not list(program.root.glob('provider_attention*.json'))


def test_experiment_line_exhaustion_keeps_other_primary_lane_running(program):
    program.behavior.budget_line_fault = 'e2'
    program.driver.main()
    names = dict(program.calls)
    assert all(name in names for name in ('e1_primary', 'ablations_primary', 'z6', 'z3'))
    assert 'ablations_secondary' not in names
    state = json.loads((program.root / 'status.json').read_text())
    assert state['lane_status'] == {'sat': 'primary complete', 'xolver': 'blocked by experiment budget'}
    attention = json.loads((program.root / 'resource_attention_xolver.json').read_text())
    assert attention['experiment'] == 'e2_xolver' and attention['billing_uncertain'] is False
    before = list(program.calls)
    program.driver.main()
    assert program.calls == before


def test_operational_dependency_is_frozen_before_collection(program, monkeypatch):
    program.plan['operational_dependencies_sha256'] = {'provider_diagnostics.py': 'expected'}
    program.save_plan()
    original = program.driver.file_digest
    monkeypatch.setattr(program.driver, 'file_digest', lambda path:
        'changed' if Path(path).name == 'provider_diagnostics.py' else original(path))
    with pytest.raises(ValueError, match='operational dependency changed'):
        program.driver.main()
    assert not program.calls


def test_startup_status_failure_restores_client_and_backend(program, monkeypatch):
    original_client = program.driver.LLMClient.complete
    original_backend = program.driver.OpenRouterSync.complete
    def broken_publish(*args):
        raise OSError('status filesystem unavailable')
    monkeypatch.setattr(program.driver, 'atomic_json', broken_publish)
    with pytest.raises(OSError, match='filesystem unavailable'):
        program.driver.main()
    assert program.driver.LLMClient.complete is original_client
    assert program.driver.OpenRouterSync.complete is original_backend
    assert not program.calls


def test_learning_family_overlap_refuses_before_any_collection(program):
    program.registry["heldout/0"].family_id = program.registry["tuning/0"].family_id
    with pytest.raises(ValueError, match="ten untouched development families"):
        program.driver.main()
    assert program.calls == []


def test_frozen_artifact_overlap_refuses_before_test_collection(program):
    program.behavior.artifact_overlap = True
    with pytest.raises(ValueError, match="overlaps frozen learning provenance"):
        program.driver.main()
    assert {name for name, _ in program.calls} == {"offline_sat", "offline_xolver"}


def test_extra_seeds_use_original_graded_cost_despite_zero_resume_spend(program):
    program.ledger.spent = 64.0
    program.behavior.seed_resource_cost = 8.0
    program.driver.main()
    names = [name for name, _ in program.calls]
    assert "e1_seed1" not in names and "e1_seed2" not in names
    status = json.loads((program.root / "status.json").read_text())
    assert status["phases"]["e1_seed1"]["status"] == "skipped by predeclared budget rule"
    assert status["phases"]["e1_seed1"]["required_allowance_usd"] == pytest.approx(10.8)


def test_optional_phases_obey_experiment_line_even_with_global_capacity(program):
    program.ledger.line_caps = {"e1_sat": 100.0, "e4_ablations": 1.0}
    program.driver.main()
    names = [name for name, _ in program.calls]
    assert "ablations_secondary" not in names
    assert "e1_seed1" in names and "e1_seed2" in names


@pytest.mark.parametrize("field", ["pending", "unreconciled"])
def test_unresolved_billing_refuses_before_collection(program, field):
    setattr(program.ledger, field, 1)
    with pytest.raises(ValueError, match="reconcile outstanding"):
        program.driver.main()
    assert program.calls == []


def _qwen_request(key='qwen3-32b'):
    return SimpleNamespace(model_key=key, sample_tag='unchanged-sample', meta={'item_id': 'train/1'})


def _rate_limit(driver, **changes):
    return driver.LLMResponse(text='', error='failed after 6 attempts: HTTP 429 rate limited',
                              attempts=6, provider='SiliconFlow', model_slug='qwen/qwen3-32b', **changes)


def _guard(program, original, request=None, stopped=None):
    import threading
    return program.driver.guarded_completion(original, SimpleNamespace(live=True),
        request or _qwen_request(), root=program.root,
        stopped=stopped or threading.Event(), ledger=program.ledger, soft_stop_usd=73,
        fault_lock=threading.RLock())


def test_unbilled_rate_limit_recovers_without_changing_success(program):
    driver = program.driver
    success = driver.LLMResponse(text='same result', generation_id='real-success', cost_usd=.001)
    responses = iter([_rate_limit(driver), success])
    assert _guard(program, lambda client, request: next(responses)) is success
    event = json.loads((program.root / 'rate_limit_recoveries.jsonl').read_text())
    assert event['attempts'] == 6 and event['will_retry'] is True
    assert event['cost_usd'] == 0 and 'operational' in event['note'].lower()
    assert not (program.root / 'provider_attention.json').exists()


@pytest.mark.parametrize('changes', [
    {'billing_uncertain': True}, {'cost_usd': .01}, {'tokens_in': 1},
    {'tokens_out': 1}, {'tokens_reasoning': 1}, {'generation_id': 'maybe-billed'},
])
def test_ambiguous_or_nonzero_rate_limit_never_retries(program, changes):
    calls = []
    def original(client, request):
        calls.append(request)
        return _rate_limit(program.driver, **changes)
    with pytest.raises(program.driver.BudgetExceeded, match='provider health gate'):
        _guard(program, original)
    assert len(calls) == 1
    assert not (program.root / 'rate_limit_recoveries.jsonl').exists()


def test_rate_limit_recovery_is_bounded_at_two_extra_waves(program):
    calls = []
    def original(client, request):
        calls.append(request)
        return _rate_limit(program.driver)
    with pytest.raises(program.driver.BudgetExceeded, match='provider health gate'):
        _guard(program, original)
    assert len(calls) == 3
    events = [json.loads(s) for s in (program.root / 'rate_limit_recoveries.jsonl').read_text().splitlines()]
    assert [e['will_retry'] for e in events] == [True, True, False]


def test_queued_cancel_preserves_first_attention_and_releases_reservation_response(program):
    original = {'error': 'first provider fault', 'generation_id': 'preserve-me'}
    atomic_json(program.root / 'provider_attention.json', original)
    response = program.driver.LLMResponse(text='', error=program.driver.QUEUE_CANCELLED)
    with pytest.raises(program.driver.BudgetExceeded, match='no request sent'):
        _guard(program, lambda client, request: response)
    assert json.loads((program.root / 'provider_attention.json').read_text()) == original


def test_cached_response_bypasses_transport_and_is_returned_unchanged(program):
    response = program.driver.LLMResponse(text='cached answer', cached=True)
    # LLMClient's cache hit returns before backend.complete. The guard must not
    # add a transport delay or alter that successful object.
    assert _guard(program, lambda client, request: response) is response
    assert response.cached and response.cost_usd == response.attempts == 0


def test_shared_qwen_variants_are_serialized_with_completion_spacing(program):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    clock = SimpleNamespace(value=0.)
    clock_lock = threading.Lock()
    def wait(seconds):
        with clock_lock:
            clock.value += seconds
        time.sleep(.0001)
    pacer = program.driver.SiliconFlowPacer(program.root, threading.Event(),
        clock=lambda: clock.value, wait=wait)
    active, maximum, starts = 0, 0, []
    counter_lock = threading.Lock()
    def original(backend, settings, messages, response_format, spec):
        nonlocal active, maximum
        with counter_lock:
            active += 1
            maximum = max(maximum, active)
            starts.append(clock.value)
        time.sleep(.002)
        with counter_lock:
            active -= 1
        return program.driver.LLMResponse(text=settings.model_key), 1
    def one(key):
        settings = SimpleNamespace(model_key=key, provider='SiliconFlow', slug='qwen/qwen3-32b')
        return pacer.complete(original, None, settings, [], None, None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one, ['qwen3-32b', 'qwen3-32b-nothink']))
    assert maximum == 1 and starts[1] - starts[0] >= 10
    assert {r[0].text for r in results} == {'qwen3-32b', 'qwen3-32b-nothink'}


def test_stop_during_queue_wait_returns_definitely_unbilled_and_unlocks(program):
    import threading
    stopped = threading.Event()
    pacer = program.driver.SiliconFlowPacer(program.root, stopped,
        wait=lambda seconds: (program.root / 'STOP').touch())
    settings = SimpleNamespace(provider='SiliconFlow', slug='qwen/qwen3-32b')
    pacer._slot.acquire()
    try:
        response, attempts = pacer.complete(lambda *args: pytest.fail('must not dispatch'),
                                             None, settings, [], None, None)
    finally:
        pacer._slot.release()
    assert attempts == 0 and response.error == program.driver.QUEUE_CANCELLED
    assert not response.billing_uncertain and response.cost_usd == 0
    assert pacer._slot.acquire(blocking=False)
    pacer._slot.release()


def test_pacer_applies_global_180_second_cooldown_and_releases_on_exception(program):
    import threading
    clock = SimpleNamespace(value=0.)
    pacer = program.driver.SiliconFlowPacer(program.root, threading.Event(),
        clock=lambda: clock.value, wait=lambda seconds: setattr(clock, 'value', clock.value + seconds))
    settings = SimpleNamespace(provider='SiliconFlow', slug='qwen/qwen3-32b')
    pacer.complete(lambda *args: (_rate_limit(program.driver), 6), None, settings, [], None, None)
    starts = []
    def crash(*args):
        starts.append(clock.value)
        raise RuntimeError('backend exception')
    with pytest.raises(RuntimeError, match='backend exception'):
        pacer.complete(crash, None, settings, [], None, None)
    assert starts == [180]
    assert pacer._slot.acquire(blocking=False)
    pacer._slot.release()


def test_lane_stop_during_qwen_cooldown_cancels_without_dispatch(program):
    import threading
    stopped, lane_stopped = threading.Event(), threading.Event()
    pacer = program.driver.SiliconFlowPacer(program.root, stopped,
        clock=lambda: 0., wait=lambda seconds: lane_stopped.set(), dispatch_blocked=lane_stopped.is_set)
    pacer._next_at = 180
    settings = SimpleNamespace(provider='SiliconFlow', slug='qwen/qwen3-32b')
    response, attempts = pacer.complete(lambda *args: pytest.fail('blocked roster must not dispatch'),
                                        None, settings, [], None, None)
    assert attempts == 0 and response.error == program.driver.QUEUE_CANCELLED
    assert not response.billing_uncertain


def test_provider_retry_after_can_extend_the_pacer_cooldown(program):
    import threading
    clock = SimpleNamespace(value=0.)
    pacer = program.driver.SiliconFlowPacer(program.root, threading.Event(),
        clock=lambda: clock.value, wait=lambda seconds: setattr(clock, 'value', clock.value + seconds))
    settings = SimpleNamespace(provider='SiliconFlow', slug='qwen/qwen3-32b')
    def response_with_retry_hint(*args):
        pacer._next_at = 600.
        return _rate_limit(program.driver), 6
    pacer.complete(response_with_retry_hint, None, settings, [], None, None)
    assert pacer._next_at == 600.


def test_billing_uncertainty_sets_global_event_and_prevents_other_roster_dispatch(program):
    import threading
    global_stop, sat_stop, qwen_stop = threading.Event(), threading.Event(), threading.Event()
    response = program.driver.LLMResponse(text='', error='ambiguous transport failure', billing_uncertain=True)
    with pytest.raises(program.driver.BudgetExceeded, match='provider health gate'):
        program.driver.guarded_completion(lambda *args: response, SimpleNamespace(live=True),
            SimpleNamespace(model_key='o3'), root=program.root, stopped=sat_stop, global_stopped=global_stop,
            ledger=program.ledger, soft_stop_usd=73, fault_lock=threading.RLock(),
            attention_name='provider_attention_sat.json')
    assert global_stop.is_set() and sat_stop.is_set() and not qwen_stop.is_set()
    with pytest.raises(program.driver.BudgetExceeded, match='stop requested'):
        program.driver.guarded_completion(lambda *args: pytest.fail('global stop must prevent Qwen dispatch'),
            SimpleNamespace(live=True), _qwen_request(), root=program.root, stopped=qwen_stop,
            global_stopped=global_stop, ledger=program.ledger, soft_stop_usd=73, fault_lock=threading.RLock())
