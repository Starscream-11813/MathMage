"""SAT-only recovery uses fake plans, ledger, models, and client; no API calls."""
import contextlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathmage.experiments.artifacts import atomic_json, digest


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'scripts' / 'run_sat_recovery.py'
    spec = importlib.util.spec_from_file_location('sat_recovery_test', source)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    root = tmp_path / 'program'
    root.mkdir()
    outputs, logs, runs = (tmp_path / name for name in ('outputs', 'logs', 'runs'))
    outputs.mkdir()
    (logs / 'calls').mkdir(parents=True)
    runs.mkdir()
    for name, value in [('OUTPUTS', outputs), ('LOGS', logs), ('RUNS', runs)]:
        monkeypatch.setattr(driver.paths, name, value)
    monkeypatch.setattr(driver, '_protocol_digest', lambda: 'protocol')
    monkeypatch.setattr(driver, 'file_digest', lambda p: {
        'models.yaml': 'models', 'budget.yaml': 'budget', 'run_sat_recovery.py': 'wrapper',
        'run_paid_program.py': 'driver'}[Path(p).name])
    plan = {'mode': 'live', 'authorization': {'cap_usd': 75}, 'soft_stop_usd': 73,
            'protocol_sha256': 'protocol', 'models_sha256': 'models', 'budget_sha256': 'budget',
            'driver_sha256': 'driver',
            'calibration': {'train_limit': 16, 'dev_limit': 10, 'extra_samples': 1}}
    plan['sha256'] = digest(plan)
    atomic_json(root / 'program_plan.json', plan)
    attention = {'time_utc': '2026-10-18T11:38:28+00:00', 'model': 'qwen3-32b',
                 'error': 'failed after 6 attempts: HTTP 429 rate limited',
                 'billing_uncertain': False, 'generation_id': None}
    atomic_json(root / 'provider_attention.json', attention)
    row = {'timestamp': attention['time_utc'], 'model_key': attention['model'], 'error': attention['error'],
           'attempts': 6, 'billing_uncertain': False, 'generation_id': None,
           'cost_usd': 0, 'tokens_in': 0, 'tokens_out': 0, 'tokens_reasoning': 0}
    (logs / 'calls' / 'offline.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    manifest = {'plan_id': 'same-offline-plan', 'immutable': 'unchanged'}
    atomic_json(root / 'offline_sat' / 'manifest.json', manifest)
    atomic_json(root / 'offline_sat_review.json', {'plan': {'plan_id': manifest['plan_id']}})
    atomic_json(root / 'status.json', {'phases': {'prior_phase': {'status': 'complete'},
        'offline_xolver': {'status': 'stopped', 'plan_id': 'keep-xolver-plan'}}, 'active_phases': []})
    balance = {'spent_total': 1.75, 'pending_total': 0., 'pending_calls': 0,
               'unreconciled_calls': 0, 'global_cap': 75}
    monkeypatch.setattr(driver, 'BudgetLedger', lambda: SimpleNamespace(summary=lambda: balance.copy()))
    roster = SimpleNamespace(members=['o3', 'haiku'], checker='haiku', card_writer='haiku', default_judge='o3')
    monkeypatch.setattr(driver, 'load_models', lambda: SimpleNamespace(rosters={'sat': roster}))
    behavior = SimpleNamespace(request_model=None, response_error=None, stop_before_request=False,
                               change_manifest=False)
    calls = []
    class Client:
        @staticmethod
        def complete(client, request):
            calls.append(('backend', request.model_key))
            return driver.LLMResponse(text='ok', error=behavior.response_error)
    monkeypatch.setattr(driver, 'LLMClient', Client)
    def offline(**options):
        calls.append(('estimate' if options.get('estimate') else 'execute', options))
        if options.get('estimate'):
            return {**manifest, 'budget_review': {}} if not behavior.change_manifest else {'plan_id': 'changed'}
        status = json.loads((root / 'status.json').read_text())
        assert status['active_phases'] == ['offline_sat']
        assert status['phases']['offline_xolver']['status'].startswith('blocked')
        assert list((root / 'sat_recovery').glob('review_*.json'))
        if behavior.stop_before_request:
            (root / 'STOP').touch()
        if behavior.request_model:
            driver.LLMClient.complete(SimpleNamespace(live=True), SimpleNamespace(model_key=behavior.request_model))
        return root / 'offline_sat' / 'artifacts.json'
    monkeypatch.setattr(driver, 'offline_phase', offline)
    return SimpleNamespace(driver=driver, root=root, attention=attention, balance=balance,
                           plan=plan, calls=calls, behavior=behavior)


def test_sat_resume_preserves_manifest_plan_and_qwen_block(recovery):
    r = recovery
    original = r.driver.LLMClient.complete
    r.driver.run(r.root)
    assert r.driver.LLMClient.complete is original
    assert json.loads((r.root / 'provider_attention.json').read_text()) == r.attention
    execution = dict(r.calls)['execute']
    assert execution['approved_plan'] == 'same-offline-plan'
    assert execution['output'] == r.root / 'offline_sat'
    assert execution['roster'] == 'sat' and execution['train_limit'] == 16
    state = json.loads((r.root / 'status.json').read_text())
    assert state['status'] == 'waiting for Qwen provider recovery'
    assert state['program_plan_sha256'] == r.plan['sha256']
    assert state['phases']['prior_phase']['status'] == 'complete'
    assert state['phases']['offline_xolver']['plan_id'] == 'keep-xolver-plan'
    assert state['phases']['offline_sat']['status'] == 'complete'


@pytest.mark.parametrize('field', ['pending_calls', 'unreconciled_calls'])
def test_outstanding_billing_refuses_before_estimate(recovery, field):
    recovery.balance[field] = 1
    with pytest.raises(ValueError, match='reconcile pending'):
        recovery.driver.run(recovery.root)
    assert not recovery.calls


def test_global_stop_and_soft_spend_guard_prevent_dispatch(recovery):
    recovery.balance['spent_total'] = 73
    with pytest.raises(ValueError, match='dispatch capacity'):
        recovery.driver.run(recovery.root)
    assert not recovery.calls
    (recovery.root / 'STOP').touch()
    with pytest.raises(ValueError, match='global STOP'):
        recovery.driver.run(recovery.root)


def test_changed_offline_plan_cannot_resume(recovery):
    recovery.behavior.change_manifest = True
    with pytest.raises(ValueError, match='exact previously approved'):
        recovery.driver.run(recovery.root)
    assert [name for name, _ in recovery.calls] == ['estimate']


def test_sat_fault_has_separate_health_record_preserving_qwen(recovery):
    r = recovery
    r.behavior.request_model, r.behavior.response_error = 'o3', 'HTTP 503'
    with pytest.raises(r.driver.BudgetExceeded, match='provider health gate'):
        r.driver.run(r.root)
    assert json.loads((r.root / 'provider_attention.json').read_text()) == r.attention
    assert json.loads((r.root / 'sat_recovery' / 'provider_attention.json').read_text())['error'] == 'HTTP 503'


def test_qwen_request_and_stop_before_request_never_reach_backend(recovery):
    r = recovery
    r.behavior.request_model = 'qwen3-32b'
    with pytest.raises(r.driver.BudgetExceeded, match='outside its frozen roster'):
        r.driver.run(r.root)
    assert not any(name == 'backend' for name, _ in r.calls)
    r.behavior.request_model, r.behavior.stop_before_request = 'o3', True
    with pytest.raises(r.driver.BudgetExceeded, match='collection stop requested'):
        r.driver.run(r.root)
    assert not any(name == 'backend' for name, _ in r.calls)


def test_main_takes_global_lock_before_any_guards(recovery, monkeypatch):
    r = recovery
    order = []
    @contextlib.contextmanager
    def lock(path):
        assert path == r.driver.paths.RUNS / '.paid_program.lock'
        order.append('locked')
        yield
        order.append('released')
    monkeypatch.setattr(r.driver, 'file_lock', lock)
    monkeypatch.setattr(r.driver, 'run', lambda root: order.append('guards'))
    monkeypatch.setattr(sys, 'argv', ['run_sat_recovery.py', '--root', str(r.root)])
    r.driver.main()
    assert order == ['locked', 'guards', 'released']


def test_attention_must_have_matching_definitively_unbilled_call(recovery):
    r = recovery
    log = r.driver.paths.LOGS / 'calls' / 'offline.jsonl'
    row = json.loads(log.read_text())
    row['cost_usd'] = .01
    log.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='definitively zero-charge'):
        r.driver.run(r.root)
    assert not r.calls
