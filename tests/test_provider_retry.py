"""Fake-clock recovery tests; no sleeping, clients, credentials or API calls."""
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from mathmage.experiments.artifacts import atomic_json
from scripts.provider_retry import ProviderRetry, recoverable_attention


@pytest.fixture
def retry(tmp_path):
    clock = SimpleNamespace(value=datetime(2026, 10, 18, 14, 0, tzinfo=timezone.utc), waits=[])
    def wait(seconds):
        clock.waits.append(seconds)
        clock.value += timedelta(seconds=seconds)
    helper = ProviderRetry(tmp_path, clock=lambda: clock.value, wait=wait)
    attention = {'time_utc': clock.value.isoformat(), 'model': 'qwen3-32b',
        'error': 'failed after 1 attempts: HTTP 429 rate limited', 'attempts': 1,
        'safe_unbilled_429': True, 'cost_usd': 0., 'tokens_in': 0, 'tokens_out': 0,
        'tokens_reasoning': 0, 'billing_uncertain': False, 'generation_id': None}
    atomic_json(helper.attention_path, attention)
    return SimpleNamespace(helper=helper, clock=clock, attention=attention, root=tmp_path, wait=wait)


@pytest.mark.parametrize('changes', [
    {'safe_unbilled_429': False}, {'billing_uncertain': True}, {'cost_usd': .0001},
    {'tokens_in': 1}, {'tokens_out': 1}, {'tokens_reasoning': 1}, {'generation_id': 'maybe-billed'},
    {'error': 'HTTP 503'}, {'error': 'failed after 6 attempts: HTTP 429 rate limited'},
    {'model': 'haiku-4.5'}, {'attempts': True}, {'time_utc': '2026-10-18T14:00:00'},
])
def test_ambiguous_or_non429_attention_never_schedules(retry, changes):
    attention = {**retry.attention, **changes}
    assert not recoverable_attention(attention)
    assert not retry.helper.wait_and_rearm(attention, stop_requested=lambda: False)
    assert retry.clock.waits == [] and not retry.helper.path.exists()


def test_missing_proof_is_not_inferred_from_zero_defaults(retry):
    attention = {key: value for key, value in retry.attention.items() if key != 'tokens_reasoning'}
    assert not recoverable_attention(attention)
    assert retry.helper.schedule(attention) is None


def test_wait_is_from_original_attention_time_and_archives_exact_proof(retry):
    retry.clock.value += timedelta(seconds=600)
    waiting = []
    assert retry.helper.wait_and_rearm(retry.attention, stop_requested=lambda: False, on_wait=waiting.append)
    assert sum(retry.clock.waits) == 1200 and max(retry.clock.waits) <= 1
    assert waiting[0]['next_retry_at_utc'] == '2026-10-18T14:30:00+00:00'
    state = retry.helper.snapshot()
    assert state['waves_used'] == 1 and state['status'] == 'retrying'
    assert state['next_retry_at_utc'] == '2026-10-18T15:00:00+00:00'
    assert not retry.helper.attention_path.exists()
    assert json.loads((retry.root / state['attention_archive']).read_text()) == retry.attention


def test_provider_retry_after_can_extend_but_never_shorten_fixed_interval(retry):
    state = retry.helper.schedule(retry.attention, provider_delay_s=60)
    assert state['next_retry_at_utc'] == '2026-10-18T14:30:00+00:00'
    state = retry.helper.schedule(retry.attention, provider_delay_s=3600)
    assert state['next_retry_at_utc'] == '2026-10-18T15:00:00+00:00'
    state = retry.helper.schedule(retry.attention, provider_delay_s=0)
    assert state['next_retry_at_utc'] == '2026-10-18T15:00:00+00:00'


def test_restart_cannot_reset_deadline_or_counter_or_exceed_six_waves(retry):
    for wave in range(1, 7):
        helper = ProviderRetry(retry.root, clock=lambda: retry.clock.value, wait=retry.wait)
        attention = {**retry.attention, 'time_utc': retry.clock.value.isoformat()}
        atomic_json(helper.attention_path, attention)
        scheduled = helper.schedule(attention)
        restarted = ProviderRetry(retry.root, clock=lambda: retry.clock.value, wait=retry.wait)
        assert restarted.schedule(attention)['next_retry_at_utc'] == scheduled['next_retry_at_utc']
        assert restarted.wait_and_rearm(attention, stop_requested=lambda: False)
        assert restarted.snapshot()['waves_used'] == wave
        restarted.mark_recovered()
        assert restarted.snapshot()['waves_used'] == wave
    final = ProviderRetry(retry.root, clock=lambda: retry.clock.value, wait=retry.wait)
    attention = {**retry.attention, 'time_utc': retry.clock.value.isoformat()}
    atomic_json(final.attention_path, attention)
    before = len(retry.clock.waits)
    assert not final.wait_and_rearm(attention, stop_requested=lambda: False)
    assert final.snapshot()['status'] == 'exhausted'
    assert len(retry.clock.waits) == before and final.attention_path.exists()


def test_stop_during_wait_does_not_consume_wave_or_clear_attention(retry):
    assert not retry.helper.wait_and_rearm(retry.attention,
        stop_requested=lambda: len(retry.clock.waits) >= 3)
    assert len(retry.clock.waits) == 3
    assert retry.helper.snapshot()['waves_used'] == 0
    assert retry.helper.attention_path.exists()


def test_crash_after_rearming_requires_next_gap_before_another_wave(retry):
    assert retry.helper.wait_and_rearm(retry.attention, stop_requested=lambda: False)
    resumed = ProviderRetry(retry.root, clock=lambda: retry.clock.value, wait=retry.wait)
    proof = resumed.attention_for_resume()
    assert proof == retry.attention
    assert resumed.wait_and_rearm(proof, stop_requested=lambda: False)
    assert retry.clock.value == datetime(2026, 10, 18, 15, 0, tzinfo=timezone.utc)
    assert resumed.snapshot()['waves_used'] == 2
    resumed.mark_recovered()
    assert resumed.attention_for_resume() is None


def test_corrupt_counter_fails_closed(retry):
    retry.helper.schedule(retry.attention)
    state = json.loads(retry.helper.path.read_text())
    state['waves_used'] = 0
    state['max_waves'] = 100
    atomic_json(retry.helper.path, state)
    with pytest.raises(ValueError, match='retry state'):
        retry.helper.snapshot()


def test_attention_changed_during_wait_is_not_cleared(retry):
    def wait(seconds):
        retry.clock.value += timedelta(seconds=1800)
        atomic_json(retry.helper.attention_path, {**retry.attention, 'error': 'different provider fault'})
    helper = ProviderRetry(retry.root, clock=lambda: retry.clock.value, wait=wait)
    with pytest.raises(ValueError, match='attention changed'):
        helper.wait_and_rearm(retry.attention, stop_requested=lambda: False)
    assert helper.snapshot()['waves_used'] == 0 and helper.attention_path.exists()


def test_deadline_extended_by_another_scheduler_is_rechecked_under_lock(retry):
    extended = []
    def wait(seconds):
        retry.wait(seconds)
        if not extended:
            extended.append(True)
            other = ProviderRetry(retry.root, clock=lambda: retry.clock.value, wait=retry.wait)
            other.schedule(retry.attention, provider_delay_s=3600)
    helper = ProviderRetry(retry.root, clock=lambda: retry.clock.value, wait=wait)
    assert helper.wait_and_rearm(retry.attention, stop_requested=lambda: False)
    assert retry.clock.value == datetime(2026, 10, 18, 15, 0, 1, tzinfo=timezone.utc)
    assert helper.snapshot()['waves_used'] == 1


@pytest.mark.parametrize('options', [{'interval_s': 1}, {'interval_s': float('nan')},
    {'interval_s': float('inf')}, {'max_waves': 7}, {'max_waves': True}])
def test_invalid_policy_is_rejected(retry, options):
    with pytest.raises(ValueError):
        ProviderRetry(retry.root, **options)
