"""Durable, bounded recovery for explicitly unbilled upstream Qwen rate limits.

This operational helper never constructs a client or sends a request. A caller
must retain the normal ledger/source guards and clear its roster event only after
``wait_and_rearm`` returns True. Counters are shared by the whole program root,
including restarts and successful intermediate recovery.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mathmage.experiments.artifacts import atomic_json, digest
from mathmage.llm.locking import file_lock


def utc_now():
    return datetime.now(timezone.utc)


def _timestamp(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError('provider retry timestamps require an explicit UTC offset')
    return stamp.astimezone(timezone.utc)


def recoverable_attention(attention):
    """Require the response proof, never merely a label or an assumed free error."""
    try:
        match = re.fullmatch(r'failed after ([1-9][0-9]*) attempts: HTTP 429 rate limited',
                             attention.get('error') or '')
        return (attention.get('model') in {'qwen3-32b', 'qwen3-32b-nothink'}
                and attention.get('safe_unbilled_429') is True
                and bool(match) and type(attention['attempts']) is int
                and attention['attempts'] == int(match.group(1))
                and attention['billing_uncertain'] is False
                and attention['generation_id'] is None
                and all(type(attention[key]) in (int, float) and attention[key] == 0
                        for key in ('cost_usd', 'tokens_in', 'tokens_out', 'tokens_reasoning'))
                and bool(_timestamp(attention['time_utc'])))
    except (KeyError, TypeError, ValueError):
        return False


class ProviderRetry:
    def __init__(self, root, *, interval_s=1800, max_waves=6, clock=utc_now, wait=None):
        if type(interval_s) not in (int, float) or not 1800 <= interval_s < float('inf'):
            raise ValueError('provider retry interval must be at least 1800 seconds')
        if type(max_waves) is not int or not 0 <= max_waves <= 6:
            raise ValueError('provider retry requires a program-wide bound of at most six waves')
        self.root, self.interval_s, self.max_waves = Path(root), interval_s, max_waves
        self.clock, self.wait = clock, wait or threading.Event().wait
        self.path = self.root / 'provider_retry_xolver.json'
        self.attention_path = self.root / 'provider_attention_xolver.json'
        self.lock_path = self.root / '.provider_retry_xolver.lock'

    def _load(self):
        if not self.path.exists():
            return {'schema_version': 1, 'roster': 'xolver', 'interval_s': self.interval_s,
                    'max_waves': self.max_waves, 'waves_used': 0, 'status': 'idle'}
        state = json.loads(self.path.read_text(encoding='utf-8'))
        expected_hash = state.pop('sha256', None)
        if (expected_hash != digest(state) or state.get('schema_version') != 1
                or state.get('roster') != 'xolver' or state.get('interval_s') != self.interval_s
                or state.get('max_waves') != self.max_waves
                or type(state.get('waves_used')) is not int or not 0 <= state['waves_used'] <= self.max_waves
                or state.get('status') not in {'idle', 'waiting', 'retrying', 'recovered', 'exhausted'}):
            raise ValueError('provider retry state or frozen policy changed; review before dispatch')
        if 'last_attention' in state and not recoverable_attention(state['last_attention']):
            raise ValueError('provider retry state lacks valid zero-charge response proof')
        for key in ('next_retry_at_utc', 'last_retry_at_utc', 'updated_at_utc'):
            if key in state:
                _timestamp(state[key])
        return state

    def _save(self, state):
        state = {**state, 'updated_at_utc': self.clock().isoformat()}
        atomic_json(self.path, {**state, 'sha256': digest(state)})
        return state

    def snapshot(self):
        with file_lock(self.lock_path):
            return self._load()

    def attention_for_resume(self):
        """A crash after reserving a retry wave must not permit an immediate probe."""
        with file_lock(self.lock_path):
            state = self._load()
            if state['status'] == 'retrying' and not self.attention_path.exists():
                return state.get('last_attention')
        return None

    def mark_recovered(self):
        """Only a real successful Qwen response can release the crash-recovery gate."""
        with file_lock(self.lock_path):
            state = self._load()
            if state['status'] == 'retrying' and not self.attention_path.exists():
                self._save({**state, 'status': 'recovered'})

    def schedule(self, attention, *, provider_delay_s=0):
        if not recoverable_attention(attention):
            return None
        if type(provider_delay_s) not in (int, float) or not 0 <= provider_delay_s < float('inf'):
            raise ValueError('provider backoff must be finite and nonnegative')
        with file_lock(self.lock_path):
            state = self._load()
            if state['waves_used'] >= self.max_waves:
                return self._save({**state, 'status': 'exhausted', 'last_attention': attention})
            fingerprint = digest(attention)
            # Restarting the same waiting wave reuses its deadline. A fresh hint
            # can extend it; neither restart nor short Retry-After can shorten it.
            earliest = _timestamp(attention['time_utc']) + timedelta(seconds=self.interval_s)
            if state.get('next_retry_at_utc'):
                earliest = max(earliest, _timestamp(state['next_retry_at_utc']))
            if provider_delay_s:
                earliest = max(earliest, self.clock() + timedelta(seconds=provider_delay_s))
            return self._save({**state, 'status': 'waiting', 'last_attention': attention,
                'attention_sha256': fingerprint, 'next_retry_at_utc': earliest.isoformat()})

    def wait_and_rearm(self, attention, *, stop_requested, on_wait=None, provider_delay_s=0):
        """Wait, durably consume one wave, archive proof, then permit one resumption.

        The caller retries the same saved phase. Its first noncached request is
        the health probe; this helper never sends a separate paid probe.
        """
        state = self.schedule(attention, provider_delay_s=provider_delay_s)
        if state is None or state['status'] == 'exhausted':
            return False
        on_wait = on_wait or (lambda state: None)
        on_wait(state.copy())
        deadline = _timestamp(state['next_retry_at_utc'])
        while self.clock() < deadline:
            if stop_requested():
                return False
            self.wait(min(1., (deadline - self.clock()).total_seconds()))
        if stop_requested():
            return False
        with file_lock(self.lock_path):
            current = self._load()
            if (current.get('attention_sha256') != digest(attention)
                    or current['status'] != 'waiting' or current['waves_used'] >= self.max_waves):
                raise ValueError('provider retry state changed while waiting')
            if self.attention_path.exists():
                actual = json.loads(self.attention_path.read_text(encoding='utf-8'))
                if actual != attention:
                    raise ValueError('provider attention changed while waiting; review the new response')
            # A second reviewed scheduler can extend this deadline. Recheck it
            # under the same lock as the wave reservation, never just in memory.
            extended = _timestamp(current['next_retry_at_utc']) > self.clock()
            if not extended:
                wave = current['waves_used'] + 1
                archive = self.root / 'provider_attention_history' / f'xolver_wave{wave:02d}_{digest(attention)[:16]}.json'
                if archive.exists() and json.loads(archive.read_text(encoding='utf-8')) != attention:
                    raise ValueError('archived provider attention changed')
                # Persist BEFORE clearing the attention or allowing a dispatch. A
                # crash here consumes this wave and preserves the next minimum gap.
                self._save({**current, 'status': 'retrying', 'waves_used': wave,
                    'last_retry_at_utc': self.clock().isoformat(),
                    'next_retry_at_utc': (self.clock() + timedelta(seconds=self.interval_s)).isoformat(),
                    'attention_archive': str(archive.relative_to(self.root))})
                archive.parent.mkdir(parents=True, exist_ok=True)
                if self.attention_path.exists():
                    self.attention_path.replace(archive)
                elif not archive.exists():
                    atomic_json(archive, attention)
        if extended:
            return self.wait_and_rearm(attention, stop_requested=stop_requested, on_wait=on_wait)
        return True
