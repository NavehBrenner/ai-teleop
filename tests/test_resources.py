"""safe_worker_count clamps to RAM and never drops below 1."""

from __future__ import annotations

from ai_teleop.common import resources
from ai_teleop.common.resources import safe_worker_count


def test_sequential_short_circuits(monkeypatch):
    # workers<=1 must not even read memory (the untouched sequential path).
    monkeypatch.setattr(resources, "_available_gb", lambda: 0.0)
    assert safe_worker_count(1) == 1
    assert safe_worker_count(0) == 1


def test_clamped_when_ram_tight(monkeypatch):
    # 5 GB available, 3 GB reserve → 2 GB budget → 1 worker at 2 GB each.
    monkeypatch.setattr(resources, "_available_gb", lambda: 5.0)
    assert safe_worker_count(8) == 1


def test_uncapped_when_ram_plenty(monkeypatch):
    # 23 GB available → 20 GB budget → 10 workers allowed, so 4 passes through.
    monkeypatch.setattr(resources, "_available_gb", lambda: 23.0)
    assert safe_worker_count(4) == 4


def test_never_below_one(monkeypatch):
    monkeypatch.setattr(resources, "_available_gb", lambda: 0.0)
    assert safe_worker_count(8) == 1
