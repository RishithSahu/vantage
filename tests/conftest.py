"""Every test starts with the environment the previous one left behind put back: the launcher's `main` adopts the product's model by setting VGPT_FRAME / VGPT_MEANING_ADAPTER, and a test that called it
would otherwise change what every later test measures (stub planners written for frame version 1 failed only when the whole suite ran)."""
import os

import pytest


@pytest.fixture(autouse=True)
def _restore_environment():
    before = dict(os.environ)
    yield
    for k in [k for k in os.environ if k not in before]: os.environ.pop(k, None)
    for k, v in before.items():
        if os.environ.get(k) != v: os.environ[k] = v


@pytest.fixture(autouse=True, scope="module")
def _release_the_adopted_vehicle():
    """A test module that opened a vehicle database (the assistant adopts that vehicle's vocabulary, which hides the concepts it lacks) must not change what the next module sees: the canonical registry is put back."""
    yield
    try:
        from vgpt.vehicle import lake
        lake.release()
    except Exception: pass
