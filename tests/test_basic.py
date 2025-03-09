from __future__ import annotations

from provider import context, engine


def test_singleton(engine):
    assert engine.__class__() == engine, "engines must be singletons"
