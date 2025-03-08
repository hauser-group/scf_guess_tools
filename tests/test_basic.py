from provider import engine


def test_singleton(engine):
    cls = engine.__class__
    assert cls() == engine, "engines must be singletons"
    original = engine
