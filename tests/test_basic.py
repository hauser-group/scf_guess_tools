from provider import engine


def test_singleton(engine):
    assert engine.__class__() == engine, "engines must be singletons"
