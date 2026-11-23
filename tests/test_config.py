from assistant.config import Settings, get_settings
from assistant.models import Citation


def test_settings_defaults(monkeypatch):
    monkeypatch.chdir("/")  # no .env picked up
    monkeypatch.delenv("TOP_K", raising=False)
    settings = Settings()
    assert settings.top_k == 6
    assert settings.daily_cap == 200
    assert settings.embed_model == "intfloat/multilingual-e5-small"


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("TOP_K", "3")
    get_settings.cache_clear()
    try:
        assert get_settings().top_k == 3
    finally:
        get_settings.cache_clear()


def _citation(**kwargs) -> Citation:
    base = dict(
        regulation="AI_ACT",
        type="article",
        number="6",
        paragraph=None,
        title=None,
        source_url="https://eur-lex.europa.eu/",
        language="en",
    )
    return Citation(**{**base, **kwargs})


def test_citation_label():
    assert _citation(paragraph="2").label == "AI Act Art. 6(2)"
    assert _citation().label == "AI Act Art. 6"
    assert _citation(regulation="GDPR", type="recital", number="26").label == "GDPR Recital 26"
    assert _citation(type="annex", number="III", paragraph="4").label == "AI Act Annex III point 4"


def test_api_keys_are_secret_and_not_in_repr(monkeypatch):
    monkeypatch.chdir("/")  # no .env picked up
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key-123")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-456")
    settings = Settings()
    assert "fake-google-key-123" not in repr(settings)
    assert "fake-openai-key-456" not in str(settings)
    assert settings.google_api_key.get_secret_value() == "fake-google-key-123"
