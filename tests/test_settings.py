from app.config.settings import get_settings


def test_settings_load() -> None:
    settings = get_settings()

    assert settings.app_name == "agentic-true-vectorless-rag"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.retrieval_top_k == 10
    assert settings.retrieval_max_pages == 20
    assert settings.ocr_enabled is True