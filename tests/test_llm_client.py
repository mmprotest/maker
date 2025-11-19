from maker_mdap.config import LLMConfig
from maker_mdap.llm_client import FakeLLMClient, LLMClient


def test_llm_client_uses_base_url(monkeypatch):
    created = {}

    class DummyOpenAI:
        def __init__(self, api_key=None, base_url=None):
            created["api_key"] = api_key
            created["base_url"] = base_url

        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class Msg:
                        content = "ok"

                    class Choice:
                        message = Msg()

                    class Resp:
                        choices = [Choice()]

                    return Resp()

    monkeypatch.setattr("maker_mdap.llm_client.openai.OpenAI", DummyOpenAI)
    config = LLMConfig(model="dummy", api_key="key", base_url="http://example")
    client = LLMClient(config)
    client.generate_step_response("sys", "user")
    assert created["api_key"] == "key"
    assert created["base_url"] == "http://example"


def test_fake_llm_client_returns_response():
    responder = lambda sys, user: f"SYS:{sys}|USER:{user}"
    client = FakeLLMClient(responder)
    out = client.generate_step_response("s", "u")
    assert out == "SYS:s|USER:u"

