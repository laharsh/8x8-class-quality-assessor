import os

import requests
from dotenv import load_dotenv

from class_assessor.config import PROJECT_ROOT

_loaded = False


def load_project_env() -> None:
    global _loaded
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=True)
    _loaded = True


def get_openai_api_key() -> str:
    load_project_env()
    raw = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
    key = raw.strip()
    if (key.startswith('"') and key.endswith('"')) or (
        key.startswith("'") and key.endswith("'")
    ):
        key = key[1:-1].strip()
    return key


def get_openai_base_url():
    load_project_env()
    base = (os.getenv("OPENAI_BASE_URL") or "").strip().rstrip("/")
    if base:
        return base
    key = get_openai_api_key()
    if key.startswith("sk-or-") or key.startswith("or-"):
        return "https://openrouter.ai/api/v1"
    return None


def llm_endpoint_label() -> str:
    base = get_openai_base_url()
    if base and "openrouter" in base:
        return "OpenRouter"
    if base:
        return base
    return "OpenAI (default)"


def _openrouter_extra_headers() -> dict:
    return {
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost:8501"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "8x8 Class Quality Assessor"),
    }


def create_openai_client():
    """OpenAI SDK client. Do not set default_headers on the client (can break auth on some SDK versions)."""
    from openai import OpenAI

    key = get_openai_api_key()
    if not key:
        raise ValueError("OPENAI_API_KEY / OPENROUTER_API_KEY not set in .env")

    os.environ["OPENAI_API_KEY"] = key
    base = get_openai_base_url()
    if base:
        return OpenAI(api_key=key, base_url=base)
    return OpenAI(api_key=key)


def _chat_extra_headers():
    base = get_openai_base_url() or ""
    if "openrouter" in base:
        return _openrouter_extra_headers()
    return None


def test_llm_connection() -> str:
    load_project_env()
    key = get_openai_api_key()
    base = get_openai_base_url()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if base and "openrouter" in base:
        url = f"{base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            **_openrouter_extra_headers(),
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "max_tokens": 5,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 401:
            raise ValueError(
                "OpenRouter 401: check OPENAI_API_KEY in .env (sk-or-…), "
                "restart Streamlit, and confirm the model name is valid for your account."
            )
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            or "OK"
        )

    client = create_openai_client()
    extra = _chat_extra_headers()
    kwargs = {}
    if extra:
        kwargs["extra_headers"] = extra
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with OK only."}],
        max_tokens=5,
        **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()
