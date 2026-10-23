import json
import os

import requests

from class_assessor.env_loader import (
    _openrouter_extra_headers,
    create_openai_client,
    get_openai_api_key,
    get_openai_base_url,
    load_project_env,
)


def chat_completion_json(system: str, user_content: str) -> dict:
    """Chat completion; OpenRouter uses explicit Bearer auth (avoids SDK 401 issues)."""
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
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code == 401:
            raise ValueError(
                "OpenRouter rejected the API key (401). Verify sk-or-… in .env and restart Streamlit."
            )
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"]
        return json.loads(raw)

    client = create_openai_client()
    from class_assessor.env_loader import _chat_extra_headers

    extra = _chat_extra_headers()
    kwargs = {"extra_headers": extra} if extra else {}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        **kwargs,
    )
    return json.loads(resp.choices[0].message.content or "{}")
