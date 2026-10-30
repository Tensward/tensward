"""When a llama-server is ready for requests."""

from __future__ import annotations

import httpx

from ..protocol import Failed, ModelListProbe, NotYet, Ready

CHAT_PATH = "/v1/chat/completions"


class LlamaCppProbe(ModelListProbe):
    """``/health`` 200, the alias in ``/v1/models``, then one one-token chat request: llama-server
    answers ``/health`` before its first request can be served."""

    timeout_s = 60.0

    def __init__(self, health_path: str, models_path: str) -> None:
        super().__init__(health_path, models_path)
        self.answered = False

    async def check(
        self, client: httpx.AsyncClient, url: str, served_name: str
    ) -> Ready | NotYet | Failed:
        listed = await super().check(client, url, served_name)
        if not isinstance(listed, Ready) or self.answered:
            return listed
        body = {
            "model": served_name,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
        try:
            response = await client.post(f"{url}{CHAT_PATH}", json=body)
        except httpx.HTTPError:
            return NotYet()
        if response.status_code == 200:
            self.answered = True
            return listed
        if response.status_code == 503:
            return NotYet()
        said = response.text.strip()[:200]
        return Failed(
            f"the server answered a one-token request with {response.status_code}: {said}"
        )
