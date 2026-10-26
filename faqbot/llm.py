"""Client for any LLM server that speaks the OpenAI-style chat API.

Works unchanged with Ollama (laptop), vLLM (HPC) or a hosted provider: only the
base_url and model name in config.yaml differ.
"""

from __future__ import annotations

import json
import re
from typing import Iterator

import requests

from .config import LLMSettings

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    """The LLM server could not be reached or returned an error."""


class LLMClient:
    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.base_url = settings.base_url.rstrip("/")
        self.model = settings.model

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"}

    def chat(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "stream": False,
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.settings.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach the LLM server at {self.base_url} ({exc.__class__.__name__})") from exc
        if response.status_code != 200:
            raise LLMError(f"LLM server returned {response.status_code}: {response.text[:300]}")
        try:
            content = response.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected reply from LLM server: {response.text[:300]}") from exc
        # Some newer models print their reasoning inside <think> tags; never show that to users.
        return _THINK_BLOCK.sub("", content).strip()

    def chat_stream(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> Iterator[str]:
        """Yield the reply piece by piece as the server writes it (OpenAI-style server-sent events)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "stream": True,
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions", json=payload, headers=self._headers(),
                timeout=self.settings.timeout_seconds, stream=True,
            )
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach the LLM server at {self.base_url} ({exc.__class__.__name__})") from exc
        if response.status_code != 200:
            body = response.text[:300]
            response.close()
            raise LLMError(f"LLM server returned {response.status_code}: {body}")

        # Hide a leading <think>...</think> block (some newer models print their reasoning).
        pending, started = "", False
        try:
            for raw in response.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    choices = json.loads(data).get("choices") or []
                except ValueError:
                    continue
                piece = ((choices[0].get("delta") or {}).get("content") or "") if choices else ""
                if not piece:
                    continue
                if started:
                    yield piece
                    continue
                pending += piece
                head = pending.lstrip()
                if head.lower().startswith("<think>"):
                    end = head.lower().find("</think>")
                    if end == -1:
                        continue
                    pending, started = head[end + len("</think>"):].lstrip(), True
                elif head and "<think>".startswith(head.lower()):
                    continue  # could still be the start of a <think> tag
                else:
                    started = True
                if started and pending:
                    yield pending
                    pending = ""
            if not started and pending and not pending.lstrip().lower().startswith("<think>"):
                yield pending
        except requests.RequestException as exc:
            raise LLMError(f"Lost connection to the LLM server ({exc.__class__.__name__})") from exc
        finally:
            response.close()

    def status(self) -> tuple[bool, str]:
        """(ready?, human-readable message). Fast check used by the UI and setup script."""
        try:
            response = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=3)
        except requests.RequestException:
            return False, f"No LLM server at {self.base_url}. Is Ollama running?"
        if response.status_code != 200:
            return False, f"LLM server answered {response.status_code}"
        try:
            ids = [m.get("id", "") for m in response.json().get("data", [])]
        except ValueError:
            return False, "LLM server sent an unreadable model list"
        if ids and self.model not in ids and f"{self.model}:latest" not in ids:
            return False, f"Server is running but model '{self.model}' is missing. Run: ollama pull {self.model}"
        return True, f"{self.model} ready"
