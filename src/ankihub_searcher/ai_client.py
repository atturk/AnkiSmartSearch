"""Zero-dependency HTTP client for LM Studio, OpenRouter, and OpenAI-compatible APIs.

Uses exclusively Python standard library (urllib.request, json, ssl).
Features resilient JSON extraction handling <think>...</think> reasoning blocks
and markdown fences commonly produced by local models.
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

_RE_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_RE_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


class AIClientError(Exception):
    """Base exception for AI client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


_LATEX_COMMANDS_BFNRT = {
    # Greek letters
    "beta",
    "nu",
    "rho",
    "varrho",
    "tau",
    "theta",
    "vartheta",
    # Common math / layout commands
    "frac",
    "times",
    "text",
    "textbf",
    "textit",
    "textnormal",
    "mathrm",
    "mathbf",
    "mathit",
    "nabla",
    "neq",
    "ne",
    "right",
    "rightarrow",
    "Rightarrow",
    "top",
    "to",
    "bar",
    "bm",
    "bold",
    "begin",
    "bullet",
    "forall",
    "notin",
    "neg",
    "not",
    "tilde",
    "tan",
}
_RE_WORD_START = re.compile(r"^[a-zA-Z]+")


def repair_json_escapes(raw_text: str) -> str:
    """Repairs invalid JSON escape sequences by doubling backslashes.

    Doubles backslashes not followed by '"/bfnrtu' or when followed by
    letters forming known LaTeX commands (e.g. \\beta, \\frac, \\nu, \\rho,
    \\tau, \\theta, \\times, etc.) to prevent invalid control characters.
    """
    res: List[str] = []
    i = 0
    n = len(raw_text)
    valid_single = set("\"\\/bfnrt")
    while i < n:
        c = raw_text[i]
        if c == "\\":
            if i + 1 < n:
                nxt = raw_text[i + 1]
                # Check if this escape forms a known LaTeX command (e.g. \beta, \frac)
                m = _RE_WORD_START.match(raw_text[i + 1 :])
                if m and m.group(0).lower() in _LATEX_COMMANDS_BFNRT:
                    res.append("\\\\")
                    i += 1
                    continue
                if nxt in valid_single:
                    res.append(c)
                    res.append(nxt)
                    i += 2
                    continue
                elif nxt == "u":
                    # Valid if followed by exactly 4 hex digits
                    if i + 5 < n and all(ch in "0123456789abcdefABCDEF" for ch in raw_text[i + 2 : i + 6]):
                        res.append(raw_text[i : i + 6])
                        i += 6
                        continue
            # Invalid escape or trailing backslash: double it
            res.append("\\\\")
            i += 1
        else:
            res.append(c)
            i += 1
    return "".join(res)


class OpenAICompatibleClient:
    """Zero-dependency client for OpenAI-compatible REST endpoints."""

    def __init__(self, default_timeout: int = 60):
        self.default_timeout = default_timeout
        # Create permissive SSL context if needed
        self._ssl_ctx = ssl.create_default_context()

    def _normalize_base_url(self, base_url: str) -> str:
        """Removes trailing slashes from base URL."""
        url = base_url.strip().rstrip("/")
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"http://{url}"
        return url

    def _make_request(
        self,
        url: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        api_key: str = "",
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Makes an HTTP request and parses JSON response."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "SmartSearch/1.0",
        }
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        body_bytes = None
        if data is not None:
            body_bytes = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers=headers,
            method=method,
        )

        effective_timeout = timeout or self.default_timeout

        try:
            with urllib.request.urlopen(req, timeout=effective_timeout, context=self._ssl_ctx) as resp:
                resp_text = resp.read().decode("utf-8")
                return json.loads(resp_text)
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise AIClientError(f"HTTP {e.code} ({e.reason}) at {url}: {err_body}", status_code=e.code) from e
        except urllib.error.URLError as e:
            raise AIClientError(f"Connection failed to {url}: {e.reason}") from e
        except TimeoutError as e:
            raise AIClientError(f"Request timed out after {effective_timeout}s to {url}") from e
        except json.JSONDecodeError as e:
            raise AIClientError(f"Invalid JSON returned from {url}: {e}") from e

    def list_models(self, base_url: str, api_key: str = "") -> List[str]:
        """Queries GET /v1/models to fetch available model names."""
        norm_url = self._normalize_base_url(base_url)
        url = f"{norm_url}/models"
        res = self._make_request(url, method="GET", api_key=api_key)
        data = res.get("data", [])
        models: List[str] = []
        for item in data:
            if isinstance(item, dict) and "id" in item:
                models.append(str(item["id"]))
        return sorted(models)

    def test_connection(self, base_url: str, api_key: str = "") -> bool:
        """Returns True if endpoint is reachable and lists models or returns 200."""
        try:
            self.list_models(base_url, api_key)
            return True
        except Exception:
            return False

    @staticmethod
    def extract_clean_json(raw_text: str) -> Any:
        """Resiliently extracts a JSON object or array from LLM text output.

        Handles <think> tags, markdown fences, conversational prelude/postlude,
        and repairs invalid backslash escapes.
        """
        if not raw_text or not raw_text.strip():
            raise AIClientError("LLM returned empty text.")

        def _has_control_chars(obj: Any) -> bool:
            if isinstance(obj, str):
                return "\x08" in obj or "\x0c" in obj
            if isinstance(obj, dict):
                return any(_has_control_chars(k) or _has_control_chars(v) for k, v in obj.items())
            if isinstance(obj, list):
                return any(_has_control_chars(item) for item in obj)
            return False

        def _try_parse(candidate_str: str) -> Optional[Any]:
            try:
                parsed = json.loads(candidate_str)
                if _has_control_chars(parsed):
                    repaired = repair_json_escapes(candidate_str)
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        return parsed
                return parsed
            except json.JSONDecodeError:
                try:
                    repaired = repair_json_escapes(candidate_str)
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    return None

        text = raw_text.strip()

        # 1. Strip reasoning blocks like <think>...</think>
        text = _RE_THINK_BLOCK.sub("", text).strip()

        # 2. Check for markdown code fences (```json ... ``` or ``` ... ```)
        fence_match = _RE_JSON_FENCE.search(text)
        if fence_match:
            candidate = fence_match.group(1).strip()
            parsed = _try_parse(candidate)
            if parsed is not None:
                return parsed
            text = candidate  # Keep trying on the inside of the fence

        # 3. Direct attempt
        parsed = _try_parse(text)
        if parsed is not None:
            return parsed

        # 4. Search for the outermost JSON object {...} or array [...]
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        start_arr = text.find("[")
        end_arr = text.rfind("]")

        candidates = []
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            candidates.append(text[start_obj : end_obj + 1])
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            candidates.append(text[start_arr : end_arr + 1])

        for cand in candidates:
            parsed = _try_parse(cand)
            if parsed is not None:
                return parsed

        snippet = (raw_text[:200] + "...") if len(raw_text) > 200 else raw_text
        raise AIClientError(f"Could not parse valid JSON from LLM output: {snippet}")

    def chat_completion_json(
        self,
        messages: List[Dict[str, str]],
        model: str,
        base_url: str,
        api_key: str = "",
        temperature: float = 0.1,
        timeout: int = 240,
        json_schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
    ) -> Any:
        """Calls POST /v1/chat/completions and returns parsed JSON content."""
        norm_url = self._normalize_base_url(base_url)
        url = f"{norm_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_schema is not None:
            if "type" in json_schema and "json_schema" in json_schema:
                payload["response_format"] = json_schema
            elif "schema" in json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": json_schema,
                }
            else:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": json_schema,
                    },
                }

        if "response_format" in payload:
            try:
                res = self._make_request(url, method="POST", data=payload, api_key=api_key, timeout=timeout)
            except AIClientError as e:
                # If server returns 400 for response_format, retry once without response_format
                if getattr(e, "status_code", None) == 400 or "HTTP 400" in str(e):
                    payload_fallback = dict(payload)
                    payload_fallback.pop("response_format", None)
                    res = self._make_request(
                        url, method="POST", data=payload_fallback, api_key=api_key, timeout=timeout
                    )
                else:
                    raise
        else:
            res = self._make_request(url, method="POST", data=payload, api_key=api_key, timeout=timeout)

        choices = res.get("choices", [])
        if not choices:
            raise AIClientError("No choices returned in chat completion response.")

        first_msg = choices[0].get("message", {})
        raw_content = first_msg.get("content", "") or ""
        if not raw_content.strip():
            # Fallback if a reasoning model placed JSON inside reasoning_content
            raw_content = first_msg.get("reasoning_content", "") or ""
        return self.extract_clean_json(raw_content)

    def get_embeddings(
        self,
        texts: List[str],
        model: str,
        base_url: str,
        api_key: str = "",
        batch_size: int = 32,
        timeout: int = 60,
    ) -> List[List[float]]:
        """Calls POST /v1/embeddings in batches and returns ordered list of float vectors."""
        if not texts:
            return []

        norm_url = self._normalize_base_url(base_url)
        url = f"{norm_url}/embeddings"

        all_vectors: List[List[float]] = []

        # Process in chunks of batch_size
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            payload = {
                "model": model,
                "input": chunk,
            }

            res = self._make_request(url, method="POST", data=payload, api_key=api_key, timeout=timeout)
            data_items = res.get("data", [])
            if not data_items:
                raise AIClientError(f"No embeddings data returned for chunk starting at index {i}.")

            # Sort by index to maintain original order
            sorted_items = sorted(data_items, key=lambda item: item.get("index", 0))
            for item in sorted_items:
                vec = item.get("embedding", [])
                if not vec:
                    raise AIClientError("Empty embedding vector in response item.")
                all_vectors.append(vec)

        return all_vectors
