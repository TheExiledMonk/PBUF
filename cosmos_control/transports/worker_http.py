"""HTTP transport for delivering HPC messages from workers to the controller."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from hpc_comms.core.messages import Message, deserialize_message, serialize_message


class WorkerHTTPTransport:
    """HTTP transport for delivering HPC messages from workers."""

    def __init__(
        self,
        controller_endpoint: str,
        *,
        timeout: float = 30.0,
        max_attempts: int = 4,
        backoff_base: float = 0.5,
        max_backoff: float = 8.0,
        auth_token: str | None = None,
    ) -> None:
        base = controller_endpoint.rstrip("/")
        self._url = f"{base}/api/v1/hpc"
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )
        if auth_token:
            self._session.headers["Authorization"] = f"Bearer {auth_token}"
        self._max_attempts = max(1, max_attempts)
        self._backoff_base = backoff_base
        self._max_backoff = max_backoff
        self._last_error: str | None = None
        self._last_retry_count = 0
        self._last_attempt: str | None = None
        self._consecutive_failures = 0

    def send(self, message: Message) -> List[Message]:
        payload = serialize_message(message)
        for attempt in range(1, self._max_attempts + 1):
            self._last_attempt = datetime.utcnow().isoformat()
            try:
                response = self._session.post(
                    self._url, data=payload, timeout=self._timeout
                )
                response.raise_for_status()
                self._last_retry_count = max(0, attempt - 1)
                self._last_error = None
                self._consecutive_failures = 0
                return self._parse_response(response)
            except requests.RequestException as exc:
                self._last_error = str(exc)
                self._last_retry_count = max(0, attempt - 1)
                self._consecutive_failures += 1
                if attempt == self._max_attempts:
                    raise
                delay = min(
                    self._backoff_base * (2 ** (attempt - 1)),
                    self._max_backoff,
                )
                delay = delay * (0.8 + 0.4 * (attempt % 2))
                time.sleep(delay)
        return []

    def transport_status(self) -> Dict[str, Any]:
        return {
            "controller_endpoint": self._url,
            "retry_count": self._last_retry_count,
            "last_error": self._last_error,
            "last_attempt": self._last_attempt,
            "consecutive_failures": self._consecutive_failures,
        }

    def _parse_response(self, response: requests.Response) -> List[Message]:
        if not response.text:
            return []
        body = response.json()
        messages: List[Message] = []
        for entry in body.get("messages", []):
            serialized = entry if isinstance(entry, str) else json.dumps(entry)
            messages.append(deserialize_message(serialized))
        return messages

    def close(self) -> None:
        self._session.close()
