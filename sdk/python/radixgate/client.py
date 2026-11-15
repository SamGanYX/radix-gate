"""
Synchronous and Asynchronous client interfaces for RadixGate.
"""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple
import httpx

from .models import (
    ClusterHealthResponse,
    ClusterMetrics,
    EvictionSyncPayload,
    EvictionSyncResponse,
    PrefixRoutingMetadata,
)


def _extract_routing_metadata(headers: httpx.Headers) -> PrefixRoutingMetadata:
    """Extracts custom X-Radix-* headers into a structured PrefixRoutingMetadata instance."""
    worker = headers.get("x-radix-worker")
    matched = int(headers.get("x-radix-matched-blocks", "0"))
    prompt = int(headers.get("x-radix-prompt-blocks", "0"))
    hit_ratio = float(headers.get("x-radix-hit-ratio", "0.0"))

    return PrefixRoutingMetadata(
        worker_id=worker,
        matched_blocks=matched,
        prompt_blocks=prompt,
        cache_hit_ratio=hit_ratio,
    )


class RadixGateClient:
    """
    Synchronous HTTP client for interacting with a RadixGate cluster.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float = 30.0,
        client: Optional[httpx.Client] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def health(self) -> ClusterHealthResponse:
        """Checks cluster liveness and counts healthy worker replicas."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        data = resp.json()
        return ClusterHealthResponse(**data)

    def metrics(self) -> ClusterMetrics:
        """Retrieves real-time telemetry metrics in JSON format."""
        resp = self._client.get("/metrics", headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        return ClusterMetrics(**data)

    def tree(self) -> str:
        """Fetches the live ASCII representation of the active Radix Tree."""
        resp = self._client.get("/api/v1/tree")
        resp.raise_for_status()
        return resp.text

    def sync_evicted_blocks(
        self,
        worker_id: str,
        evicted_blocks: List[int],
    ) -> EvictionSyncResponse:
        """Directly informs the gateway of evicted VRAM block hashes to prevent stale routing."""
        payload = EvictionSyncPayload(worker_id=worker_id, evicted_blocks=evicted_blocks)
        resp = self._client.post("/v1/cache/sync", json=payload.model_dump())
        resp.raise_for_status()
        return EvictionSyncResponse(**resp.json())

    def trigger_reload(self) -> Dict[str, Any]:
        """Triggers dynamic configuration hot-reload via POST /-/reload."""
        resp = self._client.post("/-/reload")
        resp.raise_for_status()
        return resp.json()

    def chat_completion(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], PrefixRoutingMetadata]:
        """Dispatches an OpenAI-compatible /v1/chat/completions request."""
        resp = self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        metadata = _extract_routing_metadata(resp.headers)
        return resp.json(), metadata

    def messages(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], PrefixRoutingMetadata]:
        """Dispatches an Anthropic-compatible /v1/messages request."""
        resp = self._client.post("/v1/messages", json=payload)
        resp.raise_for_status()
        metadata = _extract_routing_metadata(resp.headers)
        return resp.json(), metadata


class AsyncRadixGateClient:
    """
    Asynchronous HTTP client for interacting with a RadixGate cluster.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float = 30.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def aclose(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    async def health(self) -> ClusterHealthResponse:
        """Checks cluster liveness and counts healthy worker replicas."""
        resp = await self._client.get("/health")
        resp.raise_for_status()
        data = resp.json()
        return ClusterHealthResponse(**data)

    async def metrics(self) -> ClusterMetrics:
        """Retrieves real-time telemetry metrics in JSON format."""
        resp = await self._client.get("/metrics", headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        return ClusterMetrics(**data)

    async def tree(self) -> str:
        """Fetches the live ASCII representation of the active Radix Tree."""
        resp = await self._client.get("/api/v1/tree")
        resp.raise_for_status()
        return resp.text

    async def sync_evicted_blocks(
        self,
        worker_id: str,
        evicted_blocks: List[int],
    ) -> EvictionSyncResponse:
        """Directly informs the gateway of evicted VRAM block hashes to prevent stale routing."""
        payload = EvictionSyncPayload(worker_id=worker_id, evicted_blocks=evicted_blocks)
        resp = await self._client.post("/v1/cache/sync", json=payload.model_dump())
        resp.raise_for_status()
        return EvictionSyncResponse(**resp.json())

    async def trigger_reload(self) -> Dict[str, Any]:
        """Triggers dynamic configuration hot-reload via POST /-/reload."""
        resp = await self._client.post("/-/reload")
        resp.raise_for_status()
        return resp.json()

    async def chat_completion(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], PrefixRoutingMetadata]:
        """Dispatches an OpenAI-compatible /v1/chat/completions request."""
        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        metadata = _extract_routing_metadata(resp.headers)
        return resp.json(), metadata

    async def messages(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], PrefixRoutingMetadata]:
        """Dispatches an Anthropic-compatible /v1/messages request."""
        resp = await self._client.post("/v1/messages", json=payload)
        resp.raise_for_status()
        metadata = _extract_routing_metadata(resp.headers)
        return resp.json(), metadata
