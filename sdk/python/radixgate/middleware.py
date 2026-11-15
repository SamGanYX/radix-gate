"""
ASGI Middleware for vLLM and SGLang engine servers.
Captures physical VRAM block evictions and synchronizes them to RadixGate.
"""

import asyncio
import logging
from typing import Any, Callable, List, Optional, Set
import httpx

logger = logging.getLogger("radixgate.middleware")


class RadixGateMiddleware:
    """
    ASGI middleware wrapping vLLM or SGLang API servers.

    Features:
    1. Collects evicted block hashes recorded by engine memory managers.
    2. Attaches `x-evicted-blocks` trailer/headers to outgoing HTTP responses.
    3. Flushes evicted block batches to RadixGate's `POST /v1/cache/sync` endpoint
       either synchronously on response or asynchronously via a background task.
    """

    def __init__(
        self,
        app: Any,
        gateway_url: str = "http://127.0.0.1:8080",
        worker_id: str = "worker-01",
        flush_interval: float = 0.5,
        max_batch_size: int = 128,
        async_sync: bool = True,
    ):
        self.app = app
        self.gateway_url = gateway_url.rstrip("/")
        self.worker_id = worker_id
        self.flush_interval = flush_interval
        self.max_batch_size = max_batch_size
        self.async_sync = async_sync

        self._pending_evictions: Set[int] = set()
        self._lock = asyncio.Lock()
        self._http_client: Optional[httpx.AsyncClient] = None
        self._flush_task: Optional[asyncio.Task] = None

    def record_eviction(self, block_hash: int) -> None:
        """Called by engine event hooks when physical VRAM blocks are evicted."""
        self._pending_evictions.add(block_hash)

    def record_evictions(self, block_hashes: List[int]) -> None:
        """Called by engine event hooks when a collection of blocks is evicted."""
        self._pending_evictions.update(block_hashes)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self.gateway_url,
                timeout=2.0,
            )
        return self._http_client

    async def flush_evictions(self) -> int:
        """Flushes currently pending evicted block hashes to the RadixGate gateway."""
        if not self._pending_evictions:
            return 0

        async with self._lock:
            if not self._pending_evictions:
                return 0
            evicted_snapshot = list(self._pending_evictions)[: self.max_batch_size]
            for b in evicted_snapshot:
                self._pending_evictions.discard(b)

        try:
            client = await self._get_client()
            resp = await client.post(
                "/v1/cache/sync",
                json={
                    "worker_id": self.worker_id,
                    "evicted_blocks": evicted_snapshot,
                },
            )
            if resp.status_code == 200:
                logger.debug(
                    "Flushed %d evicted blocks for worker %s to RadixGate",
                    len(evicted_snapshot),
                    self.worker_id,
                )
                return len(evicted_snapshot)
            else:
                logger.warning(
                    "Gateway returned status %d on cache sync: %s",
                    resp.status_code,
                    resp.text,
                )
        except Exception as e:
            logger.error("Failed to transmit cache sync to gateway: %s", e)

        return 0

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Snapshot any pending evictions to attach to current response headers
        evicted_snapshot: List[int] = []
        if self._pending_evictions:
            async with self._lock:
                evicted_snapshot = list(self._pending_evictions)
                self._pending_evictions.clear()

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start" and evicted_snapshot:
                headers = list(message.get("headers", []))
                evicted_csv = ",".join(str(b) for b in evicted_snapshot)
                headers.append((b"x-evicted-blocks", evicted_csv.encode("utf-8")))
                message["headers"] = headers

                if self.async_sync:
                    # Fire-and-forget sync to gateway
                    asyncio.create_task(self._sync_direct(evicted_snapshot))

            await send(message)

        await self.app(scope, receive, send_wrapper)

    async def _sync_direct(self, blocks: List[int]) -> None:
        try:
            client = await self._get_client()
            await client.post(
                "/v1/cache/sync",
                json={
                    "worker_id": self.worker_id,
                    "evicted_blocks": blocks,
                },
            )
        except Exception as e:
            logger.error("Error in direct eviction sync: %s", e)

    async def close(self) -> None:
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
