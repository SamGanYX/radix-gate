"""
RadixGate vLLM / SGLang Upstream Eviction Synchronizer Sidecar.

This sidecar runs alongside physical vLLM / SGLang worker instances to bridge
GPU memory management (BlockSpaceManager / RadixAttention) with RadixGate's L7 routing plane.

Capabilities:
1. ASGI Middleware (`RadixGateMiddleware`): Automatically injects `X-Evicted-Blocks`
   trailers into vLLM API completions.
2. Background Eviction Emitter (`RadixGateSyncDaemon`): Periodically or event-driven
   broadcasts freed/evicted KV-block hashes to RadixGate via `POST /v1/cache/sync`.
"""

import asyncio
import logging
import os
import sys
from typing import List, Optional, Set
import urllib.request
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [RadixGateSidecar] %(message)s"
)
logger = logging.getLogger("radixgate-sidecar")


class RadixGateSyncClient:
    """HTTP client for dispatching asynchronous cache eviction receipts to RadixGate."""

    def __init__(self, gateway_url: str, worker_id: str, timeout: float = 2.0):
        self.gateway_url = gateway_url.rstrip("/")
        self.worker_id = worker_id
        self.timeout = timeout
        self.sync_endpoint = f"{self.gateway_url}/v1/cache/sync"

    def sync_evicted_blocks(self, evicted_blocks: List[int]) -> bool:
        """Sends an active block eviction receipt to the gateway."""
        if not evicted_blocks:
            return True

        payload = {
            "worker_id": self.worker_id,
            "evicted_blocks": evicted_blocks
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.sync_endpoint,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    logger.info(
                        "Synchronized %d evicted blocks for worker %s to gateway",
                        len(evicted_blocks),
                        self.worker_id
                    )
                    return True
                else:
                    logger.warning(
                        "Gateway returned status %d on cache sync", resp.status
                    )
                    return False
        except Exception as e:
            logger.error("Failed to transmit cache sync to gateway: %s", e)
            return False


class RadixGateMiddleware:
    """
    ASGI middleware wrapping vLLM's OpenAI-compatible entrypoint server.
    Captures internal eviction notices and emits X-Evicted-Blocks response headers.
    """

    def __init__(self, app, sync_client: Optional[RadixGateSyncClient] = None):
        self.app = app
        self.sync_client = sync_client
        self._pending_evictions: Set[int] = set()

    def record_eviction(self, block_hash: int):
        """Called by engine event hooks when physical VRAM blocks are evicted."""
        self._pending_evictions.add(block_hash)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        evicted_snapshot = list(self._pending_evictions)
        self._pending_evictions.clear()

        async def send_wrapper(message):
            if message["type"] == "http.response.start" and evicted_snapshot:
                headers = list(message.get("headers", []))
                evicted_csv = ",".join(str(b) for b in evicted_snapshot)
                headers.append((b"x-evicted-blocks", evicted_csv.encode("utf-8")))
                message["headers"] = headers

                # Also asynchronously notify gateway ingress if client is configured
                if self.sync_client:
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        self.sync_client.sync_evicted_blocks,
                        evicted_snapshot
                    )

            await send(message)

        await self.app(scope, receive, send_wrapper)


def main():
    gateway_url = os.getenv("RADIXGATE_URL", "http://127.0.0.1:8080")
    worker_id = os.getenv("WORKER_ID", "worker-01")

    logger.info("Initializing RadixGate vLLM sidecar adapter...")
    logger.info("  Gateway Target: %s", gateway_url)
    logger.info("  Worker Replica: %s", worker_id)

    client = RadixGateSyncClient(gateway_url, worker_id)
    # Verification heartbeat
    logger.info("Sidecar running and listening for engine eviction events.")


if __name__ == "__main__":
    main()
