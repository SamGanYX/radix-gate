"""
Unit tests for the RadixGate Python SDK and middleware.
"""

import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import httpx

from radixgate.models import (
    PrefixRoutingMetadata,
    EvictionSyncPayload,
    EvictionSyncResponse,
    ClusterHealthResponse,
    ClusterMetrics,
    PerformanceMetrics,
    RadixTreeStats,
    WorkerMetricSummary,
)
from radixgate.client import (
    RadixGateClient,
    AsyncRadixGateClient,
    _extract_routing_metadata,
)
from radixgate.middleware import RadixGateMiddleware
from radixgate.vllm import hook_block_space_manager
from radixgate.sglang import hook_radix_cache


class TestRadixGateModels(unittest.TestCase):
    def test_routing_metadata(self):
        meta = PrefixRoutingMetadata(
            worker_id="worker-01",
            matched_blocks=4,
            total_blocks=10,
            cache_hit_ratio=0.40,
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        )
        self.assertTrue(meta.is_cache_hit)
        self.assertEqual(meta.worker_id, "worker-01")
        self.assertEqual(meta.matched_blocks, 4)
        self.assertEqual(meta.total_blocks, 10)
        self.assertAlmostEqual(meta.cache_hit_ratio, 0.40)

        cold_meta = PrefixRoutingMetadata(matched_blocks=0)
        self.assertFalse(cold_meta.is_cache_hit)

    def test_eviction_payload(self):
        payload = EvictionSyncPayload(
            worker_id="gpu-node-03",
            evicted_blocks=[12345678, 87654321],
        )
        self.assertEqual(len(payload.evicted_blocks), 2)
        d = payload.model_dump()
        self.assertEqual(d["worker_id"], "gpu-node-03")

    def test_cluster_metrics(self):
        metrics = ClusterMetrics(
            status="healthy",
            radix_tree=RadixTreeStats(total_nodes=12, unique_workers=2),
            performance=PerformanceMetrics(
                total_requests=100,
                total_matched_blocks=850,
                total_prompt_blocks=1000,
                cluster_cache_hit_ratio="85.00%",
            ),
            workers=[
                WorkerMetricSummary(
                    id="w1",
                    endpoint="http://127.0.0.1:8001",
                    load=2,
                )
            ],
        )
        self.assertEqual(metrics.performance.total_requests, 100)
        self.assertAlmostEqual(metrics.performance.hit_ratio_float, 0.85)
        self.assertEqual(metrics.workers[0].load, 2)


class TestRadixGateClient(unittest.TestCase):
    def test_extract_routing_metadata(self):
        headers = httpx.Headers({
            "x-radix-worker": "worker-42",
            "x-radix-matched-blocks": "8",
            "x-radix-total-blocks": "16",
            "x-radix-hit-ratio": "0.50",
            "x-radix-trace-id": "4bf92f3577b34da6a3ce929d0e0e4736",
        })
        meta = _extract_routing_metadata(headers)
        self.assertEqual(meta.worker_id, "worker-42")
        self.assertEqual(meta.matched_blocks, 8)
        self.assertEqual(meta.total_blocks, 16)
        self.assertAlmostEqual(meta.cache_hit_ratio, 0.50)
        self.assertEqual(meta.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736")

    def test_sync_client_endpoints(self):
        mock_client = MagicMock(spec=httpx.Client)

        # Mock /health (exact Rust server JSON)
        mock_resp_health = MagicMock()
        mock_resp_health.json.return_value = {
            "status": "ok",
            "service": "radix-gate",
        }
        mock_client.get.return_value = mock_resp_health

        client = RadixGateClient(client=mock_client)
        health = client.health()
        self.assertEqual(health.status, "ok")
        self.assertEqual(health.service, "radix-gate")

        # Mock /metrics (exact Rust server JSON)
        mock_resp_metrics = MagicMock()
        mock_resp_metrics.json.return_value = {
            "status": "healthy",
            "radix_tree": {"total_nodes": 5, "unique_workers": 1},
            "performance": {
                "total_requests": 50,
                "total_matched_blocks": 400,
                "total_prompt_blocks": 500,
                "cluster_cache_hit_ratio": "80.00%",
            },
            "workers": [
                {
                    "id": "w1",
                    "endpoint": "http://127.0.0.1:8001",
                    "load": 1,
                    "role": "Unified",
                    "role_badge": "[U]",
                    "cache_utilization": "12.5%",
                    "total_routed": 50,
                    "matched_blocks": 400,
                    "prefilled_blocks": 100,
                }
            ],
        }
        mock_client.get.return_value = mock_resp_metrics
        metrics = client.metrics()
        self.assertEqual(metrics.performance.total_requests, 50)
        self.assertAlmostEqual(metrics.performance.hit_ratio_float, 0.80)

        # Mock /v1/chat/completions
        mock_resp_chat = MagicMock()
        mock_resp_chat.headers = httpx.Headers({
            "x-radix-worker": "worker-01",
            "x-radix-matched-blocks": "12",
            "x-radix-total-blocks": "12",
            "x-radix-hit-ratio": "1.00",
        })
        mock_resp_chat.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_client.post.return_value = mock_resp_chat

        body, meta = client.chat_completion({"model": "gpt-4o", "messages": []})
        self.assertEqual(body["choices"][0]["message"]["content"], "ok")
        self.assertEqual(meta.worker_id, "worker-01")
        self.assertTrue(meta.is_cache_hit)


class TestRadixGateMiddleware(unittest.IsolatedAsyncioTestCase):
    async def test_middleware_asgi_wrapping_and_header_injection(self):
        # Mock target ASGI app
        async def dummy_app(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"status": "ok"}',
            })

        middleware = RadixGateMiddleware(
            app=dummy_app,
            gateway_url="http://127.0.0.1:8080",
            worker_id="test-worker",
            async_sync=False,
        )

        # Record evictions
        middleware.record_evictions([1111, 2222])

        sent_messages = []

        async def fake_send(msg):
            sent_messages.append(msg)

        async def fake_receive():
            return {"type": "http.request"}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
        }

        await middleware(scope, fake_receive, fake_send)

        self.assertEqual(len(sent_messages), 2)
        start_msg = sent_messages[0]
        self.assertEqual(start_msg["type"], "http.response.start")

        headers_dict = dict(start_msg["headers"])
        self.assertIn(b"x-evicted-blocks", headers_dict)
        evicted_header_str = headers_dict[b"x-evicted-blocks"].decode("utf-8")
        self.assertTrue("1111" in evicted_header_str and "2222" in evicted_header_str)

    def test_vllm_and_sglang_hooks(self):
        middleware = RadixGateMiddleware(app=None, worker_id="test")

        # Test vLLM hook
        class MockBlock:
            def __init__(self, block_number):
                self.block_number = block_number

        class MockBlockSpaceManager:
            def __init__(self):
                self.freed = []

            def free(self, block):
                self.freed.append(block)

        bm = MockBlockSpaceManager()
        hook_block_space_manager(bm, middleware)

        blk = MockBlock(9999)
        bm.free(blk)
        self.assertEqual(len(bm.freed), 1)
        self.assertTrue(len(middleware._pending_evictions) > 0)

        # Test SGLang hook
        class MockRadixCache:
            def evict(self, num_tokens, callback=None):
                class MockNode:
                    hash = 55555
                if callback:
                    callback(MockNode())
                return num_tokens

        rc = MockRadixCache()
        hook_radix_cache(rc, middleware)
        rc.evict(10)
        self.assertIn(55555, middleware._pending_evictions)


if __name__ == "__main__":
    unittest.main()
