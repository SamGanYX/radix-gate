"""
vLLM engine hooks and integrations for RadixGate.
"""

import logging
from typing import Any
from .middleware import RadixGateMiddleware

logger = logging.getLogger("radixgate.vllm")


def attach_vllm_app(
    app: Any,
    gateway_url: str = "http://127.0.0.1:8080",
    worker_id: str = "worker-01",
) -> RadixGateMiddleware:
    """
    Attaches RadixGateMiddleware to a vLLM FastAPI / Starlette application.
    Usage:
        from vllm.entrypoints.openai.api_server import app
        from radixgate.vllm import attach_vllm_app

        middleware = attach_vllm_app(app, gateway_url="http://radix-gate:8080", worker_id="worker-01")
    """
    middleware = RadixGateMiddleware(
        app=app.app if hasattr(app, "app") else app,
        gateway_url=gateway_url,
        worker_id=worker_id,
    )
    if hasattr(app, "add_middleware"):
        app.add_middleware(
            RadixGateMiddleware,
            gateway_url=gateway_url,
            worker_id=worker_id,
        )
    logger.info("Attached RadixGate middleware to vLLM app for worker '%s'", worker_id)
    return middleware


def hook_block_space_manager(
    block_manager: Any,
    middleware: RadixGateMiddleware,
) -> None:
    """
    Monkey-patches or attaches to vLLM's BlockSpaceManager to capture block frees.
    """
    if not hasattr(block_manager, "free"):
        logger.warning("Target block_manager does not expose a 'free' method.")
        return

    orig_free = block_manager.free

    def wrapped_free(block: Any, *args, **kwargs):
        # Extract block hash or block id
        block_id = getattr(block, "block_number", None) or getattr(block, "block_id", None)
        if block_id is not None:
            # Map block ID into a hash representation
            middleware.record_eviction(hash(block_id))
        return orig_free(block, *args, **kwargs)

    block_manager.free = wrapped_free
    logger.info("Successfully hooked vLLM BlockSpaceManager.free for eviction sync.")
