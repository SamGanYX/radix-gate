"""
SGLang engine hooks and integrations for RadixGate.
"""

import logging
from typing import Any
from .middleware import RadixGateMiddleware

logger = logging.getLogger("radixgate.sglang")


def attach_sglang_app(
    app: Any,
    gateway_url: str = "http://127.0.0.1:8080",
    worker_id: str = "worker-01",
) -> RadixGateMiddleware:
    """
    Attaches RadixGateMiddleware to an SGLang FastAPI server instance.
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
    logger.info("Attached RadixGate middleware to SGLang app for worker '%s'", worker_id)
    return middleware


def hook_radix_cache(
    radix_cache: Any,
    middleware: RadixGateMiddleware,
) -> None:
    """
    Hooks into SGLang's internal RadixCache tree eviction routines.
    """
    if not hasattr(radix_cache, "evict"):
        logger.warning("Target radix_cache does not expose an 'evict' method.")
        return

    orig_evict = radix_cache.evict

    def wrapped_evict(num_tokens: int, evict_callback: Any = None):
        def custom_callback(node: Any):
            prefix_hash = getattr(node, "hash", None) or hash(str(getattr(node, "key", "")))
            if prefix_hash is not None:
                middleware.record_eviction(prefix_hash)
            if evict_callback is not None:
                evict_callback(node)

        return orig_evict(num_tokens, custom_callback)

    radix_cache.evict = wrapped_evict
    logger.info("Successfully hooked SGLang RadixCache.evict for eviction sync.")
