"""
Data models and response types for RadixGate client and engine sidecars.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PrefixRoutingMetadata(BaseModel):
    """
    Metadata extracted from RadixGate L7 response headers.
    Provides visibility into cache hit efficacy and upstream routing.
    """
    worker_id: Optional[str] = Field(
        default=None,
        description="The specific inference replica that served this request (from X-Radix-Worker)"
    )
    matched_blocks: int = Field(
        default=0,
        description="Count of 64-token / 1024-token blocks served warm from KV-cache (from X-Radix-Matched-Blocks)"
    )
    prompt_blocks: int = Field(
        default=0,
        description="Total token blocks in the incoming prompt (from X-Radix-Prompt-Blocks)"
    )
    cache_hit_ratio: float = Field(
        default=0.0,
        description="Instantaneous prefix cache hit ratio for this request (from X-Radix-Hit-Ratio)"
    )

    @property
    def is_cache_hit(self) -> bool:
        """Returns True if at least one prefix block was warm in worker VRAM."""
        return self.matched_blocks > 0


class EvictionSyncPayload(BaseModel):
    """Payload sent by vLLM / SGLang to synchronize physical block evictions."""
    worker_id: str
    evicted_blocks: List[int]


class EvictionSyncResponse(BaseModel):
    """Response returned by RadixGate POST /v1/cache/sync."""
    status: str
    worker_id: str
    pruned_nodes: int
    remaining_worker_blocks: int


class ClusterHealthResponse(BaseModel):
    """Health check status returned by RadixGate GET /health."""
    status: str
    healthy_workers: int
    total_workers: int


class ClusterMetrics(BaseModel):
    """Aggregate telemetry from RadixGate Prometheus / JSON metrics endpoint."""
    requests_total: int = 0
    cache_hit_ratio: float = 0.0
    matched_blocks_total: int = 0
    prompt_blocks_total: int = 0
    worker_loads: Dict[str, int] = Field(default_factory=dict)
