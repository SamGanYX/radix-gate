"""
Data models and response types for RadixGate client and engine sidecars.
Schemas match the exact JSON payloads and headers emitted by the Rust gateway.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PrefixRoutingMetadata(BaseModel):
    """
    Metadata extracted from RadixGate L7 response headers.
    Provides visibility into cache hit efficacy and upstream routing.
    """

    worker_id: Optional[str] = Field(
        default=None,
        description="The specific inference replica that served this request (from X-Radix-Worker)",
    )
    matched_blocks: int = Field(
        default=0,
        description="Count of token blocks served warm from KV-cache (from X-Radix-Matched-Blocks)",
    )
    total_blocks: int = Field(
        default=0,
        description="Total token blocks in the incoming prompt (from X-Radix-Total-Blocks)",
    )
    cache_hit_ratio: float = Field(
        default=0.0,
        description="Instantaneous prefix cache hit ratio for this request (from X-Radix-Hit-Ratio)",
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="W3C distributed trace ID assigned to this request (from X-Radix-Trace-Id)",
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
    evicted_blocks_purged: int


class ClusterHealthResponse(BaseModel):
    """Health check status returned by RadixGate GET /health."""

    status: str
    service: str = "radix-gate"


class RadixTreeStats(BaseModel):
    """Summary of the concurrent Radix Tree state."""

    total_nodes: int = 0
    unique_workers: int = 0


class PerformanceMetrics(BaseModel):
    """Aggregate cluster cache performance metrics."""

    total_requests: int = 0
    total_matched_blocks: int = 0
    total_prompt_blocks: int = 0
    cluster_cache_hit_ratio: str = "0.00%"

    @property
    def hit_ratio_float(self) -> float:
        """Parses cluster_cache_hit_ratio string (e.g. '86.80%') into a 0.0-1.0 float."""
        cleaned = self.cluster_cache_hit_ratio.strip().rstrip("%")
        try:
            return float(cleaned) / 100.0
        except ValueError:
            return 0.0


class WorkerMetricSummary(BaseModel):
    """Per-worker telemetry returned in GET /metrics JSON."""

    id: str
    endpoint: str
    load: int = 0
    role: str = "Unified"
    role_badge: str = "[U]"
    cache_utilization: str = "0.0%"
    total_routed: int = 0
    matched_blocks: int = 0
    prefilled_blocks: int = 0


class ClusterMetrics(BaseModel):
    """Aggregate telemetry from RadixGate GET /metrics (Accept: application/json)."""

    status: str = "healthy"
    radix_tree: RadixTreeStats = Field(default_factory=RadixTreeStats)
    performance: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    workers: List[WorkerMetricSummary] = Field(default_factory=list)
