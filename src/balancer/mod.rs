pub mod policies;

use crate::shadow_cache::WorkerPool;
use crate::trie::{BlockHash, RadixTree};

/// Result of a routing decision made by the balancer.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub worker_id: String,
    pub endpoint_url: String,
    pub matched_blocks: usize,
    pub total_blocks: usize,
    pub affinity_score: f64,
    pub policy_name: &'static str,
    /// When high prefix velocity / hotspot is detected on a congested worker,
    /// suggests a secondary replica to proactively replicate the prefix to.
    pub replicate_to_worker: Option<String>,
}

/// Core trait that all cluster load balancing policies implement.
pub trait RoutingPolicy: Send + Sync {
    fn select_worker(
        &self,
        pool: &WorkerPool,
        radix_tree: &RadixTree,
        prompt_blocks: &[BlockHash],
    ) -> Option<RoutingDecision>;
}
