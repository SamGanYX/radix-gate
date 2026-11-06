use crate::trie::BlockHash;
use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};

/// State of an individual inference worker replica.
#[derive(Debug)]
pub struct WorkerReplica {
    pub id: String,
    pub endpoint_url: String,
    pub max_kv_blocks: usize,
    pub max_concurrency: usize,
    pub active_requests: AtomicUsize,
    pub is_healthy: bool,

    // Shadow LRU cache state
    cached_blocks_set: HashSet<BlockHash>,
    lru_order: VecDeque<BlockHash>,

    // Performance metrics
    pub total_requests_routed: u64,
    pub total_matched_blocks: u64,
    pub total_prefilled_blocks: u64,
}

impl WorkerReplica {
    pub fn new(
        id: String,
        endpoint_url: String,
        max_kv_blocks: usize,
        max_concurrency: usize,
    ) -> Self {
        Self {
            id,
            endpoint_url,
            max_kv_blocks,
            max_concurrency,
            active_requests: AtomicUsize::new(0),
            is_healthy: true,
            cached_blocks_set: HashSet::new(),
            lru_order: VecDeque::new(),
            total_requests_routed: 0,
            total_matched_blocks: 0,
            total_prefilled_blocks: 0,
        }
    }

    /// Current in-flight requests.
    pub fn load(&self) -> usize {
        self.active_requests.load(Ordering::Relaxed)
    }

    /// Load factor in [0.0, 1.0].
    pub fn load_factor(&self) -> f64 {
        let current = self.load() as f64;
        let max = self.max_concurrency.max(1) as f64;
        (current / max).min(1.0)
    }

    /// Registers that a request was processed by this worker.
    /// Updates the shadow LRU cache with any new blocks and returns evicted blocks (if any).
    pub fn record_request(
        &mut self,
        blocks: &[BlockHash],
        matched_blocks: usize,
    ) -> Vec<BlockHash> {
        self.total_requests_routed += 1;
        self.total_matched_blocks += matched_blocks as u64;
        let new_blocks_count = blocks.len().saturating_sub(matched_blocks);
        self.total_prefilled_blocks += new_blocks_count as u64;

        let mut evicted = Vec::new();

        for &b in blocks {
            if self.cached_blocks_set.contains(&b) {
                // Refresh position in LRU
                if let Some(pos) = self.lru_order.iter().position(|&x| x == b) {
                    self.lru_order.remove(pos);
                }
                self.lru_order.push_back(b);
            } else {
                // Need to insert new block. Check capacity
                while self.lru_order.len() >= self.max_kv_blocks && !self.lru_order.is_empty() {
                    if let Some(oldest) = self.lru_order.pop_front() {
                        self.cached_blocks_set.remove(&oldest);
                        evicted.push(oldest);
                    }
                }
                self.cached_blocks_set.insert(b);
                self.lru_order.push_back(b);
            }
        }

        evicted
    }

    /// Current KV cache block utilization in [0.0, 1.0].
    pub fn cache_utilization(&self) -> f64 {
        if self.max_kv_blocks == 0 {
            return 0.0;
        }
        self.lru_order.len() as f64 / self.max_kv_blocks as f64
    }

    /// Check if a block is currently in the worker's shadow cache.
    pub fn has_block(&self, block: BlockHash) -> bool {
        self.cached_blocks_set.contains(&block)
    }
}
