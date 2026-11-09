use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

/// Type alias for 64-bit token block hash (e.g. 64 or 128 token window).
pub type BlockHash = u64;

/// Immutable shared string identifier for worker replicas to eliminate heap allocations in traversal.
pub type WorkerId = Arc<str>;

/// A node in the Radix Tree representing a sequence of token block hashes.
#[derive(Debug, Clone)]
pub struct RadixNode {
    /// Compressed prefix of block hashes for this edge.
    pub prefix: Vec<BlockHash>,
    /// Set of worker replicas that have this block sequence hot in their KV cache.
    pub cached_workers: HashSet<WorkerId>,
    /// Active in-flight prefill leases (WorkerId -> active concurrent prefill count).
    /// Used for request coalescing to prevent thundering herd cold prefills.
    pub in_flight_workers: HashMap<WorkerId, usize>,
    /// Child branches keyed by the first block hash of the subsequent branch.
    pub children: HashMap<BlockHash, RadixNode>,
    /// Last access timestamp for LRU shadow eviction.
    pub last_accessed: Instant,
    /// Total number of requests that hit this node.
    pub hit_count: u64,
}

impl RadixNode {
    pub fn new(prefix: Vec<BlockHash>, worker_id: Option<WorkerId>) -> Self {
        let mut cached_workers = HashSet::new();
        if let Some(w) = worker_id {
            cached_workers.insert(w);
        }
        Self {
            prefix,
            cached_workers,
            in_flight_workers: HashMap::new(),
            children: HashMap::new(),
            last_accessed: Instant::now(),
            hit_count: 1,
        }
    }
}
