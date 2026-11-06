use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Type alias for 64-bit token block hash (e.g. 64 or 128 token window).
pub type BlockHash = u64;

/// Unique identifier for an inference worker replica.
pub type WorkerId = String;

/// A node in the Radix Tree representing a sequence of token block hashes.
#[derive(Debug, Clone)]
pub struct RadixNode {
    /// Compressed prefix of block hashes for this edge.
    pub prefix: Vec<BlockHash>,
    /// Set of worker replicas that have this block sequence hot in their KV cache.
    pub cached_workers: HashSet<WorkerId>,
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
            children: HashMap::new(),
            last_accessed: Instant::now(),
            hit_count: 1,
        }
    }
}
