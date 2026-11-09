use super::{BlockHash, MatchResult, RadixTree};
use parking_lot::RwLock;

/// High-throughput, partitioned Radix Tree to eliminate global write lock contention.
/// Incoming prefix sequences are hashed to one of N independent trie shards.
pub struct ShardedRadixTree {
    shards: Vec<RwLock<RadixTree>>,
    num_shards: usize,
    block_size_tokens: usize,
}

impl Default for ShardedRadixTree {
    fn default() -> Self {
        Self::new(16, 64)
    }
}

impl ShardedRadixTree {
    pub fn new(num_shards: usize, block_size_tokens: usize) -> Self {
        let n = num_shards.max(1);
        let mut shards = Vec::with_capacity(n);
        for _ in 0..n {
            shards.push(RwLock::new(RadixTree::new(block_size_tokens)));
        }
        Self {
            shards,
            num_shards: n,
            block_size_tokens,
        }
    }

    #[inline]
    fn get_shard_idx(&self, blocks: &[BlockHash]) -> usize {
        if blocks.is_empty() {
            0
        } else {
            // Rapid integer mixer for deterministic partition distribution
            let h = blocks[0].wrapping_mul(0x9E3779B97F4A7C15);
            (h as usize) % self.num_shards
        }
    }

    /// Finds longest prefix on the designated shard without blocking concurrent lookups on other shards.
    pub fn find_longest_prefix(&self, blocks: &[BlockHash]) -> MatchResult {
        if blocks.is_empty() {
            return MatchResult::default();
        }
        let idx = self.get_shard_idx(blocks);
        let shard = self.shards[idx].read();
        shard.find_longest_prefix(blocks)
    }

    /// Inserts prefix into designated shard.
    pub fn insert(&self, blocks: &[BlockHash], worker_id: &str) {
        if blocks.is_empty() {
            return;
        }
        let idx = self.get_shard_idx(blocks);
        let mut shard = self.shards[idx].write();
        shard.insert(blocks, worker_id);
    }

    /// Reserves optimistic in-flight prefill lease on designated shard.
    pub fn reserve_in_flight(&self, blocks: &[BlockHash], worker_id: &str) {
        if blocks.is_empty() {
            return;
        }
        let idx = self.get_shard_idx(blocks);
        let mut shard = self.shards[idx].write();
        shard.reserve_in_flight(blocks, worker_id);
    }

    /// Releases in-flight lease on designated shard.
    pub fn release_in_flight(&self, blocks: &[BlockHash], worker_id: &str, commit: bool) {
        if blocks.is_empty() {
            return;
        }
        let idx = self.get_shard_idx(blocks);
        let mut shard = self.shards[idx].write();
        shard.release_in_flight(blocks, worker_id, commit);
    }

    /// Broadcasts dynamic block eviction across all shards.
    pub fn evict_blocks(&self, worker_id: &str, blocks: &[BlockHash]) {
        for shard_lock in &self.shards {
            let mut shard = shard_lock.write();
            shard.evict_blocks(worker_id, blocks);
        }
    }

    /// Evicts an entire worker from all shards.
    pub fn evict_worker(&self, worker_id: &str) {
        for shard_lock in &self.shards {
            let mut shard = shard_lock.write();
            shard.evict_worker(worker_id);
        }
    }

    /// Aggregates metrics across all shards.
    pub fn stats(&self) -> (usize, usize) {
        let mut total_nodes = 0;
        let mut total_workers = 0;
        for shard_lock in &self.shards {
            let shard = shard_lock.read();
            let (n, w) = shard.stats();
            total_nodes += n;
            total_workers = total_workers.max(w);
        }
        (total_nodes, total_workers)
    }

    pub fn block_size_tokens(&self) -> usize {
        self.block_size_tokens
    }
}
