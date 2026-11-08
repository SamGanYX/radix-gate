use crate::trie::BlockHash;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone)]
struct LruNode {
    block: BlockHash,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Intrusive O(1) doubly-linked list LRU cache for tracking cached token blocks.
#[derive(Debug, Default)]
struct FastLru {
    nodes: Vec<Option<LruNode>>,
    map: HashMap<BlockHash, usize>,
    free_list: Vec<usize>,
    head: Option<usize>, // Least Recently Used
    tail: Option<usize>, // Most Recently Used
    len: usize,
}

impl FastLru {
    fn new() -> Self {
        Self::default()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn contains(&self, block: &BlockHash) -> bool {
        self.map.contains_key(block)
    }

    fn touch(&mut self, block: BlockHash) {
        if let Some(&idx) = self.map.get(&block) {
            self.move_to_tail(idx);
        }
    }

    fn insert(&mut self, block: BlockHash) {
        if let Some(&idx) = self.map.get(&block) {
            self.move_to_tail(idx);
            return;
        }

        let idx = if let Some(free_idx) = self.free_list.pop() {
            self.nodes[free_idx] = Some(LruNode {
                block,
                prev: self.tail,
                next: None,
            });
            free_idx
        } else {
            let new_idx = self.nodes.len();
            self.nodes.push(Some(LruNode {
                block,
                prev: self.tail,
                next: None,
            }));
            new_idx
        };

        if let Some(Some(tail_node)) = self.tail.and_then(|t| self.nodes.get_mut(t)) {
            tail_node.next = Some(idx);
        } else {
            self.head = Some(idx);
        }

        self.tail = Some(idx);
        self.map.insert(block, idx);
        self.len += 1;
    }

    fn pop_lru(&mut self) -> Option<BlockHash> {
        let head_idx = self.head?;
        let (block, next_idx) = {
            let node = self.nodes[head_idx].as_ref()?;
            (node.block, node.next)
        };

        self.head = next_idx;
        if let Some(Some(new_head_node)) = next_idx.and_then(|n| self.nodes.get_mut(n)) {
            new_head_node.prev = None;
        } else {
            self.tail = None;
        }

        self.nodes[head_idx] = None;
        self.free_list.push(head_idx);
        self.map.remove(&block);
        self.len -= 1;

        Some(block)
    }

    fn move_to_tail(&mut self, idx: usize) {
        if self.tail == Some(idx) {
            return;
        }

        let (prev_idx, next_idx) = {
            let node = match self.nodes.get(idx) {
                Some(Some(n)) => n,
                _ => return,
            };
            (node.prev, node.next)
        };

        // Relink surrounding nodes
        if let Some(Some(prev_node)) = prev_idx.and_then(|p| self.nodes.get_mut(p)) {
            prev_node.next = next_idx;
        } else {
            self.head = next_idx;
        }

        if let Some(Some(next_node)) = next_idx.and_then(|n| self.nodes.get_mut(n)) {
            next_node.prev = prev_idx;
        }

        // Attach to current tail
        let old_tail = self.tail;
        if let Some(Some(old_tail_node)) = old_tail.and_then(|t| self.nodes.get_mut(t)) {
            old_tail_node.next = Some(idx);
        }

        if let Some(Some(curr_node)) = self.nodes.get_mut(idx) {
            curr_node.prev = old_tail;
            curr_node.next = None;
        }

        self.tail = Some(idx);
    }
}

/// State of an individual inference worker replica.
#[derive(Debug)]
pub struct WorkerReplica {
    pub id: String,
    pub endpoint_url: String,
    pub max_kv_blocks: usize,
    pub max_concurrency: usize,
    pub active_requests: AtomicUsize,
    pub is_healthy: bool,

    // High-performance O(1) shadow LRU cache state
    lru: FastLru,

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
            lru: FastLru::new(),
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
    /// Updates the shadow LRU cache with any new blocks in O(1) time and returns evicted blocks (if any).
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
            if self.lru.contains(&b) {
                self.lru.touch(b);
            } else {
                while self.lru.len() >= self.max_kv_blocks && self.lru.len() > 0 {
                    if let Some(oldest) = self.lru.pop_lru() {
                        evicted.push(oldest);
                    }
                }
                self.lru.insert(b);
            }
        }

        evicted
    }

    /// Current KV cache block utilization in [0.0, 1.0].
    pub fn cache_utilization(&self) -> f64 {
        if self.max_kv_blocks == 0 {
            return 0.0;
        }
        self.lru.len() as f64 / self.max_kv_blocks as f64
    }

    /// Check if a block is currently in the worker's shadow cache.
    pub fn has_block(&self, block: BlockHash) -> bool {
        self.lru.contains(&block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_lru_eviction() {
        let mut worker =
            WorkerReplica::new("test-worker".into(), "http://localhost:8000".into(), 3, 10);
        let blocks = vec![1, 2, 3];
        let evicted = worker.record_request(&blocks, 0);
        assert!(evicted.is_empty());
        assert_eq!(worker.lru.len(), 3);

        // Access 1 so 2 becomes the least recently used
        worker.record_request(&[1], 1);

        // Insert 4, which should evict 2
        let evicted = worker.record_request(&[4], 0);
        assert_eq!(evicted, vec![2]);
        assert!(worker.has_block(1));
        assert!(!worker.has_block(2));
        assert!(worker.has_block(3));
        assert!(worker.has_block(4));
    }
}
