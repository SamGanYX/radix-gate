pub mod node;
pub mod sharded;

pub use node::{BlockHash, RadixNode, WorkerId};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

/// A thread-safe Radix Tree (compact Trie) for matching sequences of prompt token block hashes
/// across a distributed cluster of LLM inference workers.
#[derive(Debug)]
pub struct RadixTree {
    pub root: RadixNode,
    pub block_size_tokens: usize,
}

#[derive(Debug, Clone, Default)]
pub struct MatchResult {
    /// Number of matching blocks confirmed in cache for each worker.
    pub worker_matches: HashMap<WorkerId, usize>,
    /// Number of matching blocks currently being prefilled in-flight by a worker.
    pub in_flight_matches: HashMap<WorkerId, usize>,
    /// Longest match length across all workers in blocks.
    pub max_matched_blocks: usize,
    /// Indicates whether any matched branch has high cumulative access velocity / is a hotspot.
    pub is_hotspot: bool,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new(64)
    }
}

impl RadixTree {
    /// Creates a new Radix Tree with a specified token block size.
    pub fn new(block_size_tokens: usize) -> Self {
        Self {
            root: RadixNode::new(Vec::new(), None),
            block_size_tokens,
        }
    }

    /// Inserts a sequence of block hashes associated with a worker that cached them.
    pub fn insert(&mut self, blocks: &[BlockHash], worker_id: &str) {
        if blocks.is_empty() {
            return;
        }
        let w_arc: WorkerId = Arc::from(worker_id);
        Self::insert_recursive(&mut self.root, blocks, &w_arc);
    }

    fn insert_recursive(current: &mut RadixNode, blocks: &[BlockHash], worker_id: &WorkerId) {
        if blocks.is_empty() {
            current.cached_workers.insert(Arc::clone(worker_id));
            current.last_accessed = Instant::now();
            current.hit_count += 1;
            return;
        }

        let first = blocks[0];

        if let Some(child) = current.children.get_mut(&first) {
            let common_len = child
                .prefix
                .iter()
                .zip(blocks.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common_len == child.prefix.len() {
                child.cached_workers.insert(Arc::clone(worker_id));
                child.last_accessed = Instant::now();
                child.hit_count += 1;
                Self::insert_recursive(child, &blocks[common_len..], worker_id);
            } else {
                let split_key = child.prefix[common_len];
                let split_node = RadixNode {
                    prefix: child.prefix[common_len..].to_vec(),
                    cached_workers: child.cached_workers.clone(),
                    in_flight_workers: child.in_flight_workers.clone(),
                    children: std::mem::take(&mut child.children),
                    last_accessed: child.last_accessed,
                    hit_count: child.hit_count,
                };

                child.prefix.truncate(common_len);
                child.children.clear();
                child.cached_workers.insert(Arc::clone(worker_id));
                child.last_accessed = Instant::now();
                child.hit_count += 1;

                if common_len == blocks.len() {
                    child.children.insert(split_key, split_node);
                } else {
                    let new_branch_key = blocks[common_len];
                    let new_node =
                        RadixNode::new(blocks[common_len..].to_vec(), Some(Arc::clone(worker_id)));
                    child.children.insert(split_key, split_node);
                    child.children.insert(new_branch_key, new_node);
                }
            }
        } else {
            let new_node = RadixNode::new(blocks.to_vec(), Some(Arc::clone(worker_id)));
            current.children.insert(first, new_node);
        }
    }

    /// Optimistically reserves an in-flight prefix prefill lease for a worker.
    /// This prevents thundering herd cold prefills across other replicas.
    pub fn reserve_in_flight(&mut self, blocks: &[BlockHash], worker_id: &str) {
        if blocks.is_empty() {
            return;
        }
        let w_arc: WorkerId = Arc::from(worker_id);
        Self::reserve_in_flight_recursive(&mut self.root, blocks, &w_arc);
    }

    fn reserve_in_flight_recursive(
        current: &mut RadixNode,
        blocks: &[BlockHash],
        worker_id: &WorkerId,
    ) {
        if blocks.is_empty() {
            return;
        }

        let first = blocks[0];
        if let Some(child) = current.children.get_mut(&first) {
            let common_len = child
                .prefix
                .iter()
                .zip(blocks.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common_len == child.prefix.len() {
                *child
                    .in_flight_workers
                    .entry(Arc::clone(worker_id))
                    .or_insert(0) += 1;
                Self::reserve_in_flight_recursive(child, &blocks[common_len..], worker_id);
            }
        }
    }

    /// Releases an in-flight reservation. If commit is true, transitions the prefix into confirmed cached state.
    pub fn release_in_flight(&mut self, blocks: &[BlockHash], worker_id: &str, commit: bool) {
        if blocks.is_empty() {
            return;
        }
        let w_arc: WorkerId = Arc::from(worker_id);
        Self::release_in_flight_recursive(&mut self.root, blocks, &w_arc, commit);
    }

    /// Atomically commits prefix insertion and releases in-flight prefill leases in a single write pass.
    pub fn commit_in_flight(&mut self, blocks: &[BlockHash], worker_id: &str) {
        if blocks.is_empty() {
            return;
        }
        self.insert(blocks, worker_id);
        let w_arc: WorkerId = Arc::from(worker_id);
        Self::release_in_flight_recursive(&mut self.root, blocks, &w_arc, false);
    }

    fn release_in_flight_recursive(
        current: &mut RadixNode,
        blocks: &[BlockHash],
        worker_id: &WorkerId,
        commit: bool,
    ) {
        if blocks.is_empty() {
            return;
        }

        let first = blocks[0];
        if let Some(child) = current.children.get_mut(&first) {
            let common_len = child
                .prefix
                .iter()
                .zip(blocks.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common_len == child.prefix.len() {
                if let Some(count) = child.in_flight_workers.get_mut(worker_id) {
                    if *count <= 1 {
                        child.in_flight_workers.remove(worker_id);
                    } else {
                        *count -= 1;
                    }
                }
                if commit {
                    child.cached_workers.insert(Arc::clone(worker_id));
                }
                Self::release_in_flight_recursive(child, &blocks[common_len..], worker_id, commit);
            }
        }
    }

    /// Finds the Longest Common Prefix (LCP) match length for all workers,
    /// returning both confirmed cache hits and active in-flight prefill matches.
    pub fn find_longest_prefix(&self, blocks: &[BlockHash]) -> MatchResult {
        let mut result = MatchResult::default();
        if blocks.is_empty() {
            return result;
        }

        let mut current = &self.root;
        let mut matched_depth = 0;
        let mut remaining = blocks;

        while !remaining.is_empty() {
            let first = remaining[0];
            if let Some(child) = current.children.get(&first) {
                let match_len = child
                    .prefix
                    .iter()
                    .zip(remaining.iter())
                    .take_while(|(a, b)| a == b)
                    .count();

                if match_len == 0 {
                    break;
                }

                matched_depth += match_len;
                if child.hit_count >= 5 {
                    result.is_hotspot = true;
                }

                // Track confirmed cached workers
                for worker in &child.cached_workers {
                    let entry = result.worker_matches.entry(Arc::clone(worker)).or_insert(0);
                    if matched_depth > *entry {
                        *entry = matched_depth;
                    }
                    if matched_depth > result.max_matched_blocks {
                        result.max_matched_blocks = matched_depth;
                    }
                }

                // Track in-flight prefilling workers for request coalescing
                for (worker, &count) in &child.in_flight_workers {
                    if count > 0 {
                        let entry = result
                            .in_flight_matches
                            .entry(Arc::clone(worker))
                            .or_insert(0);
                        if matched_depth > *entry {
                            *entry = matched_depth;
                        }
                    }
                }

                if match_len == child.prefix.len() {
                    remaining = &remaining[match_len..];
                    current = child;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        result
    }

    /// Evicts specific blocks for a worker (handling dynamic upstream eviction feedback).
    pub fn evict_blocks(&mut self, worker_id: &str, blocks: &[BlockHash]) {
        for &b in blocks {
            Self::evict_single_block_recursive(&mut self.root, worker_id, b);
        }
        self.prune_empty_nodes();
    }

    fn evict_single_block_recursive(node: &mut RadixNode, worker_id: &str, target: BlockHash) {
        if node.prefix.contains(&target) {
            node.cached_workers.retain(|w| &**w != worker_id);
        }
        for child in node.children.values_mut() {
            Self::evict_single_block_recursive(child, worker_id, target);
        }
    }

    /// Evicts a worker's registration across the entire tree (e.g. upon worker crash or reboot).
    pub fn evict_worker(&mut self, worker_id: &str) {
        Self::evict_worker_recursive(&mut self.root, worker_id);
    }

    fn evict_worker_recursive(node: &mut RadixNode, worker_id: &str) {
        node.cached_workers.retain(|w| &**w != worker_id);
        node.in_flight_workers.retain(|w, _| &**w != worker_id);
        for child in node.children.values_mut() {
            Self::evict_worker_recursive(child, worker_id);
        }
    }

    /// Prunes tree branches that have zero active cached workers and zero in-flight leases.
    pub fn prune_empty_nodes(&mut self) {
        Self::prune_recursive(&mut self.root);
    }

    fn prune_recursive(node: &mut RadixNode) -> bool {
        node.children
            .retain(|_, child| Self::prune_recursive(child));
        !node.children.is_empty()
            || !node.cached_workers.is_empty()
            || !node.in_flight_workers.is_empty()
    }

    /// Computes summary metrics of the current Radix Tree state.
    pub fn stats(&self) -> (usize, usize) {
        let mut total_nodes = 0;
        let mut all_workers = HashSet::new();
        Self::collect_stats(&self.root, &mut total_nodes, &mut all_workers);
        (total_nodes, all_workers.len())
    }

    fn collect_stats(node: &RadixNode, count: &mut usize, workers: &mut HashSet<WorkerId>) {
        *count += 1;
        for w in &node.cached_workers {
            workers.insert(Arc::clone(w));
        }
        for child in node.children.values() {
            Self::collect_stats(child, count, workers);
        }
    }

    /// Renders an ASCII visual tree representation of active prefix branches, cached workers, and depths.
    pub fn render_ascii(&self) -> String {
        let mut out = format!("ROOT [block_size: {} tokens]\n", self.block_size_tokens);
        Self::render_node_ascii(&self.root, "", &mut out);
        out
    }

    fn render_node_ascii(node: &RadixNode, prefix: &str, out: &mut String) {
        let mut children: Vec<(&BlockHash, &RadixNode)> = node.children.iter().collect();
        children.sort_by_key(|(k, _)| **k);
        let child_count = children.len();

        for (idx, (_key, child)) in children.iter().enumerate() {
            let is_last = idx + 1 == child_count;
            let branch = if is_last { "└── " } else { "├── " };
            let next_prefix = if is_last { "    " } else { "│   " };

            let block_range = if child.prefix.len() == 1 {
                format!("[0x{:08x}]", child.prefix[0])
            } else if child.prefix.len() > 1 {
                format!(
                    "[0x{:08x}..0x{:08x}] ({} blk)",
                    child.prefix[0],
                    child.prefix.last().copied().unwrap_or(0),
                    child.prefix.len()
                )
            } else {
                "[]".to_string()
            };

            let mut cached: Vec<&str> = child.cached_workers.iter().map(|w| &**w).collect();
            cached.sort();
            let cached_str = if cached.is_empty() {
                "none".to_string()
            } else {
                cached.join(", ")
            };

            let in_flight_total: usize = child.in_flight_workers.values().sum();
            let in_flight_str = if in_flight_total > 0 {
                format!(" [InFlight: {}]", in_flight_total)
            } else {
                String::new()
            };

            out.push_str(&format!(
                "{}{}{} [Cached: {}]{} (Hits: {})\n",
                prefix, branch, block_range, cached_str, in_flight_str, child.hit_count
            ));

            let combined_prefix = format!("{}{}", prefix, next_prefix);
            Self::render_node_ascii(child, &combined_prefix, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_tree_render_ascii() {
        let mut trie = RadixTree::new(64);
        trie.insert(&[0x1000, 0x2000], "worker-01");
        trie.insert(&[0x1000, 0x3000], "worker-02");

        let ascii = trie.render_ascii();
        assert!(ascii.contains("ROOT"));
        assert!(ascii.contains("[Cached: worker-01]"));
        assert!(ascii.contains("[Cached: worker-02]"));
    }

    #[test]
    fn test_radix_tree_in_flight_coalescing() {
        let mut trie = RadixTree::new(64);
        let prefix = vec![101, 102, 103, 104];

        // Worker 1 starts prefilling prefix (optimistic lease)
        trie.insert(&prefix, "worker-1");
        trie.reserve_in_flight(&prefix, "worker-1");

        // Concurrent request arrives before Worker 1 finishes
        let match_res = trie.find_longest_prefix(&prefix);
        assert_eq!(match_res.worker_matches.get("worker-1"), Some(&4));
        assert_eq!(match_res.in_flight_matches.get("worker-1"), Some(&4));

        // Release lease upon completion
        trie.release_in_flight(&prefix, "worker-1", true);
        let match_after = trie.find_longest_prefix(&prefix);
        assert_eq!(match_after.worker_matches.get("worker-1"), Some(&4));
        assert!(match_after.in_flight_matches.is_empty());
    }

    #[test]
    fn test_dynamic_eviction_sync() {
        let mut trie = RadixTree::new(64);
        let prefix = vec![501, 502, 503];
        trie.insert(&prefix, "worker-1");

        // Worker evicts block 502 dynamically
        trie.evict_blocks("worker-1", &[502]);
        let match_res = trie.find_longest_prefix(&prefix);
        assert_eq!(match_res.worker_matches.get("worker-1"), None);
    }
}
