pub mod node;

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
    /// Number of matching blocks for each worker.
    pub worker_matches: HashMap<WorkerId, usize>,
    /// Longest match length across all workers in blocks.
    pub max_matched_blocks: usize,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new(64) // default 64 tokens per block
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
            // Find common prefix between child.prefix and blocks
            let common_len = child
                .prefix
                .iter()
                .zip(blocks.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common_len == child.prefix.len() {
                // Entire child prefix matches, recurse into child
                child.cached_workers.insert(Arc::clone(worker_id));
                child.last_accessed = Instant::now();
                child.hit_count += 1;
                Self::insert_recursive(child, &blocks[common_len..], worker_id);
            } else {
                // Split child node at common_len
                let split_key = child.prefix[common_len];
                let split_node = RadixNode {
                    prefix: child.prefix[common_len..].to_vec(),
                    cached_workers: child.cached_workers.clone(),
                    children: std::mem::take(&mut child.children),
                    last_accessed: child.last_accessed,
                    hit_count: child.hit_count,
                };

                // Truncate existing child prefix to the common part
                child.prefix.truncate(common_len);
                child.children.clear();
                child.cached_workers.insert(Arc::clone(worker_id));
                child.last_accessed = Instant::now();
                child.hit_count += 1;

                if common_len == blocks.len() {
                    // Exact match at split point
                    child.children.insert(split_key, split_node);
                } else {
                    // New branch for the rest of blocks
                    let new_branch_key = blocks[common_len];
                    let new_node =
                        RadixNode::new(blocks[common_len..].to_vec(), Some(Arc::clone(worker_id)));
                    child.children.insert(split_key, split_node);
                    child.children.insert(new_branch_key, new_node);
                }
            }
        } else {
            // No existing branch starts with `first`, create a new one
            let new_node = RadixNode::new(blocks.to_vec(), Some(Arc::clone(worker_id)));
            current.children.insert(first, new_node);
        }
    }

    /// Finds the Longest Common Prefix (LCP) match length for all workers.
    /// Returns the number of consecutive matched blocks for each worker.
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

                // For all workers on this child, record that they matched up to this depth
                for worker in &child.cached_workers {
                    let entry = result.worker_matches.entry(Arc::clone(worker)).or_insert(0);
                    if matched_depth > *entry {
                        *entry = matched_depth;
                    }
                    if matched_depth > result.max_matched_blocks {
                        result.max_matched_blocks = matched_depth;
                    }
                }

                if match_len == child.prefix.len() {
                    // Consumed entire edge, descend into children
                    remaining = &remaining[match_len..];
                    current = child;
                } else {
                    // Partial edge match, cannot descend further
                    break;
                }
            } else {
                break;
            }
        }

        result
    }

    /// Evicts a worker's registration from nodes if a cache eviction occurred on that worker.
    pub fn evict_worker(&mut self, worker_id: &str) {
        Self::evict_worker_recursive(&mut self.root, worker_id);
    }

    fn evict_worker_recursive(node: &mut RadixNode, worker_id: &str) {
        node.cached_workers.retain(|w| &**w != worker_id);
        for child in node.children.values_mut() {
            Self::evict_worker_recursive(child, worker_id);
        }
    }

    /// Prunes tree branches that have zero active cached workers.
    pub fn prune_empty_nodes(&mut self) {
        Self::prune_recursive(&mut self.root);
    }

    fn prune_recursive(node: &mut RadixNode) -> bool {
        node.children
            .retain(|_, child| Self::prune_recursive(child));
        !node.children.is_empty() || !node.cached_workers.is_empty()
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_tree_basic_insertion_and_matching() {
        let mut trie = RadixTree::new(64);
        let prefix1 = vec![101, 102, 103, 104];
        let prefix2 = vec![101, 102, 201, 202];

        trie.insert(&prefix1, "worker-1");
        trie.insert(&prefix2, "worker-2");

        // Query matching prefix1
        let match1 = trie.find_longest_prefix(&[101, 102, 103, 104]);
        assert_eq!(match1.worker_matches.get("worker-1"), Some(&4));
        assert_eq!(match1.worker_matches.get("worker-2"), Some(&2));
        assert_eq!(match1.max_matched_blocks, 4);

        // Query matching partial common prefix
        let match_partial = trie.find_longest_prefix(&[101, 102, 999]);
        assert_eq!(match_partial.worker_matches.get("worker-1"), Some(&2));
        assert_eq!(match_partial.worker_matches.get("worker-2"), Some(&2));
        assert_eq!(match_partial.max_matched_blocks, 2);

        // Query completely disjoint prefix
        let match_disjoint = trie.find_longest_prefix(&[999, 888]);
        assert!(match_disjoint.worker_matches.is_empty());
        assert_eq!(match_disjoint.max_matched_blocks, 0);
    }

    #[test]
    fn test_radix_tree_branch_split() {
        let mut trie = RadixTree::new(64);
        let prefix_a = vec![1, 2, 3, 4, 5];
        trie.insert(&prefix_a, "worker-a");

        let prefix_b = vec![1, 2, 7, 8];
        trie.insert(&prefix_b, "worker-b");

        let match_b = trie.find_longest_prefix(&[1, 2, 7, 8, 9]);
        assert_eq!(match_b.worker_matches.get("worker-b"), Some(&4));
        assert_eq!(match_b.worker_matches.get("worker-a"), Some(&2));
    }
}
