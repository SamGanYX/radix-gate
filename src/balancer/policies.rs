use super::{RoutingDecision, RoutingPolicy};
use crate::shadow_cache::WorkerPool;
use crate::trie::{BlockHash, RadixTree};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Deterministic, Prefix-Aware Radix Router.
/// Balances KV-cache prefix reusability (LCP) against worker in-flight load.
pub struct PrefixAwarePolicy {
    /// Weight given to cache affinity [0.0 - 1.0].
    pub alpha: f64,
    /// Weight given to in-flight load penalty [0.0 - 1.0].
    pub beta: f64,
}

impl Default for PrefixAwarePolicy {
    fn default() -> Self {
        Self {
            alpha: 0.75,
            beta: 0.25,
        }
    }
}

impl PrefixAwarePolicy {
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }
}

impl RoutingPolicy for PrefixAwarePolicy {
    fn select_worker(
        &self,
        pool: &WorkerPool,
        radix_tree: &RadixTree,
        prompt_blocks: &[BlockHash],
    ) -> Option<RoutingDecision> {
        let worker_ids = pool.get_worker_ids();
        if worker_ids.is_empty() {
            return None;
        }

        // Query Radix Tree for longest common prefix matches
        let match_result = radix_tree.find_longest_prefix(prompt_blocks);
        let total_blocks = prompt_blocks.len().max(1);

        let mut best_worker_id = None;
        let mut best_endpoint = String::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_matched_blocks = 0;

        for id in &worker_ids {
            let Some(worker_lock) = pool.get_worker(id) else {
                continue;
            };
            let worker = worker_lock.read();
            if !worker.is_healthy {
                continue;
            }

            let matched = match_result.worker_matches.get(id).copied().unwrap_or(0);
            let cache_affinity = matched as f64 / total_blocks as f64;
            let load_factor = worker.load_factor();

            // Multi-objective score: reward warm KV cache, penalize overloaded workers
            let score = (self.alpha * cache_affinity) - (self.beta * load_factor);

            if score > best_score {
                best_score = score;
                best_worker_id = Some(id.clone());
                best_endpoint = worker.endpoint_url.clone();
                best_matched_blocks = matched;
            }
        }

        best_worker_id.map(|w_id| RoutingDecision {
            worker_id: w_id,
            endpoint_url: best_endpoint,
            matched_blocks: best_matched_blocks,
            total_blocks: prompt_blocks.len(),
            affinity_score: best_score,
            policy_name: "prefix-aware-radix",
        })
    }
}

/// Baseline: Classic Round-Robin load balancer (prefix-blind).
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
}

impl Default for RoundRobinPolicy {
    fn default() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

impl RoutingPolicy for RoundRobinPolicy {
    fn select_worker(
        &self,
        pool: &WorkerPool,
        radix_tree: &RadixTree,
        prompt_blocks: &[BlockHash],
    ) -> Option<RoutingDecision> {
        let worker_ids = pool.get_worker_ids();
        if worker_ids.is_empty() {
            return None;
        }

        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % worker_ids.len();
        let w_id = &worker_ids[idx];
        let worker_lock = pool.get_worker(w_id)?;
        let worker = worker_lock.read();

        // Calculate matched blocks for tracking comparison
        let match_result = radix_tree.find_longest_prefix(prompt_blocks);
        let matched = match_result.worker_matches.get(w_id).copied().unwrap_or(0);

        Some(RoutingDecision {
            worker_id: w_id.clone(),
            endpoint_url: worker.endpoint_url.clone(),
            matched_blocks: matched,
            total_blocks: prompt_blocks.len(),
            affinity_score: 0.0,
            policy_name: "round-robin",
        })
    }
}

/// Baseline: Least-Loaded (min in-flight requests) balancer (prefix-blind).
#[derive(Default)]
pub struct LeastLoadedPolicy;

impl RoutingPolicy for LeastLoadedPolicy {
    fn select_worker(
        &self,
        pool: &WorkerPool,
        radix_tree: &RadixTree,
        prompt_blocks: &[BlockHash],
    ) -> Option<RoutingDecision> {
        let worker_ids = pool.get_worker_ids();
        if worker_ids.is_empty() {
            return None;
        }

        let mut best_id = None;
        let mut min_load = usize::MAX;
        let mut endpoint = String::new();

        for id in &worker_ids {
            let Some(worker_lock) = pool.get_worker(id) else {
                continue;
            };
            let worker = worker_lock.read();
            if !worker.is_healthy {
                continue;
            }
            let load = worker.load();
            if load < min_load {
                min_load = load;
                best_id = Some(id.clone());
                endpoint = worker.endpoint_url.clone();
            }
        }

        let chosen_id = best_id.or_else(|| worker_ids.first().cloned())?;
        let match_result = radix_tree.find_longest_prefix(prompt_blocks);
        let matched = match_result
            .worker_matches
            .get(&chosen_id)
            .copied()
            .unwrap_or(0);

        Some(RoutingDecision {
            worker_id: chosen_id,
            endpoint_url: endpoint,
            matched_blocks: matched,
            total_blocks: prompt_blocks.len(),
            affinity_score: 0.0,
            policy_name: "least-loaded",
        })
    }
}
