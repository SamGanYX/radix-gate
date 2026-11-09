use super::{RoutingDecision, RoutingPolicy};
use crate::shadow_cache::WorkerPool;
use crate::trie::{BlockHash, RadixTree};
use rand::seq::SliceRandom;
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

            let confirmed_matched = match_result
                .worker_matches
                .get(id.as_str())
                .copied()
                .unwrap_or(0);
            let in_flight_matched = match_result
                .in_flight_matches
                .get(id.as_str())
                .copied()
                .unwrap_or(0);

            // In-flight warming prefill is given 90% weight of confirmed cache hit to coalesce thundering herds
            let effective_matched = confirmed_matched as f64 + (0.90 * in_flight_matched as f64);
            let cache_affinity = effective_matched / total_blocks as f64;
            let load_factor = worker.load_factor();

            // Multi-objective score: reward warm KV cache, penalize overloaded workers
            let score = (self.alpha * cache_affinity) - (self.beta * load_factor);

            if score > best_score {
                best_score = score;
                best_worker_id = Some(id.clone());
                best_endpoint = worker.endpoint_url.clone();
                best_matched_blocks = confirmed_matched;
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

/// Power-of-Two-Choices with Prefix Bias (P2C-Prefix).
/// Eliminates the stampede and herd-flapping failure modes of greedy scoring algorithms.
/// Samples two candidate replicas from the set sharing prefix affinity (or healthy pool),
/// routing to the replica with strictly lower in-flight queue depth.
#[derive(Debug, Clone)]
pub struct PowerOfTwoChoicesPolicy {
    /// Saturation threshold after which traffic spills over to trigger replica warming.
    pub saturation_threshold: f64,
}

impl Default for PowerOfTwoChoicesPolicy {
    fn default() -> Self {
        Self {
            saturation_threshold: 0.85,
        }
    }
}

impl RoutingPolicy for PowerOfTwoChoicesPolicy {
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

        let match_result = radix_tree.find_longest_prefix(prompt_blocks);
        let max_matched = match_result.max_matched_blocks;

        // Collect healthy workers and their match lengths
        let mut warm_candidates: Vec<String> = Vec::new();
        let mut healthy_pool: Vec<String> = Vec::new();

        for id in &worker_ids {
            let Some(worker_lock) = pool.get_worker(id) else {
                continue;
            };
            let worker = worker_lock.read();
            if !worker.is_healthy {
                continue;
            }

            healthy_pool.push(id.clone());

            let matched = match_result
                .worker_matches
                .get(id.as_str())
                .copied()
                .unwrap_or(0);

            // Candidate is "warm" if it holds at least 75% of the max common prefix (or max_matched if small)
            if max_matched > 0 && matched >= (max_matched * 3) / 4 {
                warm_candidates.push(id.clone());
            }
        }

        if healthy_pool.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();

        // Select candidate subset: prefer warm pool unless empty or heavily saturated
        let candidates = if warm_candidates.len() >= 2 {
            warm_candidates
        } else if warm_candidates.len() == 1 {
            let warm_id = &warm_candidates[0];
            let is_saturated = pool
                .get_worker(warm_id)
                .map(|w| w.read().load_factor() >= self.saturation_threshold)
                .unwrap_or(false);

            if is_saturated && healthy_pool.len() >= 2 {
                // Warm worker is congested; sample from healthy pool to trigger dynamic prefix replication
                healthy_pool
            } else {
                // Route directly to the sole warm worker
                let worker_lock = pool.get_worker(warm_id)?;
                let worker = worker_lock.read();
                let matched = match_result
                    .worker_matches
                    .get(warm_id.as_str())
                    .copied()
                    .unwrap_or(0);

                return Some(RoutingDecision {
                    worker_id: warm_id.clone(),
                    endpoint_url: worker.endpoint_url.clone(),
                    matched_blocks: matched,
                    total_blocks: prompt_blocks.len(),
                    affinity_score: 1.0,
                    policy_name: "p2c-prefix",
                });
            }
        } else {
            healthy_pool
        };

        // Power of Two Choices: sample 2 random distinct candidates from the pool
        let (choice_a, choice_b) = if candidates.len() == 1 {
            (&candidates[0], &candidates[0])
        } else {
            let sampled: Vec<&String> = candidates.choose_multiple(&mut rng, 2).collect();
            (sampled[0], sampled[1])
        };

        let worker_a = pool.get_worker(choice_a)?;
        let worker_b = pool.get_worker(choice_b)?;

        let load_a = worker_a.read().load();
        let load_b = worker_b.read().load();

        let chosen_id = if load_a <= load_b { choice_a } else { choice_b };
        let chosen_worker = pool.get_worker(chosen_id)?;
        let worker = chosen_worker.read();

        let matched = match_result
            .worker_matches
            .get(chosen_id.as_str())
            .copied()
            .unwrap_or(0);

        Some(RoutingDecision {
            worker_id: chosen_id.clone(),
            endpoint_url: worker.endpoint_url.clone(),
            matched_blocks: matched,
            total_blocks: prompt_blocks.len(),
            affinity_score: 0.5,
            policy_name: "p2c-prefix",
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

        let match_result = radix_tree.find_longest_prefix(prompt_blocks);
        let matched = match_result
            .worker_matches
            .get(w_id.as_str())
            .copied()
            .unwrap_or(0);

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
            .get(chosen_id.as_str())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadow_cache::WorkerReplica;

    #[test]
    fn test_p2c_prefix_load_balancing() {
        let pool = WorkerPool::new();
        pool.add_worker(WorkerReplica::new("w-01".into(), "http://w1".into(), 1000, 10));
        pool.add_worker(WorkerReplica::new("w-02".into(), "http://w2".into(), 1000, 10));

        let mut tree = RadixTree::new(64);
        let prefix = vec![1, 2, 3];
        tree.insert(&prefix, "w-01");
        tree.insert(&prefix, "w-02");

        // Simulate w-01 having 3 active requests, w-02 having 0
        if let Some(w) = pool.get_worker("w-01") {
            w.read().active_requests.store(3, Ordering::SeqCst);
        }

        let policy = PowerOfTwoChoicesPolicy::default();
        let decision = policy.select_worker(&pool, &tree, &prefix).unwrap();

        // P2C must pick w-02 because both hold the prefix but w-02 has lower active load
        assert_eq!(decision.worker_id, "w-02");
        assert_eq!(decision.matched_blocks, 3);
    }
}
