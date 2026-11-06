use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, Color, Row, Table};
use rand::seq::SliceRandom;
use rand::Rng;
use std::sync::Arc;

use crate::balancer::policies::{LeastLoadedPolicy, PrefixAwarePolicy, RoundRobinPolicy};
use crate::balancer::RoutingPolicy;
use crate::shadow_cache::{WorkerPool, WorkerReplica};
use crate::trie::RadixTree;

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub num_workers: usize,
    pub worker_kv_blocks: usize,
    pub total_requests: usize,
    pub num_distinct_agent_templates: usize,
    pub turns_per_session: usize,
    pub block_size: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            worker_kv_blocks: 500,
            total_requests: 600,
            num_distinct_agent_templates: 6,
            turns_per_session: 5,
            block_size: 64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PolicyBenchmarkResult {
    pub policy_name: String,
    pub total_requests: usize,
    pub total_prompt_blocks: usize,
    pub total_matched_blocks: usize,
    pub cache_hit_rate: f64,
    pub avg_ttft_ms: f64,
    pub p95_ttft_ms: f64,
    pub p99_ttft_ms: f64,
    pub total_prefill_compute_ms: f64,
}

pub struct ClusterSimulator {
    config: BenchmarkConfig,
}

impl ClusterSimulator {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Generates a realistic synthetic agent workload with shared system/tool prefixes.
    pub fn generate_synthetic_workload(&self) -> Vec<Vec<u64>> {
        let mut rng = rand::thread_rng();
        let mut workload = Vec::with_capacity(self.config.total_requests);

        // Pre-generate agent system prompt templates (e.g. SWE-bench agent, CodeReviewer)
        let mut templates = Vec::new();
        for t_idx in 0..self.config.num_distinct_agent_templates {
            let template_len = rng.gen_range(12..28); // 12-28 blocks (~750-1800 tokens)
            let base_blocks: Vec<u64> = (0..template_len)
                .map(|b| ((t_idx + 1) * 1_000_000 + b) as u64)
                .collect();
            templates.push(base_blocks);
        }

        let num_sessions = self.config.total_requests / self.config.turns_per_session;

        for s_idx in 0..num_sessions {
            // Select template for this session
            let template = templates[s_idx % templates.len()].clone();
            let mut conversation_blocks = template;

            for _turn in 0..self.config.turns_per_session {
                // Each turn adds user query + tool output blocks
                let turn_len = rng.gen_range(2..6);
                let turn_blocks: Vec<u64> = (0..turn_len)
                    .map(|_| rng.gen_range(10_000_000..99_000_000))
                    .collect();

                conversation_blocks.extend_from_slice(&turn_blocks);
                workload.push(conversation_blocks.clone());
            }
        }

        // Shuffle requests slightly to simulate interleaved concurrent arrival
        workload.shuffle(&mut rng);
        workload
    }

    /// Executes a benchmark run for a specific routing policy against the mock cluster.
    pub fn benchmark_policy(
        &self,
        policy: Arc<dyn RoutingPolicy>,
        policy_name: &str,
        workload: &[Vec<u64>],
    ) -> PolicyBenchmarkResult {
        let pool = WorkerPool::new();
        for i in 0..self.config.num_workers {
            let worker = WorkerReplica::new(
                format!("worker-{:02}", i + 1),
                format!("http://127.0.0.1:800{}", i + 1),
                self.config.worker_kv_blocks,
                32,
            );
            pool.add_worker(worker);
        }

        let mut radix_tree = RadixTree::new(self.config.block_size);
        let mut ttft_samples = Vec::with_capacity(workload.len());
        let mut total_matched_blocks = 0;
        let mut total_prompt_blocks = 0;
        let mut total_prefill_compute_ms = 0.0;

        // Simulate concurrent in-flight request queue (e.g. 16 concurrent requests)
        let concurrency_slots = 16;
        let mut in_flight: Vec<(String, usize)> = Vec::new(); // (worker_id, remaining_ticks)

        for prompt_blocks in workload {
            // Tick in-flight requests and decrement active requests
            in_flight.retain_mut(|(w_id, ticks)| {
                if *ticks <= 1 {
                    if let Some(w) = pool.get_worker(w_id) {
                        w.read().active_requests.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    false
                } else {
                    *ticks -= 1;
                    true
                }
            });

            total_prompt_blocks += prompt_blocks.len();

            let decision = policy
                .select_worker(&pool, &radix_tree, prompt_blocks)
                .expect("Worker selection failed");

            total_matched_blocks += decision.matched_blocks;

            // Physics of KV cache latency:
            // Cold prefill = ~1.2ms per block (compute intensive matrix mult)
            // Warm cache hit = ~0.04ms per block (fast memory lookup)
            let cold_blocks = prompt_blocks.len().saturating_sub(decision.matched_blocks);
            let prefill_ms = (cold_blocks as f64 * 1.2) + (decision.matched_blocks as f64 * 0.04);
            let queue_penalty_ms = if in_flight.len() >= concurrency_slots { 2.5 } else { 0.0 };
            let ttft_ms = prefill_ms + queue_penalty_ms;

            ttft_samples.push(ttft_ms);
            total_prefill_compute_ms += prefill_ms;

            // Mark worker as active in-flight
            if let Some(worker_arc) = pool.get_worker(&decision.worker_id) {
                worker_arc.read().active_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                in_flight.push((decision.worker_id.clone(), (prefill_ms as usize).max(2).min(8)));

                let mut worker = worker_arc.write();
                let evicted = worker.record_request(prompt_blocks, decision.matched_blocks);
                radix_tree.insert(prompt_blocks, &decision.worker_id);
                if !evicted.is_empty() {
                    radix_tree.prune_empty_nodes();
                }
            }
        }

        ttft_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let avg_ttft = ttft_samples.iter().sum::<f64>() / ttft_samples.len() as f64;
        let p95_idx = ((ttft_samples.len() as f64) * 0.95) as usize;
        let p99_idx = ((ttft_samples.len() as f64) * 0.99) as usize;

        let hit_rate = if total_prompt_blocks > 0 {
            total_matched_blocks as f64 / total_prompt_blocks as f64
        } else {
            0.0
        };

        PolicyBenchmarkResult {
            policy_name: policy_name.to_string(),
            total_requests: workload.len(),
            total_prompt_blocks,
            total_matched_blocks,
            cache_hit_rate: hit_rate,
            avg_ttft_ms: avg_ttft,
            p95_ttft_ms: ttft_samples[p95_idx.min(ttft_samples.len() - 1)],
            p99_ttft_ms: ttft_samples[p99_idx.min(ttft_samples.len() - 1)],
            total_prefill_compute_ms,
        }
    }

    /// Runs comparative benchmarks across RoundRobin, LeastLoaded, and RadixGate.
    pub fn run_comparative_suite(&self) -> Vec<PolicyBenchmarkResult> {
        let workload = self.generate_synthetic_workload();

        let policies: Vec<(Arc<dyn RoutingPolicy>, &str)> = vec![
            (Arc::new(RoundRobinPolicy::default()), "Round-Robin (Prefix-Blind)"),
            (Arc::new(LeastLoadedPolicy::default()), "Least-Loaded (Prefix-Blind)"),
            (Arc::new(PrefixAwarePolicy::default()), "RadixGate (Prefix-Aware)"),
        ];

        policies
            .into_iter()
            .map(|(pol, name)| self.benchmark_policy(pol, name, &workload))
            .collect()
    }

    /// Renders a terminal comparison table of benchmark results.
    pub fn render_results_table(results: &[PolicyBenchmarkResult]) {
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_header(vec![
                Cell::new("Routing Policy").fg(Color::Cyan),
                Cell::new("Requests"),
                Cell::new("KV Cache Hit Rate").fg(Color::Yellow),
                Cell::new("Avg TTFT (ms)"),
                Cell::new("P95 TTFT (ms)"),
                Cell::new("P99 TTFT (ms)"),
                Cell::new("Prefill Compute (s)").fg(Color::Green),
                Cell::new("Speedup vs RR").fg(Color::Magenta),
            ]);

        let baseline_prefill = results.first().map(|r| r.total_prefill_compute_ms).unwrap_or(1.0);

        for res in results {
            let speedup = baseline_prefill / res.total_prefill_compute_ms.max(0.001);
            let hit_color = if res.cache_hit_rate > 0.6 {
                Color::Green
            } else if res.cache_hit_rate > 0.3 {
                Color::Yellow
            } else {
                Color::Red
            };

            table.add_row(Row::from(vec![
                Cell::new(&res.policy_name),
                Cell::new(res.total_requests.to_string()),
                Cell::new(format!("{:.1}%", res.cache_hit_rate * 100.0)).fg(hit_color),
                Cell::new(format!("{:.2} ms", res.avg_ttft_ms)),
                Cell::new(format!("{:.2} ms", res.p95_ttft_ms)),
                Cell::new(format!("{:.2} ms", res.p99_ttft_ms)),
                Cell::new(format!("{:.2} s", res.total_prefill_compute_ms / 1000.0)),
                Cell::new(format!("{:.2}x", speedup)).fg(Color::Cyan),
            ]));
        }

        println!("\n==========================================================================================");
        println!("               RADIX-GATE: DISTRIBUTED KV-CACHE ROUTER BENCHMARK RESULTS                  ");
        println!("==========================================================================================");
        println!("{table}\n");
    }
}
