use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, Color, Row, Table};
use serde::{Deserialize, Serialize};

/// Supported API endpoints for load testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadTestEndpoint {
    Messages,        // Anthropic /v1/messages
    ChatCompletions, // OpenAI /v1/chat/completions
}

impl std::str::FromStr for LoadTestEndpoint {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "messages" | "/v1/messages" | "anthropic" => Ok(Self::Messages),
            "chat" | "chat/completions" | "/v1/chat/completions" | "openai" => {
                Ok(Self::ChatCompletions)
            }
            _ => Err(format!(
                "Unknown endpoint '{}', expected 'messages' or 'chat/completions'",
                s
            )),
        }
    }
}

/// Configuration parameters for high-concurrency socket load generation.
#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    pub target_url: String,
    pub concurrency: usize,
    pub total_requests: usize,
    pub endpoint: LoadTestEndpoint,
    pub num_shared_prefixes: usize,
    pub turns_per_session: usize,
    pub request_timeout: Duration,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            target_url: "http://127.0.0.1:8080".to_string(),
            concurrency: 20,
            total_requests: 200,
            endpoint: LoadTestEndpoint::Messages,
            num_shared_prefixes: 4,
            turns_per_session: 3,
            request_timeout: Duration::from_secs(10),
        }
    }
}

/// Result of an individual HTTP socket request.
#[derive(Debug, Clone)]
pub struct RequestMetric {
    pub latency: Duration,
    pub status: u16,
    pub matched_blocks: usize,
    pub total_blocks: usize,
    pub worker_id: String,
    pub is_success: bool,
}

/// Aggregated latency percentiles and cache throughput metrics.
#[derive(Debug, Clone)]
pub struct LoadTestSummary {
    pub total_requests: usize,
    pub success_count: usize,
    pub error_count: usize,
    pub elapsed_time: Duration,
    pub requests_per_sec: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub matched_blocks: usize,
    pub prompt_blocks: usize,
    pub cache_hit_ratio: f64,
    pub worker_distribution: HashMap<String, usize>,
}

/// Core high-concurrency async load test harness.
pub struct LoadTestRunner {
    config: LoadTestConfig,
}

impl LoadTestRunner {
    pub fn new(config: LoadTestConfig) -> Self {
        Self { config }
    }

    /// Predefined production-realistic system prompts to simulate distinct agent templates.
    fn system_prompt_templates() -> [&'static str; 4] {
        [
            "You are an expert Anthropic performance systems engineer reviewing distributed inference architectures. Analyze KV-cache radix trees, chunked prefill, and paged attention.",
            "You are a principal database kernel developer optimizing lock-free concurrency, LSM trees, memory-mapped I/O, and zero-copy ring buffers for ultra-low latency.",
            "You are a compiler engineer designing LLVM middle-end optimizations, loop vectorization, SIMD intrinsics, and instruction scheduling for modern microarchitectures.",
            "You are a high-throughput network systems researcher evaluating eBPF, DPDK packet processing, TCP Congestion Control (BBRv3), and RDMA RoCEv2 clusters.",
        ]
    }

    /// Assembles an OpenAI or Anthropic payload with shared system prefixes and multi-turn user/assistant history.
    pub fn generate_payload(
        endpoint: LoadTestEndpoint,
        template_idx: usize,
        turn_idx: usize,
    ) -> serde_json::Value {
        let templates = Self::system_prompt_templates();
        let system_prompt = templates[template_idx % templates.len()];

        let mut messages = Vec::new();
        messages.push(serde_json::json!({
            "role": "system",
            "content": system_prompt
        }));

        for t in 0..=turn_idx {
            messages.push(serde_json::json!({
                "role": "user",
                "content": format!("Step {}: What are the exact throughput constraints and cache hit bounds for this setup?", t + 1)
            }));
            if t < turn_idx {
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": format!("In step {}, the cache hit ratio is bounded by prefix reuse and working set size.", t + 1)
                }));
            }
        }

        match endpoint {
            LoadTestEndpoint::Messages => serde_json::json!({
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 128,
                "system": system_prompt,
                "messages": messages.into_iter().filter(|m| m["role"] != "system").collect::<Vec<_>>()
            }),
            LoadTestEndpoint::ChatCompletions => serde_json::json!({
                "model": "gpt-4o",
                "messages": messages,
                "max_tokens": 128
            }),
        }
    }

    /// Runs the benchmark against the target URL with C concurrent async tasks.
    pub async fn run(&self) -> Result<LoadTestSummary, Box<dyn std::error::Error + Send + Sync>> {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(self.config.concurrency.max(16))
            .tcp_nodelay(true)
            .timeout(self.config.request_timeout)
            .build()?;

        let endpoint_path = match self.config.endpoint {
            LoadTestEndpoint::Messages => "/v1/messages",
            LoadTestEndpoint::ChatCompletions => "/v1/chat/completions",
        };
        let target_url = format!(
            "{}{}",
            self.config.target_url.trim_end_matches('/'),
            endpoint_path
        );

        let requests_counter = Arc::new(AtomicUsize::new(0));
        let total_requests = self.config.total_requests;
        let start_time = Instant::now();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<RequestMetric>();

        let mut handles = Vec::with_capacity(self.config.concurrency);

        for _ in 0..self.config.concurrency {
            let counter = Arc::clone(&requests_counter);
            let client = client.clone();
            let target_url = target_url.clone();
            let endpoint = self.config.endpoint;
            let num_shared_prefixes = self.config.num_shared_prefixes.max(1);
            let turns_per_session = self.config.turns_per_session.max(1);
            let tx = tx.clone();

            handles.push(tokio::spawn(async move {
                loop {
                    let req_idx = counter.fetch_add(1, Ordering::Relaxed);
                    if req_idx >= total_requests {
                        break;
                    }

                    let prefix_idx = (req_idx / turns_per_session) % num_shared_prefixes;
                    let turn_idx = req_idx % turns_per_session;
                    let payload = LoadTestRunner::generate_payload(endpoint, prefix_idx, turn_idx);

                    let req_start = Instant::now();
                    let response_res = client
                        .post(&target_url)
                        .header("content-type", "application/json")
                        .json(&payload)
                        .send()
                        .await;

                    let latency = req_start.elapsed();

                    match response_res {
                        Ok(resp) => {
                            let status = resp.status().as_u16();
                            let is_success = resp.status().is_success();

                            let matched = resp
                                .headers()
                                .get("X-Radix-Matched-Blocks")
                                .and_then(|h| h.to_str().ok())
                                .and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(0);

                            let total_b = resp
                                .headers()
                                .get("X-Radix-Total-Blocks")
                                .and_then(|h| h.to_str().ok())
                                .and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(0);

                            let worker = resp
                                .headers()
                                .get("X-Radix-Worker")
                                .and_then(|h| h.to_str().ok())
                                .unwrap_or("unknown")
                                .to_string();

                            // Ensure body stream is consumed
                            let _ = resp.bytes().await;

                            let _ = tx.send(RequestMetric {
                                latency,
                                status,
                                matched_blocks: matched,
                                total_blocks: total_b,
                                worker_id: worker,
                                is_success,
                            });
                        }
                        Err(_) => {
                            let _ = tx.send(RequestMetric {
                                latency,
                                status: 0,
                                matched_blocks: 0,
                                total_blocks: 0,
                                worker_id: "error".to_string(),
                                is_success: false,
                            });
                        }
                    }
                }
            }));
        }

        // Drop original tx so rx terminates when all workers finish
        drop(tx);

        let mut metrics = Vec::with_capacity(total_requests);
        while let Some(m) = rx.recv().await {
            metrics.push(m);
        }

        for h in handles {
            let _ = h.await;
        }

        let elapsed = start_time.elapsed();
        Ok(Self::calculate_summary(metrics, elapsed))
    }

    /// Computes statistical metrics across all gathered request metrics.
    pub fn calculate_summary(metrics: Vec<RequestMetric>, elapsed: Duration) -> LoadTestSummary {
        let total = metrics.len();
        let mut success = 0;
        let mut error = 0;
        let mut matched_blocks = 0;
        let mut prompt_blocks = 0;
        let mut worker_distribution = HashMap::new();
        let mut latencies_ms = Vec::with_capacity(total);

        for m in &metrics {
            if m.is_success {
                success += 1;
                matched_blocks += m.matched_blocks;
                prompt_blocks += m.total_blocks;
                *worker_distribution.entry(m.worker_id.clone()).or_insert(0) += 1;
            } else {
                error += 1;
            }
            latencies_ms.push(m.latency.as_secs_f64() * 1000.0);
        }

        latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50 = Self::percentile(&latencies_ms, 50.0);
        let p90 = Self::percentile(&latencies_ms, 90.0);
        let p95 = Self::percentile(&latencies_ms, 95.0);
        let p99 = Self::percentile(&latencies_ms, 99.0);
        let min = latencies_ms.first().copied().unwrap_or(0.0);
        let max = latencies_ms.last().copied().unwrap_or(0.0);
        let mean = if total > 0 {
            latencies_ms.iter().sum::<f64>() / total as f64
        } else {
            0.0
        };

        let elapsed_sec = elapsed.as_secs_f64().max(0.0001);
        let rps = total as f64 / elapsed_sec;

        let hit_ratio = if prompt_blocks > 0 {
            (matched_blocks as f64 / prompt_blocks as f64) * 100.0
        } else {
            0.0
        };

        LoadTestSummary {
            total_requests: total,
            success_count: success,
            error_count: error,
            elapsed_time: elapsed,
            requests_per_sec: rps,
            p50_ms: p50,
            p90_ms: p90,
            p95_ms: p95,
            p99_ms: p99,
            min_ms: min,
            max_ms: max,
            mean_ms: mean,
            matched_blocks,
            prompt_blocks,
            cache_hit_ratio: hit_ratio,
            worker_distribution,
        }
    }

    fn percentile(sorted: &[f64], pct: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let rank = (pct / 100.0) * (sorted.len() - 1) as f64;
        let lower = rank.floor() as usize;
        let upper = rank.ceil() as usize;
        if lower == upper {
            sorted[lower]
        } else {
            let weight = rank - lower as f64;
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        }
    }

    /// Renders an executive ASCII table summarizing load test results.
    pub fn render_summary_table(summary: &LoadTestSummary) {
        println!("\n=== RADIXGATE HIGH-CONCURRENCY SOCKET BENCHMARK RESULTS ===");

        let mut overview_table = Table::new();
        overview_table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_header(vec![
                Cell::new("Metric").fg(Color::Cyan),
                Cell::new("Value").fg(Color::Yellow),
            ]);

        overview_table.add_row(Row::from(vec![
            Cell::new("Total Requests Completed"),
            Cell::new(format!("{}", summary.total_requests)),
        ]));
        overview_table.add_row(Row::from(vec![
            Cell::new("Successful (HTTP 2xx)"),
            Cell::new(format!("{}", summary.success_count)).fg(Color::Green),
        ]));
        overview_table.add_row(Row::from(vec![
            Cell::new("Failed / Errored"),
            Cell::new(format!("{}", summary.error_count)).fg(if summary.error_count > 0 {
                Color::Red
            } else {
                Color::Green
            }),
        ]));
        overview_table.add_row(Row::from(vec![
            Cell::new("Elapsed Benchmark Time"),
            Cell::new(format!("{:.3}s", summary.elapsed_time.as_secs_f64())),
        ]));
        overview_table.add_row(Row::from(vec![
            Cell::new("Throughput (Requests / Sec)"),
            Cell::new(format!("{:.1} req/s", summary.requests_per_sec)).fg(Color::Green),
        ]));
        overview_table.add_row(Row::from(vec![
            Cell::new("Global KV-Cache Hit Ratio"),
            Cell::new(format!("{:.2}%", summary.cache_hit_ratio)).fg(Color::Yellow),
        ]));
        overview_table.add_row(Row::from(vec![
            Cell::new("Matched / Total Blocks"),
            Cell::new(format!(
                "{} / {} blocks",
                summary.matched_blocks, summary.prompt_blocks
            )),
        ]));

        println!("{overview_table}");

        let mut latency_table = Table::new();
        latency_table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_header(vec![
                Cell::new("Latency Percentile").fg(Color::Cyan),
                Cell::new("Duration (ms)").fg(Color::Yellow),
            ]);

        latency_table.add_row(Row::from(vec![
            Cell::new("Min"),
            Cell::new(format!("{:.2} ms", summary.min_ms)),
        ]));
        latency_table.add_row(Row::from(vec![
            Cell::new("P50 (Median)"),
            Cell::new(format!("{:.2} ms", summary.p50_ms)),
        ]));
        latency_table.add_row(Row::from(vec![
            Cell::new("Mean"),
            Cell::new(format!("{:.2} ms", summary.mean_ms)),
        ]));
        latency_table.add_row(Row::from(vec![
            Cell::new("P90"),
            Cell::new(format!("{:.2} ms", summary.p90_ms)),
        ]));
        latency_table.add_row(Row::from(vec![
            Cell::new("P95"),
            Cell::new(format!("{:.2} ms", summary.p95_ms)),
        ]));
        latency_table.add_row(Row::from(vec![
            Cell::new("P99"),
            Cell::new(format!("{:.2} ms", summary.p99_ms)),
        ]));
        latency_table.add_row(Row::from(vec![
            Cell::new("Max"),
            Cell::new(format!("{:.2} ms", summary.max_ms)),
        ]));

        println!("\nLatency Distribution:");
        println!("{latency_table}");

        if !summary.worker_distribution.is_empty() {
            let mut worker_table = Table::new();
            worker_table
                .load_preset(UTF8_FULL)
                .apply_modifier(UTF8_ROUND_CORNERS)
                .set_header(vec![
                    Cell::new("Worker Replica").fg(Color::Cyan),
                    Cell::new("Requests Routed").fg(Color::Yellow),
                    Cell::new("Traffic Share").fg(Color::Magenta),
                ]);

            let mut sorted_workers: Vec<(&String, &usize)> =
                summary.worker_distribution.iter().collect();
            sorted_workers.sort_by_key(|(k, _)| *k);

            for (worker, count) in sorted_workers {
                let share = if summary.success_count > 0 {
                    (*count as f64 / summary.success_count as f64) * 100.0
                } else {
                    0.0
                };
                worker_table.add_row(Row::from(vec![
                    Cell::new(worker),
                    Cell::new(format!("{}", count)),
                    Cell::new(format!("{:.1}%", share)),
                ]));
            }

            println!("\nWorker Traffic Distribution:");
            println!("{worker_table}");
        }

        println!("===========================================================\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p50 = LoadTestRunner::percentile(&values, 50.0);
        let p90 = LoadTestRunner::percentile(&values, 90.0);
        assert!((p50 - 5.5).abs() < 1e-4);
        assert!((p90 - 9.1).abs() < 1e-4);
    }

    #[test]
    fn test_payload_generation() {
        let anthropic = LoadTestRunner::generate_payload(LoadTestEndpoint::Messages, 0, 1);
        assert_eq!(anthropic["model"], "claude-3-5-sonnet-20241022");
        assert!(anthropic["system"].is_string());

        let openai = LoadTestRunner::generate_payload(LoadTestEndpoint::ChatCompletions, 1, 0);
        assert_eq!(openai["model"], "gpt-4o");
        assert!(openai["messages"].is_array());
    }

    #[tokio::test]
    async fn test_high_concurrency_socket_benchmark_against_gateway() {
        use crate::proxy::{AppState, create_router};
        use crate::shadow_cache::{WorkerPool, WorkerReplica};
        use crate::tokenizer::BlockChunker;
        use crate::trie::RadixTree;
        use parking_lot::RwLock;

        let radix_tree = Arc::new(RwLock::new(RadixTree::new(64)));
        let worker_pool = Arc::new(WorkerPool::new());
        worker_pool.add_worker(WorkerReplica::new(
            "worker-01".into(),
            "http://127.0.0.1:8001".into(),
            1000,
            16,
        ));
        worker_pool.add_worker(WorkerReplica::new(
            "worker-02".into(),
            "http://127.0.0.1:8002".into(),
            1000,
            16,
        ));
        let chunker = BlockChunker::new(64);
        let state = Arc::new(AppState::new(radix_tree, worker_pool, chunker));
        let router = create_router(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let config = LoadTestConfig {
            target_url: format!("http://127.0.0.1:{}", port),
            concurrency: 4,
            total_requests: 20,
            endpoint: LoadTestEndpoint::Messages,
            num_shared_prefixes: 2,
            turns_per_session: 2,
            request_timeout: Duration::from_secs(5),
        };

        let runner = LoadTestRunner::new(config);
        let summary = runner.run().await.expect("load test runner failed");

        assert_eq!(summary.total_requests, 20);
        assert_eq!(summary.success_count, 20);
        assert_eq!(summary.error_count, 0);
        assert!(summary.requests_per_sec > 0.0);
        assert!(
            summary.matched_blocks > 0,
            "Multi-turn prompts must hit the prefix cache!"
        );
        assert!(summary.p50_ms > 0.0);
    }
}
