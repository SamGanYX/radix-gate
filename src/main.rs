use clap::{Parser, Subcommand};
use parking_lot::RwLock;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use radix_gate::proxy::{AppState, create_router};
use radix_gate::shadow_cache::{WorkerPool, WorkerReplica};
use radix_gate::simulator::{BenchmarkConfig, ClusterSimulator};
use radix_gate::tokenizer::BlockChunker;
use radix_gate::trie::RadixTree;

#[derive(Parser)]
#[command(name = "radix-gate")]
#[command(about = "Deterministic Prefix-Aware KV-Cache Gateway & Radix Router for LLM Clusters", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the high-performance reverse proxy server
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value_t = 64)]
        block_size: usize,
        #[arg(long, default_value_t = 4)]
        mock_workers: usize,
        /// Comma-separated list of live upstream worker URLs (e.g. "http://10.0.0.1:8000,http://10.0.0.2:8000")
        #[arg(long)]
        upstreams: Option<String>,
    },
    /// Run the empirical comparative benchmark suite (Round-Robin vs Least-Loaded vs RadixGate)
    Bench {
        #[arg(long, default_value_t = 4)]
        workers: usize,
        #[arg(long, default_value_t = 800)]
        requests: usize,
        #[arg(long, default_value_t = 500)]
        kv_blocks: usize,
        #[arg(long, default_value_t = 6)]
        agents: usize,
        #[arg(long, default_value_t = 64)]
        block_size: usize,
    },
    /// Inspect the active prefix tree topology via ASCII visualizer from a running instance
    Tree {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "radix_gate=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Bench {
            workers,
            requests,
            kv_blocks,
            agents,
            block_size,
        } => {
            println!("\n[RadixGate] Initializing comparative KV-cache benchmark...");
            println!("  Workers:               {workers}");
            println!("  Total Requests:        {requests}");
            println!("  Worker KV Capacity:    {kv_blocks} blocks");
            println!("  Agent Templates:       {agents}");
            println!("  Block Window:          {block_size} tokens");

            let config = BenchmarkConfig {
                num_workers: workers,
                worker_kv_blocks: kv_blocks,
                total_requests: requests,
                num_distinct_agent_templates: agents,
                turns_per_session: 5,
                block_size,
            };

            let simulator = ClusterSimulator::new(config);
            let results = simulator.run_comparative_suite();
            ClusterSimulator::render_results_table(&results);
        }
        Commands::Serve {
            host,
            port,
            block_size,
            mock_workers,
            upstreams,
        } => {
            let radix_tree = Arc::new(RwLock::new(RadixTree::new(block_size)));
            let worker_pool = Arc::new(WorkerPool::new());
            let chunker = BlockChunker::new(block_size);

            if let Some(upstream_list) = upstreams {
                let urls: Vec<&str> = upstream_list
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();
                for (idx, url) in urls.into_iter().enumerate() {
                    let id = format!("upstream-{:02}", idx + 1);
                    worker_pool.add_worker(WorkerReplica::new(id, url.to_string(), 20_000, 64));
                }
            } else {
                for i in 0..mock_workers {
                    let id = format!("worker-{:02}", i + 1);
                    let url = format!("http://127.0.0.1:800{}", i + 1);
                    worker_pool.add_worker(WorkerReplica::new(id, url, 10_000, 32));
                }
            }

            let app_state = Arc::new(AppState::new(radix_tree, worker_pool.clone(), chunker));

            let router = create_router(app_state);
            let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
            info!("[RadixGate] Server listening on http://{}", addr);
            info!(
                "[RadixGate] Registered {} active inference workers",
                worker_pool.len()
            );
            info!(
                "[RadixGate] Endpoints: POST /v1/chat/completions | POST /v1/messages | GET /metrics"
            );

            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, router).await?;
        }
        Commands::Tree { url } => {
            let target_url = format!("{}/api/v1/tree", url.trim_end_matches('/'));
            println!(
                "[RadixGate] Querying Radix Tree visualizer from {}...",
                target_url
            );
            let client = reqwest::Client::new();
            match client.get(&target_url).send().await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let text = resp.text().await?;
                        println!("\n=== LIVE RADIX TREE TOPOLOGY ===");
                        print!("{}", text);
                        println!("================================\n");
                    } else {
                        eprintln!(
                            "[Error] Server returned HTTP {}: {}",
                            resp.status(),
                            resp.text().await.unwrap_or_default()
                        );
                    }
                }
                Err(err) => {
                    eprintln!(
                        "[Error] Failed to connect to radix-gate instance at {}: {}",
                        target_url, err
                    );
                }
            }
        }
    }

    Ok(())
}
