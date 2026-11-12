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
        /// UDP bind address for distributed gateway mesh peer gossip (e.g. "0.0.0.0:9090")
        #[arg(long)]
        gossip_bind: Option<String>,
        /// Comma-separated list of peer gateway UDP addresses for cluster cache coherence
        #[arg(long)]
        peers: Option<String>,
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
    /// High-concurrency async socket load generator against /v1/messages or /v1/chat/completions
    LoadTest {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        #[arg(long, default_value_t = 20)]
        concurrency: usize,
        #[arg(long, default_value_t = 200)]
        requests: usize,
        #[arg(long, default_value = "messages")]
        endpoint: String,
        #[arg(long, default_value_t = 4)]
        shared_prefixes: usize,
        #[arg(long, default_value_t = 3)]
        turns: usize,
    },
    /// Launch the interactive terminal dashboard (Ratatui TUI) monitoring the cluster
    Top {
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
            gossip_bind,
            peers,
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

            let mesh_opt = if let Some(bind_str) = gossip_bind {
                let bind_addr: SocketAddr = bind_str.parse()?;
                let peer_addrs: Vec<SocketAddr> = peers
                    .unwrap_or_default()
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse().ok())
                    .collect();

                let config = radix_gate::MeshConfig {
                    node_id: format!("gateway-{}:{}", host, port),
                    bind_addr,
                    peer_addrs,
                };

                let mesh =
                    Arc::new(radix_gate::GatewayMesh::bind(config, radix_tree.clone()).await?);
                let _listener = mesh.clone().start_listener();
                Some(mesh)
            } else {
                None
            };

            let mut app_state_raw = AppState::new(radix_tree, worker_pool.clone(), chunker);
            if let Some(m) = mesh_opt {
                app_state_raw = app_state_raw.with_mesh(m);
            }
            let app_state = Arc::new(app_state_raw);

            let _prober = radix_gate::proxy::start_health_prober(
                worker_pool.clone(),
                reqwest::Client::new(),
                std::time::Duration::from_secs(5),
            );

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
        Commands::LoadTest {
            url,
            concurrency,
            requests,
            endpoint,
            shared_prefixes,
            turns,
        } => {
            let ep = endpoint
                .parse::<radix_gate::LoadTestEndpoint>()
                .unwrap_or(radix_gate::LoadTestEndpoint::Messages);
            println!("\n[RadixGate] Launching async socket load generator...");
            println!("  Target URL:        {url}");
            println!("  Concurrency:       {concurrency} async workers");
            println!("  Total Requests:    {requests}");
            println!("  Endpoint:          {ep:?}");
            println!("  Shared Prefixes:   {shared_prefixes}");
            println!("  Turns Per Session: {turns}");

            let config = radix_gate::LoadTestConfig {
                target_url: url,
                concurrency,
                total_requests: requests,
                endpoint: ep,
                num_shared_prefixes: shared_prefixes,
                turns_per_session: turns,
                request_timeout: std::time::Duration::from_secs(15),
            };

            let runner = radix_gate::LoadTestRunner::new(config);
            match runner.run().await {
                Ok(summary) => {
                    radix_gate::LoadTestRunner::render_summary_table(&summary);
                }
                Err(e) => {
                    eprintln!("[Error] Load generator failed: {e}");
                }
            }
        }
        Commands::Top { url } => {
            let mut app = radix_gate::TuiApp::new(url);
            app.run().await?;
        }
    }

    Ok(())
}
