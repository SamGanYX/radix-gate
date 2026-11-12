pub mod handler;
pub mod streaming;
pub mod types;

use axum::Router;
use axum::routing::{get, post};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub use handler::AppState;
pub use streaming::{InFlightGuard, TrackedStream};

/// Spawns a background worker health prober that queries `{worker_url}/health` every interval,
/// automatically recovering workers once they return 200 OK.
pub fn start_health_prober(
    worker_pool: Arc<crate::shadow_cache::WorkerPool>,
    client: reqwest::Client,
    interval: std::time::Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            let worker_ids = worker_pool.get_worker_ids();
            for id in worker_ids {
                if let Some(w_arc) = worker_pool.get_worker(&id) {
                    let (url, is_healthy) = {
                        let w = w_arc.read();
                        (w.endpoint_url.clone(), w.is_healthy)
                    };
                    if !is_healthy
                        && !url.contains(":800")
                        && (url.starts_with("http://") || url.starts_with("https://"))
                    {
                        let health_url = format!("{}/health", url.trim_end_matches('/'));
                        if let Ok(resp) = client
                            .get(&health_url)
                            .timeout(std::time::Duration::from_secs(2))
                            .send()
                            .await
                            && resp.status().is_success()
                        {
                            tracing::info!(
                                worker = %id,
                                "Unhealthy worker recovered; restored to healthy pool"
                            );
                            w_arc.write().mark_healthy();
                        }
                    }
                }
            }
        }
    })
}

/// Assembles the Axum router for the Radix Gateway.
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(handler::health))
        .route("/metrics", get(handler::metrics))
        .route("/api/v1/tree", get(handler::handle_tree))
        .route("/v1/cache/sync", post(handler::handle_cache_sync))
        .route("/-/reload", post(handler::handle_reload))
        .route(
            "/v1/chat/completions",
            post(handler::handle_chat_completions),
        )
        .route("/v1/messages", post(handler::handle_anthropic_messages))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
