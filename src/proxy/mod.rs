pub mod handler;
pub mod types;

use axum::Router;
use axum::routing::{get, post};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub use handler::AppState;

/// Assembles the Axum router for the Radix Gateway.
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(handler::health))
        .route("/metrics", get(handler::metrics))
        .route("/api/v1/tree", get(handler::handle_tree))
        .route("/v1/cache/sync", post(handler::handle_cache_sync))
        .route(
            "/v1/chat/completions",
            post(handler::handle_chat_completions),
        )
        .route("/v1/messages", post(handler::handle_anthropic_messages))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
