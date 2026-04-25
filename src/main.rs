mod handlers;
mod models;
mod state;
mod wasm_runner;

use axum::{
    Router,
    http::{HeaderValue, Method},
    routing::post,
};
use state::AppState;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let state = AppState::from_files("config.json", "sha3_wasm_bg.7b9ca65ddd.wasm")?;
    let bind_address = state.config.bind_address.clone();
    let cors_origins = state.config.cors_origins.clone();
    let allow_origins = cors_origins
        .iter()
        .map(|origin| HeaderValue::from_str(origin))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("invalid cors origin in config: {e}"))?;

    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(allow_origins))
        .allow_methods(AllowMethods::list([
            Method::GET,
            Method::POST,
            Method::OPTIONS,
        ]))
        .allow_headers(AllowHeaders::any())
        .allow_credentials(true);

    let app = Router::new()
        .route("/chat/completions", post(handlers::chat_completions))
        .route("/claude/messages", post(handlers::claude_messages))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/anthropic/v1/messages", post(handlers::claude_messages))
        .with_state(state)
        .layer(cors);

    let listener = tokio::net::TcpListener::bind(bind_address).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
