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

    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(vec![
            HeaderValue::from_static("http://localhost:3000"),
            HeaderValue::from_static("http://127.0.0.1:3000"),
        ]))
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

    let listener = tokio::net::TcpListener::bind("0.0.0.0:5001").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
