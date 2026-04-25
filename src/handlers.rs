use std::{convert::Infallible, time::Duration};

use anyhow::{Context, anyhow};
use axum::{
    body::Body,
    extract::{Json, State},
    http::{HeaderMap, HeaderValue, Response, StatusCode},
    response::{
        IntoResponse,
        sse::{Event, KeepAlive, Sse},
    },
};
use base64::Engine as _;
use futures_util::StreamExt;
use serde_json::{Value, json};

use crate::{
    models::{
        Account, ChatCompletionRequest, ChatMessage, ClaudeRequest, ErrorBody, MessageContent,
    },
    state::AppState,
    wasm_runner::solve_pow,
};

const DEEPSEEK_HOST: &str = "chat.deepseek.com";
const DEEPSEEK_LOGIN_URL: &str = "https://chat.deepseek.com/api/v0/users/login";
const DEEPSEEK_CREATE_SESSION_URL: &str = "https://chat.deepseek.com/api/v0/chat_session/create";
const DEEPSEEK_CREATE_POW_URL: &str = "https://chat.deepseek.com/api/v0/chat/create_pow_challenge";
const DEEPSEEK_COMPLETION_URL: &str = "https://chat.deepseek.com/api/v0/chat/completion";

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    msg: String,
}

impl ApiError {
    fn new(status: StatusCode, msg: impl Into<String>) -> Self {
        Self {
            status,
            msg: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(ErrorBody { error: self.msg })).into_response()
    }
}

fn base_headers(token: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("host", HeaderValue::from_static(DEEPSEEK_HOST));
    headers.insert("accept", HeaderValue::from_static("application/json"));
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    headers.insert("x-client-platform", HeaderValue::from_static("android"));
    headers.insert(
        "x-client-version",
        HeaderValue::from_static("1.3.0-auto-resume"),
    );
    headers.insert("x-client-locale", HeaderValue::from_static("zh_CN"));
    if let Ok(v) = HeaderValue::from_str(&format!("Bearer {token}")) {
        headers.insert("authorization", v);
    }
    headers
}

fn extract_bearer(headers: &HeaderMap) -> Result<String, ApiError> {
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| ApiError::new(StatusCode::UNAUTHORIZED, "missing bearer token"))?;

    let token = auth
        .strip_prefix("Bearer ")
        .ok_or_else(|| ApiError::new(StatusCode::UNAUTHORIZED, "invalid authorization format"))?
        .trim();

    if token.len() < 8 || token.len() > 2048 || token.chars().any(char::is_whitespace) {
        return Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "invalid bearer token",
        ));
    }

    Ok(token.to_string())
}

async fn login_account(state: &AppState, account: &mut Account) -> anyhow::Result<()> {
    let password = account.password.trim();
    let email = account.email.clone().unwrap_or_default();
    let mobile = account.mobile.clone().unwrap_or_default();

    if password.is_empty() || (email.trim().is_empty() && mobile.trim().is_empty()) {
        return Err(anyhow!("account is missing credentials"));
    }

    let payload = if !email.trim().is_empty() {
        json!({
            "email": email,
            "password": password,
            "device_id": "deepseek_to_api",
            "os": "android"
        })
    } else {
        json!({
            "mobile": mobile,
            "area_code": Value::Null,
            "password": password,
            "device_id": "deepseek_to_api",
            "os": "android"
        })
    };

    let resp = state
        .safari_client
        .post(DEEPSEEK_LOGIN_URL)
        .json(&payload)
        .send()
        .await
        .context("login request failed")?;

    if !resp.status().is_success() {
        return Err(anyhow!("login failed with status {}", resp.status()));
    }

    let body: Value = resp.json().await.context("invalid login json")?;
    let token = body
        .pointer("/data/biz_data/user/token")
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow!("missing login token in response"))?;

    account.token = Some(token.to_string());
    Ok(())
}

async fn resolve_deepseek_token(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<(String, Option<Account>), ApiError> {
    let bearer = extract_bearer(headers)?;

    if !state.keys.contains(&bearer) {
        return Ok((bearer, None));
    }

    let mut account = state.checkout_account().await.ok_or_else(|| {
        ApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "no available configured accounts",
        )
    })?;

    if account.token.clone().unwrap_or_default().trim().is_empty() {
        login_account(state, &mut account).await.map_err(|e| {
            ApiError::new(
                StatusCode::BAD_GATEWAY,
                format!("account login failed: {e}"),
            )
        })?;
    }

    let token = account.token.clone().ok_or_else(|| {
        ApiError::new(StatusCode::BAD_GATEWAY, "configured account token missing")
    })?;

    Ok((token, Some(account)))
}

fn to_prompt(messages: &[ChatMessage]) -> String {
    let mut processed: Vec<(String, String)> = messages
        .iter()
        .map(|m| (m.role.clone(), m.normalized_text()))
        .collect();

    if processed.is_empty() {
        return String::new();
    }

    let mut merged: Vec<(String, String)> = vec![processed.remove(0)];
    for (role, text) in processed {
        if let Some((last_role, last_text)) = merged.last_mut()
            && *last_role == role
        {
            last_text.push_str("\n\n");
            last_text.push_str(&text);
        } else {
            merged.push((role, text));
        }
    }

    merged
        .into_iter()
        .enumerate()
        .map(|(idx, (role, text))| match role.as_str() {
            "assistant" => format!("<｜Assistant｜>{text}<｜end▁of▁sentence｜>"),
            "user" | "system" if idx > 0 => format!("<｜User｜>{text}"),
            _ => text,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn model_flags(model: &str) -> Result<(bool, bool), ApiError> {
    match model.to_lowercase().as_str() {
        "deepseek-v3" | "deepseek-chat" => Ok((false, false)),
        "deepseek-r1" | "deepseek-reasoner" => Ok((true, false)),
        "deepseek-v3-search" | "deepseek-chat-search" => Ok((false, true)),
        "deepseek-r1-search" | "deepseek-reasoner-search" => Ok((true, true)),
        other => Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("unsupported model '{other}'"),
        )),
    }
}

async fn create_session(state: &AppState, token: &str) -> anyhow::Result<String> {
    let resp = state
        .chrome_client
        .post(DEEPSEEK_CREATE_SESSION_URL)
        .headers(base_headers(token))
        .json(&json!({"agent": "chat"}))
        .send()
        .await
        .context("create session request failed")?;

    let body: Value = resp.json().await.context("invalid create session json")?;
    if body.get("code").and_then(Value::as_i64) != Some(0) {
        return Err(anyhow!("create session failed: {body}"));
    }

    body.pointer("/data/biz_data/id")
        .and_then(Value::as_str)
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("create session missing id"))
}

async fn create_pow(state: &AppState, token: &str) -> anyhow::Result<String> {
    let resp = state
        .safari_client
        .post(DEEPSEEK_CREATE_POW_URL)
        .headers(base_headers(token))
        .json(&json!({"target_path":"/api/v0/chat/completion"}))
        .send()
        .await
        .context("pow challenge request failed")?;

    let body: Value = resp.json().await.context("invalid pow json")?;
    if body.get("code").and_then(Value::as_i64) != Some(0) {
        return Err(anyhow!("pow challenge failed: {body}"));
    }

    let challenge = body
        .pointer("/data/biz_data/challenge")
        .cloned()
        .ok_or_else(|| anyhow!("pow challenge missing challenge data"))?;

    let algorithm = challenge
        .get("algorithm")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if algorithm != "DeepSeekHashV1" {
        return Err(anyhow!("unsupported pow algorithm {algorithm}"));
    }

    let challenge_str = challenge
        .get("challenge")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing challenge string"))?;
    let salt = challenge
        .get("salt")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing challenge salt"))?;
    let signature = challenge
        .get("signature")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing challenge signature"))?;
    let target_path = challenge
        .get("target_path")
        .and_then(Value::as_str)
        .unwrap_or("/api/v0/chat/completion");
    let difficulty = challenge
        .get("difficulty")
        .and_then(Value::as_i64)
        .unwrap_or(144000);
    let expire_at = challenge
        .get("expire_at")
        .and_then(Value::as_i64)
        .unwrap_or(1_680_000_000);

    let answer = solve_pow(
        &state.engine,
        &state.module,
        challenge_str,
        salt,
        difficulty,
        expire_at,
    )?;

    let payload = json!({
        "algorithm": algorithm,
        "challenge": challenge_str,
        "salt": salt,
        "answer": answer,
        "signature": signature,
        "target_path": target_path,
    })
    .to_string();

    Ok(base64::engine::general_purpose::STANDARD.encode(payload.as_bytes()))
}

async fn forward_to_deepseek(
    state: &AppState,
    token: &str,
    req: &ChatCompletionRequest,
) -> Result<Response<Body>, ApiError> {
    let (thinking_enabled, search_enabled) = model_flags(&req.model)?;
    let final_prompt = to_prompt(&req.messages);
    let session_id = create_session(state, token).await.map_err(|e| {
        ApiError::new(
            StatusCode::BAD_GATEWAY,
            format!("create_session failed: {e}"),
        )
    })?;
    let pow = create_pow(state, token)
        .await
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, format!("pow failed: {e}")))?;

    let mut headers = base_headers(token);
    headers.insert(
        "x-ds-pow-response",
        HeaderValue::from_str(&pow)
            .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "invalid PoW header"))?,
    );

    let payload = json!({
        "chat_session_id": session_id,
        "parent_message_id": Value::Null,
        "prompt": final_prompt,
        "ref_file_ids": [],
        "thinking_enabled": thinking_enabled,
        "search_enabled": search_enabled,
    });

    let upstream = state
        .chrome_client
        .post(DEEPSEEK_COMPLETION_URL)
        .headers(headers)
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            ApiError::new(
                StatusCode::BAD_GATEWAY,
                format!("completion request failed: {e}"),
            )
        })?;

    if req.stream {
        let stream = upstream.bytes_stream().map(|chunk| {
            let evt = match chunk {
                Ok(bytes) => Event::default().data(String::from_utf8_lossy(&bytes)),
                Err(err) => Event::default().event("error").data(err.to_string()),
            };
            Ok::<Event, Infallible>(evt)
        });

        let sse = Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10)));
        return Ok(sse.into_response());
    }

    let status = upstream.status();
    let body = upstream
        .text()
        .await
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, format!("failed to read body: {e}")))?;

    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response<Body>, ApiError> {
    if req.model.trim().is_empty() || req.messages.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "request must include non-empty model and messages",
        ));
    }

    let (token, account) = resolve_deepseek_token(&state, &headers).await?;
    let result = forward_to_deepseek(&state, &token, &req).await;

    if let Some(acc) = account {
        state.release_account(acc).await;
    }

    result
}

pub async fn claude_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ClaudeRequest>,
) -> Result<Response<Body>, ApiError> {
    let mut messages = Vec::new();

    if let Some(system) = req.system.clone().filter(|v| !v.trim().is_empty()) {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: MessageContent::Text(system),
        });
    }

    for m in &req.messages {
        let content = if let Some(s) = m.content.as_str() {
            MessageContent::Text(s.to_string())
        } else {
            MessageContent::Text(m.content.to_string())
        };
        messages.push(ChatMessage {
            role: m.role.clone(),
            content,
        });
    }

    let model = if req.model.to_lowercase().contains("reason")
        || req.model.to_lowercase().contains("slow")
    {
        "deepseek-reasoner".to_string()
    } else {
        "deepseek-chat".to_string()
    };

    let chat_req = ChatCompletionRequest {
        model,
        messages,
        stream: req.stream,
    };

    let response = chat_completions(State(state), headers, Json(chat_req)).await?;

    if req.stream {
        return Ok(response);
    }

    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .map_err(|e| {
            ApiError::new(
                StatusCode::BAD_GATEWAY,
                format!("failed to parse completion body: {e}"),
            )
        })?;
    let value: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| json!({"raw": String::from_utf8_lossy(&bytes)}));
    let text = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let claude = json!({
        "id": format!("msg_{}", uuid::Uuid::new_v4()),
        "type": "message",
        "role": "assistant",
        "model": req.model,
        "content": [{"type":"text","text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": Value::Null,
    });

    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(claude.to_string()))
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}
