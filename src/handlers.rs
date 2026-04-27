use anyhow::{Context, anyhow};
use axum::{
    body::Body,
    extract::{Json, State},
    http::{HeaderMap, HeaderValue, Response, StatusCode},
    response::IntoResponse,
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

    if account.token.clone().unwrap_or_default().trim().is_empty()
        && let Err(err) = login_account(state, &mut account).await
    {
        let api_error = ApiError::new(
            StatusCode::BAD_GATEWAY,
            format!("account login failed: {err}"),
        );
        state.release_account(account).await;
        return Err(api_error);
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
        state.instance_pre.as_ref(),
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
    let status = upstream.status();

    let s_state = state.clone();
    let s_token = token.to_string();
    let s_session_id = session_id.clone();

    if req.stream {
        let mut stream = upstream.bytes_stream();

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(100);

        let has_tools = req.tools.is_some() && !req.tools.as_ref().unwrap().is_empty();

        tokio::spawn(async move {
            let mut acc_text = String::new();
            while let Some(chunk_res) = stream.next().await {
                if let Ok(chunk) = chunk_res {
                    if let Ok(s) = std::str::from_utf8(&chunk) {
                        acc_text.push_str(s);

                        if let Some(done_idx) = s.find("data: [DONE]") {
                            if !has_tools {
                                let before_done = &s[..done_idx];
                                if !before_done.is_empty() {
                                    let _ = tx.send(Ok(axum::body::Bytes::from(before_done.to_string()))).await;
                                }
                            }
                            break;
                        }
                    }
                    if !has_tools {
                        let _ = tx.send(Ok(chunk)).await;
                    }
                } else {
                    if !has_tools {
                        let _ = tx.send(chunk_res.map_err(|e| std::io::Error::other(e.to_string()))).await;
                    }
                    break;
                }
            }

            if has_tools {
                let (tools, remaining) = detect_and_parse_tool_calls(&acc_text);

                // If there is remaining text, send it as a text chunk
                if !remaining.is_empty() {
                    let text_chunk = json!({
                        "id": format!("msg_{}", uuid::Uuid::new_v4()),
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": remaining},
                            "finish_reason": null
                        }]
                    });
                    let sse = format!("data: {}\n\n", text_chunk);
                    let _ = tx.send(Ok(axum::body::Bytes::from(sse))).await;
                }

                if let Some(tool_calls) = tools {
                    let tool_chunk = json!({
                        "id": format!("msg_{}", uuid::Uuid::new_v4()),
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "",
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": tool_calls},
                            "finish_reason": "tool_calls"
                        }]
                    });
                    let sse = format!("data: {}\n\n", tool_chunk);
                    let _ = tx.send(Ok(axum::body::Bytes::from(sse))).await;
                }
            } else {
                let (tools, _) = detect_and_parse_tool_calls(&acc_text);
                if let Some(tool_calls) = tools {
                    let tool_chunk = json!({
                        "id": format!("msg_{}", uuid::Uuid::new_v4()),
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "",
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": tool_calls},
                            "finish_reason": "tool_calls"
                        }]
                    });
                    let sse = format!("data: {}\n\n", tool_chunk);
                    let _ = tx.send(Ok(axum::body::Bytes::from(sse))).await;
                }
            }

            let _ = tx.send(Ok(axum::body::Bytes::from("data: [DONE]\n\n"))).await;

            tokio::spawn(delete_session(s_state, s_token, s_session_id));
        });

        let out_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        return Response::builder()
            .status(status)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("connection", "keep-alive")
            .body(Body::from_stream(out_stream))
            .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
    }

    let body_bytes = upstream
        .bytes()
        .await
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, format!("failed to read body: {e}")))?;

    let mut body = String::from_utf8_lossy(&body_bytes).into_owned();

    if let Ok(mut parsed) = serde_json::from_str::<Value>(&body)
        && let Some(content) = parsed.pointer("/choices/0/message/content").and_then(|v| v.as_str()) {
            let (tools, remaining) = detect_and_parse_tool_calls(content);
            if let Some(tool_calls) = tools {
                if let Some(message) = parsed.pointer_mut("/choices/0/message") {
                    message["content"] = Value::String(remaining);
                    message["tool_calls"] = Value::Array(tool_calls);
                }
                if let Some(choice) = parsed.pointer_mut("/choices/0") {
                    choice["finish_reason"] = Value::String("tool_calls".to_string());
                }
                body = parsed.to_string();
            }
        }

    tokio::spawn(delete_session(s_state, s_token, s_session_id));

    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

fn inject_tools(messages: &mut Vec<ChatMessage>, tools: &[Value]) {
    let mut schemas = Vec::new();
    for tool in tools {
        if let Some(func) = tool.get("function") {
            let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
            let desc = func.get("description").and_then(|v| v.as_str()).unwrap_or("No description available");
            let mut info = format!("Tool: {name}\nDescription: {desc}");

            if let Some(params) = func.get("parameters")
                && let Some(props) = params.get("properties").and_then(|v| v.as_object()) {
                    let required = params.get("required").and_then(|v| v.as_array()).map(|a| {
                        a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>()
                    }).unwrap_or_default();

                    let mut prop_strs = Vec::new();
                    for (k, v) in props {
                        let p_type = v.get("type").and_then(|t| t.as_str()).unwrap_or("string");
                        let p_desc = v.get("description").and_then(|t| t.as_str()).unwrap_or("");
                        let is_req = if required.contains(&k.as_str()) { " (required)" } else { "" };
                        prop_strs.push(format!("  - {k}: {p_type}{is_req} - {p_desc}"));
                    }
                    if !prop_strs.is_empty() {
                        info.push_str("\nParameters:\n");
                        info.push_str(&prop_strs.join("\n"));
                    }
                }
            schemas.push(info);
        }
    }

    let schema_str = schemas.join("\n\n");
    let prompt = format!("You have access to the following tools:\n\n{schema_str}\n\nWhen you need to use a tool, respond with a JSON object in this exact format:\n{{\"tool_calls\": [{{\"id\": \"call_xxx\", \"type\": \"function\", \"function\": {{\"name\": \"tool_name\", \"arguments\": \"{{\\\"param\\\": \\\"value\\\"}}\"}}}}]}}\n\nYou can call multiple tools in one response by adding more objects to the tool_calls array.\nIMPORTANT: The \"arguments\" field must be a JSON string, not a JSON object.\n\nExample:\n{{\"tool_calls\": [{{\"id\": \"call_001\", \"type\": \"function\", \"function\": {{\"name\": \"get_weather\", \"arguments\": \"{{\\\"location\\\": \\\"Beijing\\\"}}\"}}}}]}}\n\nAfter calling tools, you will receive the results and can continue the conversation.");

    if let Some(first_sys) = messages.iter_mut().find(|m| m.role == "system") {
        if let crate::models::MessageContent::Text(t) = &mut first_sys.content {
            t.push_str("\n\n");
            t.push_str(&prompt);
        }
    } else {
        messages.insert(0, ChatMessage {
            role: "system".to_string(),
            content: crate::models::MessageContent::Text(prompt),
        });
    }
}

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<ChatCompletionRequest>,
) -> Result<Response<Body>, ApiError> {
    if req.model.trim().is_empty() || req.messages.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "request must include non-empty model and messages",
        ));
    }

    if let Some(tools) = &req.tools
        && !tools.is_empty() {
            inject_tools(&mut req.messages, tools);
        }

    let (token, account) = resolve_deepseek_token(&state, &headers).await?;
    let result = forward_to_deepseek(&state, &token, &req).await;

    if let Some(acc) = account {
        state.release_account(acc).await;
    }

    result
}

fn inject_claude_tools(messages: &mut Vec<ChatMessage>, tools: &[Value]) {
    let mut schemas = Vec::new();
    for tool in tools {
        let name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
        let desc = tool.get("description").and_then(|v| v.as_str()).unwrap_or("No description available");
        let mut info = format!("Tool: {name}\nDescription: {desc}");

        if let Some(schema) = tool.get("input_schema")
            && let Some(props) = schema.get("properties").and_then(|v| v.as_object()) {
                let required = schema.get("required").and_then(|v| v.as_array()).map(|a| {
                    a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>()
                }).unwrap_or_default();

                let mut prop_strs = Vec::new();
                for (k, v) in props {
                    let p_type = v.get("type").and_then(|t| t.as_str()).unwrap_or("string");
                    let is_req = if required.contains(&k.as_str()) { " (required)" } else { "" };
                    prop_strs.push(format!("  - {k}: {p_type}{is_req}"));
                }
                if !prop_strs.is_empty() {
                    info.push_str("\nParameters:\n");
                    info.push_str(&prop_strs.join("\n"));
                }
            }
        schemas.push(info);
    }

    let schema_str = schemas.join("\n\n");
    let prompt = format!("You are Claude, a helpful AI assistant. You have access to these tools:\n\n{schema_str}\n\nWhen you need to use tools, you can call multiple tools in a single response. Use this format:\n\n{{\"tool_calls\": [\n  {{\"name\": \"tool1\", \"input\": {{\"param\": \"value\"}}}},\n  {{\"name\": \"tool2\", \"input\": {{\"param\": \"value\"}}}}\n]}}\n\nIMPORTANT: You can call multiple tools in ONE response. If you need to:\n1. Create a directory - include that in tool_calls\n2. Write a file - include that in the SAME tool_calls array\n3. Run a command - include that in the SAME tool_calls array\n\nExample of multiple tool calls in one response:\n{{\"tool_calls\": [\n  {{\"name\": \"str_replace_editor\", \"input\": {{\"command\": \"create\", \"path\": \"pp1/hello.py\", \"file_text\": \"print('Hello, World!')\"}}}},\n  {{\"name\": \"Bash\", \"input\": {{\"command\": \"python pp1/hello.py\"}}}}\n]}}\n\nExamples:\n- For TodoWrite: {{\"name\": \"TodoWrite\", \"input\": {{\"todos\": [{{\"content\": \"task\", \"status\": \"pending\", \"activeForm\": \"doing task\"}}]}}}}\n- For str_replace_editor: {{\"name\": \"str_replace_editor\", \"input\": {{\"command\": \"create\", \"path\": \"file.py\", \"file_text\": \"code\"}}}}\n- For Bash: {{\"name\": \"Bash\", \"input\": {{\"command\": \"cd /path && python file.py\"}}}}\n\nRemember: Output ONLY the JSON, no other text. The response must start with {{ and end with ]}}");

    if let Some(first_sys) = messages.iter_mut().find(|m| m.role == "system") {
        if let crate::models::MessageContent::Text(t) = &mut first_sys.content {
            t.push_str("\n\n");
            t.push_str(&prompt);
        }
    } else {
        messages.insert(0, ChatMessage {
            role: "system".to_string(),
            content: crate::models::MessageContent::Text(prompt),
        });
    }
}

pub async fn claude_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ClaudeRequest>,
) -> Result<Response<Body>, ApiError> {
    if req.stream {
        return Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "Streaming translation from DeepSeek to Claude is not yet implemented in Rust.",
        ));
    }

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

    if let Some(tools) = &req.tools
        && !tools.is_empty() {
            inject_claude_tools(&mut messages, tools);
        }

    let (fast_model, slow_model) = if let Some(mapping) = &state.config.claude_model_mapping {
        (mapping.fast.clone(), mapping.slow.clone())
    } else {
        ("deepseek-chat".to_string(), "deepseek-chat".to_string())
    };

    let model = if req.model.to_lowercase().contains("opus")
        || req.model.to_lowercase().contains("reason")
        || req.model.to_lowercase().contains("slow")
    {
        slow_model
    } else {
        fast_model
    };

    let chat_req = ChatCompletionRequest {
        model,
        messages,
        stream: req.stream,
        tools: None,
    };

    let response = chat_completions(State(state), headers, Json(chat_req)).await?;

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

    let (tools, remaining) = detect_and_parse_tool_calls(&text);

    let mut content_array = vec![json!({"type": "text", "text": remaining})];
    let stop_reason = if let Some(tool_calls) = tools {
        for (i, call) in tool_calls.iter().enumerate() {
            let name = call.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = call.get("input").cloned().unwrap_or_else(|| json!({}));

            content_array.push(json!({
                "type": "tool_use",
                "id": format!("toolu_{}", i),
                "name": name,
                "input": args
            }));
        }
        "tool_use"
    } else {
        "end_turn"
    };

    let claude = json!({
        "id": format!("msg_{}", uuid::Uuid::new_v4()),
        "type": "message",
        "role": "assistant",
        "model": req.model,
        "content": content_array,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
    });

    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(claude.to_string()))
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

pub async fn models() -> impl IntoResponse {
    let models_list = json!({
        "object": "list",
        "data": [
            {
                "id": "deepseek-chat",
                "object": "model",
                "created": 1677610602,
                "owned_by": "deepseek",
                "permission": [],
            },
            {
                "id": "deepseek-reasoner",
                "object": "model",
                "created": 1677610602,
                "owned_by": "deepseek",
                "permission": [],
            },
            {
                "id": "deepseek-chat-search",
                "object": "model",
                "created": 1677610602,
                "owned_by": "deepseek",
                "permission": [],
            },
            {
                "id": "deepseek-reasoner-search",
                "object": "model",
                "created": 1677610602,
                "owned_by": "deepseek",
                "permission": [],
            },
        ]
    });
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(models_list.to_string()))
        .unwrap()
}

pub async fn claude_models() -> impl IntoResponse {
    let models_list = json!({
        "object": "list",
        "data": [
            {
                "id": "claude-sonnet-4-20250514",
                "object": "model",
                "created": 1715635200,
                "owned_by": "anthropic",
            },
            {
                "id": "claude-sonnet-4-20250514-fast",
                "object": "model",
                "created": 1715635200,
                "owned_by": "anthropic",
            },
            {
                "id": "claude-sonnet-4-20250514-slow",
                "object": "model",
                "created": 1715635200,
                "owned_by": "anthropic",
            },
        ]
    });
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(models_list.to_string()))
        .unwrap()
}

pub async fn count_tokens(
    Json(req): Json<crate::models::CountTokensRequest>,
) -> Result<Response<Body>, ApiError> {
    if req.model.trim().is_empty() || req.messages.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "Request must include 'model' and 'messages'.",
        ));
    }

    let mut input_tokens = 0;

    if let Some(system) = req.system {
        input_tokens += system.len() / 4;
    }

    for message in &req.messages {
        input_tokens += 2; // Role token overhead
        if let Some(content) = message.content.as_str() {
            input_tokens += content.len() / 4;
        } else if let Some(content_array) = message.content.as_array() {
            for block in content_array {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    input_tokens += text.len() / 4;
                } else if let Some(content) = block.get("content").and_then(|c| c.as_str()) {
                    input_tokens += content.len() / 4;
                } else {
                    input_tokens += block.to_string().len() / 4;
                }
            }
        } else {
            input_tokens += message.content.to_string().len() / 4;
        }
    }

    if let Some(tools) = req.tools {
        for tool in tools {
            if let Some(name) = tool.get("name").and_then(|n| n.as_str()) {
                input_tokens += name.len() / 4;
            }
            if let Some(desc) = tool.get("description").and_then(|d| d.as_str()) {
                input_tokens += desc.len() / 4;
            }
            if let Some(schema) = tool.get("input_schema") {
                input_tokens += schema.to_string().len() / 4;
            }
        }
    }

    let response = json!({
        "input_tokens": std::cmp::max(1, input_tokens)
    });

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(response.to_string()))
        .unwrap())
}

pub async fn stop_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<crate::models::StopStreamRequest>,
) -> Result<Response<Body>, ApiError> {
    if req.chat_session_id.trim().is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "缺少 chat_session_id 参数",
        ));
    }

    let (token, account) = resolve_deepseek_token(&state, &headers).await?;

    let payload = json!({
        "chat_session_id": req.chat_session_id,
        "message_id": null
    });

    let upstream = state
        .safari_client
        .post("https://chat.chat.deepseek.com/api/v0/chat/stop_stream")
        .headers(base_headers(&token))
        .json(&payload)
        .send()
        .await
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, format!("stop stream failed: {e}")))?;

    let status = upstream.status();
    let body = upstream.text().await.unwrap_or_default();

    if let Some(acc) = account {
        state.release_account(acc).await;
    }

    if status.is_success() {
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(json!({"success": true, "message": "已停止流式响应"}).to_string()))
            .unwrap())
    } else {
        Ok(Response::builder()
            .status(status)
            .header("content-type", "application/json")
            .body(Body::from(json!({"success": false, "message": format!("停止失败: {body}")}).to_string()))
            .unwrap())
    }
}

pub async fn index() -> impl IntoResponse {
    let html = std::fs::read_to_string("templates/welcome.html").unwrap_or_default();
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .body(Body::from(html))
        .unwrap()
}

pub async fn delete_session(state: AppState, token: String, session_id: String) {
    let payload = json!({
        "chat_session_id": session_id
    });
    let _ = state
        .safari_client
        .post("https://chat.deepseek.com/api/v0/chat_session/delete")
        .headers(base_headers(&token))
        .json(&payload)
        .send()
        .await;
}

pub fn detect_and_parse_tool_calls(content: &str) -> (Option<Vec<Value>>, String) {
    let mut tool_calls = None;
    let mut remaining = content.to_string();

    if let Some(start_idx) = content.find("{\"tool_calls\":")
        && let Some(end_idx) = content[start_idx..].find("]}") {
            let full_end = start_idx + end_idx + 2;
            let matched = &content[start_idx..full_end];
            if let Ok(parsed) = serde_json::from_str::<Value>(matched)
                && let Some(calls) = parsed.get("tool_calls").and_then(|c| c.as_array()) {
                    tool_calls = Some(calls.clone());
                    remaining = content.replace(matched, "").trim().to_string();
                }
        }

    (tool_calls, remaining)
}
