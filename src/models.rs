use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub keys: Vec<String>,
    #[serde(default)]
    pub accounts: Vec<Account>,
    #[serde(default = "default_bind_address")]
    pub bind_address: String,
    #[serde(default = "default_cors_origins")]
    pub cors_origins: Vec<String>,
}

fn default_bind_address() -> String {
    "0.0.0.0:5001".to_string()
}

fn default_cors_origins() -> Vec<String> {
    vec![
        "http://localhost:3000".to_string(),
        "http://127.0.0.1:3000".to_string(),
    ]
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Account {
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub mobile: Option<String>,
    #[serde(default)]
    pub password: String,
    #[serde(default)]
    pub token: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<MessagePart>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct MessagePart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

impl ChatMessage {
    pub fn normalized_text(&self) -> String {
        match &self.content {
            MessageContent::Text(v) => v.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter(|p| p.kind == "text")
                .filter_map(|p| p.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClaudeRequest {
    pub model: String,
    pub messages: Vec<ClaudeMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub system: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClaudeMessage {
    pub role: String,
    pub content: Value,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: String,
}
