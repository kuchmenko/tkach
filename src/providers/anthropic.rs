use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ProviderError;
use crate::message::{Content, StopReason, Usage};
use crate::provider::{LlmProvider, Request, Response};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

/// Anthropic LLM provider (Claude).
pub struct Anthropic {
    api_key: String,
    client: reqwest::Client,
}

impl Anthropic {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Create from the `ANTHROPIC_API_KEY` environment variable.
    pub fn from_env() -> Self {
        let api_key =
            std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY env var is required");
        Self::new(api_key)
    }
}

#[async_trait]
impl LlmProvider for Anthropic {
    async fn stream(
        &self,
        _request: Request,
    ) -> Result<crate::stream::ProviderEventStream, ProviderError> {
        // Real SSE implementation lands in stage 1.
        Err(ProviderError::Other(
            "Anthropic streaming not yet implemented".into(),
        ))
    }

    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        let body = build_request_body(&request);

        let response = self
            .client
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status().as_u16();

        if status >= 400 {
            let retry_after_ms = parse_retry_after(response.headers());
            let text = response.text().await.unwrap_or_default();
            return Err(classify_error(status, text, retry_after_ms));
        }

        let api_response: ApiResponse = response.json().await?;
        Ok(convert_response(api_response))
    }
}

/// Classify an Anthropic API error into a [`ProviderError`].
///
/// - 429 ⇒ `RateLimit`
/// - 529, 503 ⇒ `Overloaded` (529 is Anthropic-specific; 503 is generic
///   service unavailable — both are transient server-side pressure signals)
/// - 500, 502, 504 ⇒ retryable `Api`
/// - other 5xx ⇒ retryable `Api`, other 4xx ⇒ non-retryable `Api`
fn classify_error(status: u16, message: String, retry_after_ms: Option<u64>) -> ProviderError {
    match status {
        429 => ProviderError::RateLimit { retry_after_ms },
        529 | 503 => ProviderError::Overloaded { retry_after_ms },
        500 | 502 | 504 => ProviderError::Api {
            status,
            message,
            retryable: true,
        },
        s => ProviderError::Api {
            status: s,
            message,
            retryable: (500..600).contains(&s),
        },
    }
}

fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    // Spec allows either delay-seconds (integer) or HTTP-date. We only
    // parse the integer form — OpenAI/Anthropic both use seconds in practice.
    raw.trim().parse::<u64>().ok().map(|s| s * 1_000)
}

// --- Anthropic API types ---

#[derive(Serialize)]
struct ApiRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ApiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: Vec<ApiContent>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
enum ApiContent {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

#[derive(Serialize)]
struct ApiTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Deserialize)]
struct ApiResponse {
    content: Vec<ApiContent>,
    stop_reason: String,
    usage: ApiUsage,
}

#[derive(Deserialize)]
struct ApiUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// --- Conversion ---

fn build_request_body(request: &Request) -> ApiRequest {
    let messages = request
        .messages
        .iter()
        .map(|msg| ApiMessage {
            role: match msg.role {
                crate::message::Role::User => "user".to_string(),
                crate::message::Role::Assistant => "assistant".to_string(),
            },
            content: msg.content.iter().map(content_to_api).collect(),
        })
        .collect();

    let tools = request
        .tools
        .iter()
        .map(|t| ApiTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        })
        .collect();

    ApiRequest {
        model: request.model.clone(),
        max_tokens: request.max_tokens,
        messages,
        system: request.system.clone(),
        tools,
        temperature: request.temperature,
    }
}

fn content_to_api(content: &Content) -> ApiContent {
    match content {
        Content::Text { text } => ApiContent::Text { text: text.clone() },
        Content::ToolUse { id, name, input } => ApiContent::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        },
        Content::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => ApiContent::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: content.clone(),
            is_error: *is_error,
        },
    }
}

fn convert_response(api: ApiResponse) -> Response {
    let content = api
        .content
        .into_iter()
        .map(|c| match c {
            ApiContent::Text { text } => Content::Text { text },
            ApiContent::ToolUse { id, name, input } => Content::ToolUse { id, name, input },
            ApiContent::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => Content::ToolResult {
                tool_use_id,
                content,
                is_error,
            },
        })
        .collect();

    let stop_reason = match api.stop_reason.as_str() {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence,
        "pause_turn" => StopReason::PauseTurn,
        // Unknown future values — don't guess; fall through to EndTurn.
        _ => StopReason::EndTurn,
    };

    Response {
        content,
        stop_reason,
        usage: Usage {
            input_tokens: api.usage.input_tokens,
            output_tokens: api.usage.output_tokens,
        },
    }
}
