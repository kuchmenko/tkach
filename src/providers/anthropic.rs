use std::collections::HashMap;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ProviderError;
use crate::message::{Content, StopReason, Usage};
use crate::provider::{LlmProvider, Request, Response};
use crate::stream::{ProviderEventStream, StreamEvent};

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
    async fn stream(&self, request: Request) -> Result<ProviderEventStream, ProviderError> {
        let mut body = build_request_body(&request);
        body.stream = true;

        let response = self
            .client
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await?;

        let status = response.status().as_u16();

        if status >= 400 {
            // Pre-stream error: same classification as complete().
            let retry_after_ms = parse_retry_after(response.headers());
            let text = response.text().await.unwrap_or_default();
            return Err(classify_error(status, text, retry_after_ms));
        }

        // Stream OK; wrap the byte stream in an SSE parser then walk our
        // own state machine over Anthropic's event taxonomy.
        let event_stream = response.bytes_stream().eventsource();
        Ok(Box::pin(anthropic_event_stream(event_stream)))
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

// --- Streaming SSE state machine ---
//
// Anthropic emits events in this lifecycle:
//
//   message_start            (carries initial input_tokens in usage)
//   content_block_start[i]   (text or tool_use; tool_use has id, name)
//   content_block_delta[i] * (text_delta or input_json_delta fragments)
//   content_block_stop[i]    (close block; for tool_use we now emit
//                             one atomic StreamEvent::ToolUse with
//                             accumulated JSON parsed)
//   ... more blocks ...
//   message_delta            (final stop_reason + final output_tokens)
//   message_stop             (terminal)
//
// `ping` and unknown event types are ignored. `error` events surface
// as `Err(ProviderError::Other)` items in the stream and terminate it.

#[derive(Deserialize)]
#[serde(tag = "type")]
enum StreamingPayload {
    #[serde(rename = "message_start")]
    MessageStart {
        #[serde(default)]
        message: MessageStartPayload,
    },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlockStart,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: BlockDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaPayload,
        #[serde(default)]
        usage: Option<ApiUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: ErrorPayload },
}

#[derive(Deserialize, Default)]
struct MessageStartPayload {
    #[serde(default)]
    usage: Option<ApiUsage>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ContentBlockStart {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum BlockDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(Deserialize)]
struct MessageDeltaPayload {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct ErrorPayload {
    #[serde(default, rename = "type")]
    kind: String,
    #[serde(default)]
    message: String,
}

/// Per-block accumulation state. Text deltas are emitted live as
/// `ContentDelta`; tool_use blocks instead buffer `partial_json`
/// fragments and emit one atomic `ToolUse` event on `content_block_stop`.
enum BlockState {
    Text,
    ToolUse {
        id: String,
        name: String,
        json_buf: String,
    },
}

fn anthropic_event_stream<S>(
    sse: S,
) -> impl futures::Stream<Item = Result<StreamEvent, ProviderError>>
where
    S: futures::Stream<
            Item = Result<
                eventsource_stream::Event,
                eventsource_stream::EventStreamError<reqwest::Error>,
            >,
        > + Send
        + Unpin
        + 'static,
{
    use std::collections::VecDeque;

    let initial: (
        S,
        HashMap<usize, BlockState>,
        VecDeque<Result<StreamEvent, ProviderError>>,
    ) = (sse, HashMap::new(), VecDeque::new());

    futures::stream::unfold(initial, |(mut sse, mut blocks, mut buffer)| async move {
        // Drain any pending events first — one SSE event can produce
        // multiple StreamEvents (e.g. message_delta → MessageDelta + Usage).
        loop {
            if let Some(ev) = buffer.pop_front() {
                return Some((ev, (sse, blocks, buffer)));
            }

            let next = sse.next().await?;
            let event = match next {
                Ok(ev) => ev,
                Err(e) => {
                    let err = ProviderError::Other(format!("SSE read error: {e}"));
                    return Some((Err(err), (sse, blocks, buffer)));
                }
            };

            // Ignore unknown event types (ping etc. that we choose not to
            // type — eventsource-stream gives us them with .event set).
            // Real Anthropic events all have payload.type matching event.event.
            let payload: StreamingPayload = match serde_json::from_str(&event.data) {
                Ok(p) => p,
                Err(_) => continue, // unknown / malformed payload, skip
            };

            process_payload(payload, &mut blocks, &mut buffer);
        }
    })
}

fn process_payload(
    payload: StreamingPayload,
    blocks: &mut HashMap<usize, BlockState>,
    buffer: &mut std::collections::VecDeque<Result<StreamEvent, ProviderError>>,
) {
    match payload {
        StreamingPayload::MessageStart { message } => {
            if let Some(usage) = message.usage {
                buffer.push_back(Ok(StreamEvent::Usage(Usage {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                })));
            }
        }
        StreamingPayload::ContentBlockStart {
            index,
            content_block,
        } => {
            let state = match content_block {
                ContentBlockStart::Text => BlockState::Text,
                ContentBlockStart::ToolUse { id, name, input } => BlockState::ToolUse {
                    id,
                    name,
                    // Some Anthropic responses ship a partial input on start;
                    // serialize it back as JSON to seed the buffer so later
                    // input_json_delta concatenation yields a valid object.
                    json_buf: if input.is_null() || input == Value::Object(Default::default()) {
                        String::new()
                    } else {
                        serde_json::to_string(&input).unwrap_or_default()
                    },
                },
            };
            blocks.insert(index, state);
        }
        StreamingPayload::ContentBlockDelta { index, delta } => match delta {
            BlockDelta::TextDelta { text } => {
                buffer.push_back(Ok(StreamEvent::ContentDelta(text)));
            }
            BlockDelta::InputJsonDelta { partial_json } => {
                if let Some(BlockState::ToolUse { json_buf, .. }) = blocks.get_mut(&index) {
                    json_buf.push_str(&partial_json);
                }
            }
        },
        StreamingPayload::ContentBlockStop { index } => {
            if let Some(BlockState::ToolUse { id, name, json_buf }) = blocks.remove(&index) {
                let input: Value = if json_buf.trim().is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_str(&json_buf)
                        .unwrap_or_else(|_| Value::String(json_buf.clone()))
                };
                buffer.push_back(Ok(StreamEvent::ToolUse { id, name, input }));
            }
        }
        StreamingPayload::MessageDelta { delta, usage } => {
            if let Some(stop) = delta.stop_reason {
                let stop_reason = map_stop_reason(&stop);
                buffer.push_back(Ok(StreamEvent::MessageDelta { stop_reason }));
            }
            if let Some(u) = usage {
                buffer.push_back(Ok(StreamEvent::Usage(Usage {
                    input_tokens: u.input_tokens,
                    output_tokens: u.output_tokens,
                })));
            }
        }
        StreamingPayload::MessageStop => {
            buffer.push_back(Ok(StreamEvent::Done));
        }
        StreamingPayload::Ping => {
            // Anthropic uses ping to keep the connection alive; nothing to emit.
        }
        StreamingPayload::Error { error } => {
            buffer.push_back(Err(ProviderError::Other(format!(
                "anthropic stream error ({}): {}",
                error.kind, error.message
            ))));
        }
    }
}

fn map_stop_reason(s: &str) -> StopReason {
    match s {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence,
        "pause_turn" => StopReason::PauseTurn,
        _ => StopReason::EndTurn,
    }
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
    /// `stream: true` switches the response to SSE; default false for
    /// `complete()`.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
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
        stream: false,
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

    let stop_reason = map_stop_reason(&api.stop_reason);

    Response {
        content,
        stop_reason,
        usage: Usage {
            input_tokens: api.usage.input_tokens,
            output_tokens: api.usage.output_tokens,
        },
    }
}
