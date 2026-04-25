use async_trait::async_trait;
use serde_json::Value;

use crate::error::ProviderError;
use crate::message::{Content, Message, StopReason, Usage};
use crate::stream::ProviderEventStream;

/// Definition of a tool that gets sent to the LLM.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Request to the LLM provider.
pub struct Request {
    pub model: String,
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
}

/// Response from the LLM provider.
pub struct Response {
    pub content: Vec<Content>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

/// Trait for LLM providers (Anthropic, OpenAI, etc.).
///
/// Implement this trait to add support for a new LLM provider.
///
/// Two API surfaces, two transports:
///
/// - [`complete`](LlmProvider::complete) — single-shot HTTP, fully
///   buffered response. Lowest overhead when the caller wants the
///   final answer in one go.
/// - [`stream`](LlmProvider::stream) — Server-Sent Events HTTP, emits
///   incremental [`StreamEvent`](crate::stream::StreamEvent)s as the
///   model produces them.
///
/// Implementations are **independent code paths**: streaming is not
/// derived from `complete()` and vice-versa. Errors that happen
/// before the stream begins (auth, malformed request, connection
/// refused) surface from the `stream(...)` async fn itself; errors
/// that surface mid-stream (parse failures, mid-body HTTP errors)
/// arrive as `Err` items inside the stream.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, request: Request) -> Result<Response, ProviderError>;

    async fn stream(&self, request: Request) -> Result<ProviderEventStream, ProviderError>;
}
