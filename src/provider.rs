use async_trait::async_trait;
use serde_json::Value;

use crate::error::ProviderError;
use crate::message::{Content, Message, StopReason, Usage};

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
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, request: Request) -> Result<Response, ProviderError>;
}
