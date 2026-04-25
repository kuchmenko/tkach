//! OpenAI-compatible provider.
//!
//! Works against any endpoint that speaks `POST /chat/completions` in
//! OpenAI's non-streaming shape: OpenAI itself, Moonshot (Kimi), DeepSeek,
//! Ollama's OpenAI-compat layer, Together, Groq, and many more. Point
//! `base_url` at the endpoint root (without `/chat/completions`) and
//! authenticate with a bearer API key. OAuth is explicitly out of scope.
//!
//! ## Format bridge
//!
//! The OpenAI message model differs from ours in two ways that need care:
//!
//! 1. **`tool_calls.arguments` is a JSON string**, not an object. We
//!    serialize ours on the way out and parse on the way back.
//! 2. **Tool results are a separate message per call** with
//!    `role: "tool"` and `tool_call_id`. Our single `user` message
//!    carrying multiple `ToolResult` blocks fans out into N messages on
//!    the wire.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ProviderError;
use crate::message::{Content, Message, Role, StopReason, Usage};
use crate::provider::{LlmProvider, Request, Response};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

pub struct OpenAICompatible {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl OpenAICompatible {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Override the endpoint root (without trailing `/chat/completions`).
    ///
    /// Examples:
    /// - Moonshot: `https://api.moonshot.cn/v1`
    /// - DeepSeek: `https://api.deepseek.com/v1`
    /// - Ollama:   `http://localhost:11434/v1`
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Read `OPENAI_API_KEY` from the environment.
    pub fn from_env() -> Self {
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var is required");
        Self::new(api_key)
    }
}

#[async_trait]
impl LlmProvider for OpenAICompatible {
    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        let body = build_request_body(&request);
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
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

        // Read body as text first, then parse explicitly. `response.json()`
        // would map serde failures to `reqwest::Error` → `ProviderError::Http`
        // which `is_retryable()` treats as retryable — wrong for malformed
        // 2xx payloads. Persistent garbage should fail fast, not loop.
        let body = response.text().await?;
        let api_response: ApiResponse = serde_json::from_str(&body)?;
        convert_response(api_response)
    }
}

fn classify_error(status: u16, message: String, retry_after_ms: Option<u64>) -> ProviderError {
    match status {
        429 => ProviderError::RateLimit { retry_after_ms },
        503 => ProviderError::Overloaded { retry_after_ms },
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
    raw.trim().parse::<u64>().ok().map(|s| s * 1_000)
}

// --- Wire types ---

#[derive(Serialize)]
struct ApiRequest {
    model: String,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ApiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ApiMessage {
    /// system / user — simple content string.
    Simple { role: &'static str, content: String },
    /// assistant — may have text content, tool_calls, or both.
    Assistant {
        role: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ApiToolCallOut>,
    },
    /// tool — result paired with its originating tool_call_id.
    Tool {
        role: &'static str,
        tool_call_id: String,
        content: String,
    },
}

#[derive(Serialize)]
struct ApiToolCallOut {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: ApiFunctionOut,
}

#[derive(Serialize)]
struct ApiFunctionOut {
    name: String,
    /// JSON-encoded string, not a nested object — OpenAI's quirk.
    arguments: String,
}

#[derive(Serialize)]
struct ApiTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: ApiFunctionDef,
}

#[derive(Serialize)]
struct ApiFunctionDef {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Deserialize)]
struct ApiResponse {
    choices: Vec<ApiChoice>,
    #[serde(default)]
    usage: Option<ApiUsage>,
}

#[derive(Deserialize)]
struct ApiChoice {
    message: ApiResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ApiResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ApiToolCallIn>,
}

#[derive(Deserialize)]
struct ApiToolCallIn {
    id: String,
    #[serde(default)]
    function: ApiFunctionIn,
}

#[derive(Deserialize, Default)]
struct ApiFunctionIn {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize)]
struct ApiUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

// --- Conversion ---

fn build_request_body(request: &Request) -> ApiRequest {
    let mut messages: Vec<ApiMessage> = Vec::new();

    if let Some(system) = request.system.as_ref() {
        messages.push(ApiMessage::Simple {
            role: "system",
            content: system.clone(),
        });
    }

    for msg in &request.messages {
        extend_with_message(&mut messages, msg);
    }

    let tools = request
        .tools
        .iter()
        .map(|t| ApiTool {
            kind: "function",
            function: ApiFunctionDef {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect();

    ApiRequest {
        model: request.model.clone(),
        messages,
        tools,
        max_tokens: Some(request.max_tokens),
        temperature: request.temperature,
    }
}

/// Fan our single `Message` out into one-or-more OpenAI messages.
///
/// - `user` with `Text` → simple `{role: user, content}`
/// - `user` with N `ToolResult` blocks → N separate `{role: tool, ...}`
/// - `assistant` with text and/or `ToolUse` blocks → one
///   `{role: assistant, content?, tool_calls?}`
fn extend_with_message(out: &mut Vec<ApiMessage>, msg: &Message) {
    match msg.role {
        Role::User => {
            let mut text_buf = String::new();
            for c in &msg.content {
                match c {
                    Content::Text { text } => {
                        if !text_buf.is_empty() {
                            text_buf.push('\n');
                        }
                        text_buf.push_str(text);
                    }
                    Content::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        // Flush any pending user text before the tool results.
                        if !text_buf.is_empty() {
                            out.push(ApiMessage::Simple {
                                role: "user",
                                content: std::mem::take(&mut text_buf),
                            });
                        }
                        // OpenAI's `role: "tool"` schema has no is_error field;
                        // tools that returned errors would otherwise look
                        // identical to successful results to the next turn.
                        // Prefix the content with [error] so the model can
                        // disambiguate. Anthropic-via-OpenRouter strips this
                        // back out on its side; native OpenAI sees it inline.
                        let wire_content = if *is_error {
                            format!("[error] {content}")
                        } else {
                            content.clone()
                        };
                        out.push(ApiMessage::Tool {
                            role: "tool",
                            tool_call_id: tool_use_id.clone(),
                            content: wire_content,
                        });
                    }
                    Content::ToolUse { .. } => {
                        // Should not appear in a user message; skip silently.
                    }
                }
            }
            if !text_buf.is_empty() {
                out.push(ApiMessage::Simple {
                    role: "user",
                    content: text_buf,
                });
            }
        }
        Role::Assistant => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut tool_calls: Vec<ApiToolCallOut> = Vec::new();
            for c in &msg.content {
                match c {
                    Content::Text { text } => text_parts.push(text.clone()),
                    Content::ToolUse { id, name, input } => {
                        tool_calls.push(ApiToolCallOut {
                            id: id.clone(),
                            kind: "function",
                            function: ApiFunctionOut {
                                name: name.clone(),
                                // arguments is a JSON string on the wire.
                                arguments: serde_json::to_string(input)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            },
                        });
                    }
                    Content::ToolResult { .. } => {
                        // Not expected on assistant side; skip.
                    }
                }
            }
            // An empty assistant message ({"role":"assistant"}) is rejected
            // by many compat backends ("messages must have content"). This
            // can happen if the model produced a turn with no text and no
            // tool_calls — replay through the agent's stateless flow would
            // otherwise corrupt history. Skip it; the next user message
            // follows directly.
            if text_parts.is_empty() && tool_calls.is_empty() {
                return;
            }
            out.push(ApiMessage::Assistant {
                role: "assistant",
                content: if text_parts.is_empty() {
                    None
                } else {
                    Some(text_parts.join("\n"))
                },
                tool_calls,
            });
        }
    }
}

fn convert_response(api: ApiResponse) -> Result<Response, ProviderError> {
    let choice = api
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| ProviderError::Other("response had no choices".into()))?;

    let mut content: Vec<Content> = Vec::new();
    if let Some(text) = choice.message.content {
        if !text.is_empty() {
            content.push(Content::Text { text });
        }
    }
    for tc in choice.message.tool_calls {
        // arguments is a JSON string per OpenAI spec. Treat unparseable
        // arguments as an empty object rather than failing the whole
        // response — tool-side schema validation will catch abuse.
        let input = if tc.function.arguments.trim().is_empty() {
            Value::Object(Default::default())
        } else {
            serde_json::from_str(&tc.function.arguments)
                .unwrap_or(Value::Object(Default::default()))
        };
        content.push(Content::ToolUse {
            id: tc.id,
            name: tc.function.name,
            input,
        });
    }

    let has_tool_use = content.iter().any(|c| matches!(c, Content::ToolUse { .. }));

    let stop_reason = match choice.finish_reason.as_deref() {
        Some("stop") => StopReason::EndTurn,
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("content_filter") => StopReason::EndTurn,
        // `stop_sequence`-style markers aren't standard in OpenAI; map to
        // our StopSequence when we see it (some providers use this).
        Some("stop_sequence") => StopReason::StopSequence,
        // Missing or unknown finish_reason: if the response has tool_calls
        // we know the model wants to use them — defaulting to EndTurn
        // would make the agent loop terminate without invoking the tool.
        // Some compat backends omit finish_reason on tool-use turns.
        _ if has_tool_use => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    };

    let usage = api
        .usage
        .map(|u| Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
        })
        .unwrap_or_default();

    Ok(Response {
        content,
        stop_reason,
        usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_maps_system_and_user_text() {
        let req = Request {
            model: "gpt-4".into(),
            system: Some("be brief".into()),
            messages: vec![Message::user_text("hi")],
            tools: vec![],
            max_tokens: 100,
            temperature: Some(0.5),
        };
        let body = build_request_body(&req);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["model"], "gpt-4");
        assert_eq!(json["messages"][0]["role"], "system");
        assert_eq!(json["messages"][0]["content"], "be brief");
        assert_eq!(json["messages"][1]["role"], "user");
        assert_eq!(json["messages"][1]["content"], "hi");
        assert_eq!(json["temperature"], 0.5);
        assert_eq!(json["max_tokens"], 100);
    }

    #[test]
    fn request_fans_out_tool_results_to_separate_tool_messages() {
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![Message::user(vec![
                Content::tool_result("call_1", "ok", false),
                Content::tool_result("call_2", "bad", true),
            ])],
            tools: vec![],
            max_tokens: 10,
            temperature: None,
        };
        let body = build_request_body(&req);
        let json = serde_json::to_value(&body).unwrap();
        let msgs = json["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "call_1");
        assert_eq!(msgs[1]["tool_call_id"], "call_2");
    }

    #[test]
    fn request_encodes_assistant_tool_use_as_tool_calls_with_string_arguments() {
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![Message::assistant(vec![
                Content::text("let me check"),
                Content::ToolUse {
                    id: "call_x".into(),
                    name: "bash".into(),
                    input: serde_json::json!({"command": "ls"}),
                },
            ])],
            tools: vec![],
            max_tokens: 10,
            temperature: None,
        };
        let body = build_request_body(&req);
        let json = serde_json::to_value(&body).unwrap();
        let msg = &json["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["content"], "let me check");
        let tc = &msg["tool_calls"][0];
        assert_eq!(tc["id"], "call_x");
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "bash");
        // Arguments on the wire are a JSON *string*, not an object.
        let args_str = tc["function"]["arguments"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(parsed["command"], "ls");
    }

    #[test]
    fn response_decodes_text_and_tool_calls() {
        let raw = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "calling a tool",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": "{\"command\":\"echo hi\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 3 }
        });
        let api: ApiResponse = serde_json::from_value(raw).unwrap();
        let resp = convert_response(api).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 3);
        match &resp.content[0] {
            Content::Text { text } => assert_eq!(text, "calling a tool"),
            _ => panic!("expected text"),
        }
        match &resp.content[1] {
            Content::ToolUse { id, name, input } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "bash");
                assert_eq!(input["command"], "echo hi");
            }
            _ => panic!("expected tool_use"),
        }
    }

    #[test]
    fn response_maps_finish_reasons() {
        fn stop_for(reason: &str) -> StopReason {
            let raw = serde_json::json!({
                "choices": [{
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": reason
                }]
            });
            let api: ApiResponse = serde_json::from_value(raw).unwrap();
            convert_response(api).unwrap().stop_reason
        }
        assert_eq!(stop_for("stop"), StopReason::EndTurn);
        assert_eq!(stop_for("length"), StopReason::MaxTokens);
        assert_eq!(stop_for("tool_calls"), StopReason::ToolUse);
        assert_eq!(stop_for("content_filter"), StopReason::EndTurn);
    }

    #[test]
    fn classify_maps_retryable_status_codes() {
        assert!(matches!(
            classify_error(429, "".into(), Some(1000)),
            ProviderError::RateLimit {
                retry_after_ms: Some(1000)
            }
        ));
        assert!(matches!(
            classify_error(503, "".into(), None),
            ProviderError::Overloaded {
                retry_after_ms: None
            }
        ));
        assert!(matches!(
            classify_error(500, "oops".into(), None),
            ProviderError::Api {
                retryable: true,
                ..
            }
        ));
        assert!(matches!(
            classify_error(400, "bad".into(), None),
            ProviderError::Api {
                retryable: false,
                ..
            }
        ));
    }

    #[test]
    fn response_infers_tool_use_when_finish_reason_missing() {
        // Some compat backends omit `finish_reason` on tool-use turns.
        // The response carries `tool_calls` so we should still recognise
        // it as a ToolUse stop, not default to EndTurn.
        let raw = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"}
                    }]
                }
                // finish_reason intentionally absent
            }]
        });
        let api: ApiResponse = serde_json::from_value(raw).unwrap();
        let resp = convert_response(api).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn request_marks_error_tool_results_with_prefix() {
        // The OpenAI tool-message schema has no is_error field; we must
        // encode the error state in the content so the next assistant
        // turn can disambiguate success from failure.
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![Message::user(vec![
                Content::tool_result("call_ok", "all good", false),
                Content::tool_result("call_bad", "something broke", true),
            ])],
            tools: vec![],
            max_tokens: 10,
            temperature: None,
        };
        let body = build_request_body(&req);
        let json = serde_json::to_value(&body).unwrap();
        let msgs = json["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["content"], "all good");
        assert_eq!(msgs[1]["content"], "[error] something broke");
    }

    #[test]
    fn request_skips_empty_assistant_messages() {
        // An assistant turn with neither text nor tool_calls would emit
        // {"role":"assistant"} — many compat endpoints reject this on the
        // next call. The encoder should drop it instead.
        let req = Request {
            model: "m".into(),
            system: None,
            messages: vec![
                Message::user_text("hi"),
                Message::assistant(vec![]), // empty turn
                Message::user_text("still there?"),
            ],
            tools: vec![],
            max_tokens: 10,
            temperature: None,
        };
        let body = build_request_body(&req);
        let json = serde_json::to_value(&body).unwrap();
        let msgs = json["messages"].as_array().unwrap();
        // Expect: user "hi", user "still there?" — empty assistant skipped.
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "user");
    }
}
