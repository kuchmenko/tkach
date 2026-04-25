use std::path::PathBuf;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::error::AgentError;
use crate::executor::{AllowAll, ToolCall, ToolExecutor, ToolPolicy, ToolRegistry};
use crate::message::{Content, Message, StopReason, Usage};
use crate::provider::{LlmProvider, Request, ToolDefinition};
use crate::tool::{Tool, ToolContext};

/// Fallback `stop_reason` for partial results returned before any turn
/// has completed (e.g. provider failure on turn 0). Documented as
/// "no successful turn yet" — callers can disambiguate using the
/// outer `AgentError` variant. `EndTurn` is the least-misleading
/// concrete StopReason for the empty-history case.
const FALLBACK_STOP_REASON: StopReason = StopReason::EndTurn;

/// Result of an agent run.
///
/// The agent is stateless: it does **not** retain conversation history
/// between calls. Callers pass in the full history each time and receive
/// back only the **delta** of new messages the agent appended during this
/// run (assistant responses + tool-result user messages).
///
/// Typical consumer pattern:
///
/// ```ignore
/// let mut history = load_session();
/// history.push(Message::user_text(input));
/// let result = agent.run(history.clone(), cancel).await?;
/// history.extend(result.new_messages);
/// save_session(&history);
/// ```
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// Messages appended by this run only — **not** the full history.
    pub new_messages: Vec<Message>,
    /// Final assistant text output.
    pub text: String,
    /// Aggregated token usage across all turns in this run.
    pub usage: Usage,
    /// Stop reason from the last provider response, or `Cancelled` if the
    /// caller's `CancellationToken` fired.
    pub stop_reason: StopReason,
}

/// The core agent runtime.
///
/// Runs an LLM-driven tool loop: sends messages to the LLM, executes any
/// requested tools via the [`ToolExecutor`], feeds results back, and
/// repeats until the LLM produces a final text response, max turns are
/// reached, or the caller cancels.
pub struct Agent {
    provider: Arc<dyn LlmProvider>,
    model: String,
    system: Option<String>,
    executor: Arc<ToolExecutor>,
    max_turns: usize,
    max_tokens: u32,
    temperature: Option<f32>,
    working_dir: PathBuf,
    max_depth: usize,
    depth: usize,
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    /// Borrow the tool executor this agent was built with. Exposed so that
    /// sub-agents can share the parent's registry + policy without having
    /// to reconstruct them.
    pub fn executor(&self) -> &Arc<ToolExecutor> {
        &self.executor
    }

    /// Tool definitions sent to the LLM, sorted by name for deterministic
    /// ordering (so prompt-cache hashes stay stable across turns).
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut defs: Vec<ToolDefinition> = self
            .executor
            .registry()
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    fn make_context(&self, cancel: CancellationToken) -> ToolContext {
        ToolContext {
            working_dir: self.working_dir.clone(),
            cancel,
            depth: self.depth,
            max_depth: self.max_depth,
            executor: Arc::clone(&self.executor),
        }
    }

    /// Run the agent loop against the given message history.
    ///
    /// The agent is **stateless**: this method does not mutate `self` and
    /// the caller owns the conversation. `messages` is the full history
    /// (typically the prior session plus the new user message). The
    /// returned [`AgentResult::new_messages`] is the delta — the caller
    /// should extend their history with it to persist progress.
    ///
    /// `cancel` is a cooperative cancellation signal. The loop checks it
    /// between turns and after each tool batch, returning
    /// [`AgentError::Cancelled`] promptly. Tools receive the same token
    /// via [`ToolContext::cancel`] and are expected to honour it for any
    /// long-running work.
    ///
    /// On any error, the [`AgentError::partial`] accessor returns the
    /// progress accumulated up to the failure point, so the caller can
    /// still persist what succeeded.
    pub async fn run(
        &self,
        messages: Vec<Message>,
        cancel: CancellationToken,
    ) -> Result<AgentResult, AgentError> {
        let mut history = messages;
        let mut new_messages: Vec<Message> = Vec::new();
        let mut total_usage = Usage::default();
        // None until the first provider response lands. Using Option here
        // (rather than seeding with `EndTurn`) avoids a misleading
        // `partial.stop_reason: EndTurn` on first-turn provider failures.
        let mut last_stop: Option<StopReason> = None;

        let tool_defs = self.tool_definitions();
        let ctx = self.make_context(cancel.clone());

        for turn in 0..self.max_turns {
            info!(turn, "agent turn");

            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled {
                    partial: build_partial(&new_messages, &total_usage, StopReason::Cancelled, ""),
                });
            }

            let request = Request {
                model: self.model.clone(),
                system: self.system.clone(),
                messages: history.clone(),
                tools: tool_defs.clone(),
                max_tokens: self.max_tokens,
                temperature: self.temperature,
            };

            let response = match self.provider.complete(request).await {
                Ok(r) => r,
                Err(source) => {
                    return Err(AgentError::Provider {
                        source,
                        partial: build_partial(
                            &new_messages,
                            &total_usage,
                            last_stop.unwrap_or(FALLBACK_STOP_REASON),
                            "",
                        ),
                    });
                }
            };

            total_usage.input_tokens += response.usage.input_tokens;
            total_usage.output_tokens += response.usage.output_tokens;
            last_stop = Some(response.stop_reason);

            let assistant_msg = Message::assistant(response.content.clone());
            history.push(assistant_msg.clone());
            new_messages.push(assistant_msg);

            let tool_calls: Vec<ToolCall> = response
                .content
                .iter()
                .filter_map(|c| match c {
                    Content::ToolUse { id, name, input } => Some(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    }),
                    _ => None,
                })
                .collect();

            if tool_calls.is_empty() || response.stop_reason == StopReason::EndTurn {
                let text = extract_text(&response.content);
                info!(turn, "agent finished");
                // Safe to unwrap: we just assigned `Some` above when this
                // response was decoded.
                return Ok(AgentResult {
                    new_messages,
                    text,
                    usage: total_usage,
                    stop_reason: last_stop.unwrap_or(FALLBACK_STOP_REASON),
                });
            }

            // Cancellation can fire while `provider.complete` is in flight
            // (a multi-second LLM call). The pre-turn check at the top of
            // the loop misses this window. Bail out before invoking any
            // tools so a cancelled run never starts new mutating work.
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled {
                    partial: build_partial(&new_messages, &total_usage, StopReason::Cancelled, ""),
                });
            }

            debug!(count = tool_calls.len(), "executing tool batch");
            let results = self.executor.execute_batch(tool_calls, &ctx).await;

            let user_msg = Message::user(results);
            history.push(user_msg.clone());
            new_messages.push(user_msg);

            // If cancel fired while tools were running, skip the next
            // provider round-trip — cooperative tools have already returned
            // Cancelled error results, and another LLM call would only be
            // wasted tokens before the pre-turn check stops us anyway.
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled {
                    partial: build_partial(&new_messages, &total_usage, StopReason::Cancelled, ""),
                });
            }
        }

        Err(AgentError::MaxTurnsReached {
            turns: self.max_turns,
            partial: build_partial(
                &new_messages,
                &total_usage,
                last_stop.unwrap_or(FALLBACK_STOP_REASON),
                "",
            ),
        })
    }
}

fn build_partial(
    new_messages: &[Message],
    usage: &Usage,
    stop_reason: StopReason,
    text: &str,
) -> AgentResult {
    AgentResult {
        new_messages: new_messages.to_vec(),
        text: text.to_string(),
        usage: usage.clone(),
        stop_reason,
    }
}

fn extract_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

// --- Builder ---

pub struct AgentBuilder {
    provider: Option<Arc<dyn LlmProvider>>,
    model: Option<String>,
    system: Option<String>,
    tools: Vec<Arc<dyn Tool>>,
    policy: Option<Arc<dyn ToolPolicy>>,
    executor_override: Option<Arc<ToolExecutor>>,
    max_turns: usize,
    max_tokens: u32,
    temperature: Option<f32>,
    working_dir: Option<PathBuf>,
    max_depth: usize,
    depth: usize,
}

impl AgentBuilder {
    fn new() -> Self {
        Self {
            provider: None,
            model: None,
            system: None,
            tools: Vec::new(),
            policy: None,
            executor_override: None,
            max_turns: 50,
            max_tokens: 16384,
            temperature: None,
            working_dir: None,
            max_depth: 3,
            depth: 0,
        }
    }

    pub fn provider(mut self, provider: impl LlmProvider + 'static) -> Self {
        self.provider = Some(Arc::new(provider));
        self
    }

    /// Use a shared provider (typically for sub-agent spawning).
    pub fn provider_arc(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Register a tool by value — convenient for concrete built-in tools.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Register tools as shared trait objects. Matches the shape of
    /// [`crate::tools::defaults`] and allows one tool instance to live in
    /// multiple registries.
    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Install a tool-invocation policy. Without this, [`AllowAll`] is used.
    pub fn policy(mut self, policy: impl ToolPolicy + 'static) -> Self {
        self.policy = Some(Arc::new(policy));
        self
    }

    /// Re-use an existing [`ToolExecutor`] instead of building one from
    /// the `tools` + `policy` accumulated in the builder. Intended for
    /// sub-agent spawning, where the child inherits the parent's full
    /// registry automatically.
    ///
    /// When set, `.tool()`, `.tools()`, and `.policy()` are ignored.
    pub fn executor(mut self, executor: Arc<ToolExecutor>) -> Self {
        self.executor_override = Some(executor);
        self
    }

    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn working_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Maximum nesting depth for sub-agent recursion. Default: 3.
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub(crate) fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    pub fn build(self) -> Agent {
        let executor = self.executor_override.unwrap_or_else(|| {
            let registry = Arc::new(ToolRegistry::new(self.tools));
            let policy: Arc<dyn ToolPolicy> = self.policy.unwrap_or_else(|| Arc::new(AllowAll));
            Arc::new(ToolExecutor::new(registry, policy))
        });

        Agent {
            provider: self.provider.expect("provider is required"),
            model: self.model.expect("model is required"),
            system: self.system,
            executor,
            max_turns: self.max_turns,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            working_dir: self
                .working_dir
                .unwrap_or_else(|| std::env::current_dir().expect("failed to get current dir")),
            max_depth: self.max_depth,
            depth: self.depth,
        }
    }
}
