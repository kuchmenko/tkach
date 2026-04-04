use std::path::PathBuf;
use std::sync::Arc;

use tracing::{debug, info, warn};

use crate::error::AgentError;
use crate::message::{Content, Message, StopReason, Usage};
use crate::provider::{LlmProvider, Request, ToolDefinition};
use crate::tool::{Tool, ToolContext};

/// Result of an agent run.
#[derive(Debug)]
pub struct AgentResult {
    /// Full conversation history.
    pub messages: Vec<Message>,
    /// Final text output from the agent.
    pub text: String,
    /// Aggregated token usage across all turns.
    pub usage: Usage,
}

/// The core agent runtime.
///
/// Runs an LLM-driven tool loop: sends messages to the LLM, executes any
/// requested tools, feeds results back, and repeats until the LLM produces
/// a final text response or max turns are reached.
pub struct Agent {
    provider: Arc<dyn LlmProvider>,
    model: String,
    system: Option<String>,
    tools: Vec<Box<dyn Tool>>,
    max_turns: usize,
    max_tokens: u32,
    temperature: Option<f32>,
    working_dir: PathBuf,
    max_agent_depth: usize,
    agent_depth: usize,
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }

    fn make_context(&self) -> ToolContext {
        ToolContext {
            working_dir: self.working_dir.clone(),
            provider: Arc::clone(&self.provider),
            model: self.model.clone(),
            max_turns: self.max_turns,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            agent_depth: self.agent_depth,
            max_agent_depth: self.max_agent_depth,
        }
    }

    fn find_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.as_ref())
    }

    /// Run the agent loop with the given prompt.
    ///
    /// Returns the final result when the LLM produces a text response
    /// without tool calls, or errors if max turns are exceeded.
    pub async fn run(&self, prompt: &str) -> Result<AgentResult, AgentError> {
        let mut messages = vec![Message::user_text(prompt)];
        let mut total_usage = Usage::default();
        let tool_defs = self.tool_definitions();
        let ctx = self.make_context();

        for turn in 0..self.max_turns {
            info!(turn, "agent turn");

            let request = Request {
                model: self.model.clone(),
                system: self.system.clone(),
                messages: messages.clone(),
                tools: tool_defs.clone(),
                max_tokens: self.max_tokens,
                temperature: self.temperature,
            };

            let response = self.provider.complete(request).await?;

            total_usage.input_tokens += response.usage.input_tokens;
            total_usage.output_tokens += response.usage.output_tokens;

            messages.push(Message::assistant(response.content.clone()));

            // Collect tool calls
            let tool_calls: Vec<_> = response
                .content
                .iter()
                .filter_map(|c| match c {
                    Content::ToolUse { id, name, input } => {
                        Some((id.clone(), name.clone(), input.clone()))
                    }
                    _ => None,
                })
                .collect();

            if tool_calls.is_empty() || response.stop_reason == StopReason::EndTurn {
                let text = response
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");

                info!(turn, "agent finished");
                return Ok(AgentResult {
                    messages,
                    text,
                    usage: total_usage,
                });
            }

            // Execute tools sequentially
            let mut results = Vec::new();
            for (id, name, input) in &tool_calls {
                match self.find_tool(name) {
                    Some(tool) => {
                        debug!(tool = name.as_str(), "executing tool");
                        match tool.execute(input.clone(), &ctx).await {
                            Ok(output) => {
                                results.push(Content::tool_result(
                                    id,
                                    output.content(),
                                    output.is_error(),
                                ));
                            }
                            Err(e) => {
                                warn!(tool = name.as_str(), error = %e, "tool failed");
                                results.push(Content::tool_result(id, format!("Error: {e}"), true));
                            }
                        }
                    }
                    None => {
                        warn!(tool = name.as_str(), "tool not found");
                        results.push(Content::tool_result(
                            id,
                            format!("Error: tool '{name}' not found"),
                            true,
                        ));
                    }
                }
            }

            messages.push(Message::user(results));
        }

        Err(AgentError::MaxTurnsReached(self.max_turns))
    }
}

// --- Builder ---

pub struct AgentBuilder {
    provider: Option<Arc<dyn LlmProvider>>,
    model: Option<String>,
    system: Option<String>,
    tools: Vec<Box<dyn Tool>>,
    max_turns: usize,
    max_tokens: u32,
    temperature: Option<f32>,
    working_dir: Option<PathBuf>,
    max_agent_depth: usize,
    agent_depth: usize,
}

impl AgentBuilder {
    fn new() -> Self {
        Self {
            provider: None,
            model: None,
            system: None,
            tools: Vec::new(),
            max_turns: 50,
            max_tokens: 16384,
            temperature: None,
            working_dir: None,
            max_agent_depth: 3,
            agent_depth: 0,
        }
    }

    pub fn provider(mut self, provider: impl LlmProvider + 'static) -> Self {
        self.provider = Some(Arc::new(provider));
        self
    }

    /// Use a shared provider (for sub-agent spawning).
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

    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    pub fn tools(mut self, tools: Vec<Box<dyn Tool>>) -> Self {
        self.tools.extend(tools);
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

    pub fn max_agent_depth(mut self, depth: usize) -> Self {
        self.max_agent_depth = depth;
        self
    }

    pub(crate) fn agent_depth(mut self, depth: usize) -> Self {
        self.agent_depth = depth;
        self
    }

    pub fn build(self) -> Agent {
        Agent {
            provider: self.provider.expect("provider is required"),
            model: self.model.expect("model is required"),
            system: self.system,
            tools: self.tools,
            max_turns: self.max_turns,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            working_dir: self
                .working_dir
                .unwrap_or_else(|| std::env::current_dir().expect("failed to get current dir")),
            max_agent_depth: self.max_agent_depth,
            agent_depth: self.agent_depth,
        }
    }
}
