use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::ToolError;
use crate::provider::LlmProvider;

/// Context passed to every tool execution.
pub struct ToolContext {
    /// Working directory for file operations.
    pub working_dir: PathBuf,

    // --- Internal fields for sub-agent support ---
    pub(crate) provider: Arc<dyn LlmProvider>,
    pub(crate) model: String,
    pub(crate) max_turns: usize,
    pub(crate) max_tokens: u32,
    pub(crate) temperature: Option<f32>,
    pub(crate) agent_depth: usize,
    pub(crate) max_agent_depth: usize,
}

/// Result of a tool execution.
pub enum ToolOutput {
    Text(String),
    Error(String),
}

impl ToolOutput {
    pub fn text(s: impl Into<String>) -> Self {
        ToolOutput::Text(s.into())
    }

    pub fn error(s: impl Into<String>) -> Self {
        ToolOutput::Error(s.into())
    }

    pub fn is_error(&self) -> bool {
        matches!(self, ToolOutput::Error(_))
    }

    pub fn content(&self) -> &str {
        match self {
            ToolOutput::Text(s) | ToolOutput::Error(s) => s,
        }
    }
}

/// Trait for tools that the agent can use.
///
/// Implement this trait to create custom tools.
///
/// # Example
///
/// ```ignore
/// use agent_runtime::{Tool, ToolContext, ToolOutput, ToolError};
/// use serde_json::{json, Value};
///
/// struct MyTool;
///
/// #[async_trait::async_trait]
/// impl Tool for MyTool {
///     fn name(&self) -> &str { "my_tool" }
///     fn description(&self) -> &str { "Does something useful" }
///     fn input_schema(&self) -> Value {
///         json!({
///             "type": "object",
///             "properties": {
///                 "query": { "type": "string" }
///             },
///             "required": ["query"]
///         })
///     }
///     async fn execute(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
///         let query = input["query"].as_str().unwrap_or_default();
///         Ok(ToolOutput::text(format!("Result for: {query}")))
///     }
/// }
/// ```
/// Side-effect class of a tool.
///
/// Used by the executor to safely parallelise consecutive read-only calls
/// in a single batch while keeping mutating calls sequential. Mutating is
/// the default because misclassifying a side-effectful tool as `ReadOnly`
/// can lead to subtle ordering bugs (two "mutating" writes racing against
/// each other); the reverse is merely a missed optimisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolClass {
    /// No observable side effects. Safe to run concurrently with other
    /// `ReadOnly` tools. Examples: `Read`, `Glob`, `Grep`, `WebFetch`.
    ReadOnly,
    /// Changes state — file system, external services, processes, or
    /// nested agents. Must run sequentially to preserve ordering.
    /// Examples: `Write`, `Edit`, `Bash`, `SubAgent`.
    Mutating,
}

#[async_trait]
pub trait Tool: Send + Sync {
    /// Unique name of the tool (used by the LLM to invoke it).
    fn name(&self) -> &str;

    /// Human-readable description of what the tool does.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's input parameters.
    fn input_schema(&self) -> Value;

    /// Side-effect class. Defaults to `Mutating` — the safe choice for
    /// tools that are not explicitly marked read-only. Override to return
    /// [`ToolClass::ReadOnly`] only when you are certain the tool has no
    /// observable side effects.
    fn class(&self) -> ToolClass {
        ToolClass::Mutating
    }

    /// Execute the tool with the given input.
    async fn execute(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError>;
}
