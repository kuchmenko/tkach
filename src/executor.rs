//! Tool dispatch: registry, policy, and executor.
//!
//! The executor is the single entry point through which the agent loop
//! invokes tools. It handles three failure modes uniformly — returning
//! a `tool_result` `Content` with `is_error: true` so the LLM can observe
//! and adapt, rather than terminating the loop:
//!
//! 1. **Policy denial** — `ToolPolicy::is_allowed` returned false.
//! 2. **Missing tool** — the LLM invoked a name that is not in the registry.
//! 3. **Tool error** — the tool itself returned `Err(ToolError)`.
//!
//! Separating dispatch from the agent loop lets sub-agents (and future
//! orchestration tools) share the same registry via [`ToolContext`].

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::message::Content;
use crate::tool::{Tool, ToolContext};

/// A single tool invocation decoded from an LLM `tool_use` block.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

/// Name-keyed collection of tools. Construction is one-shot from a
/// `Vec<Arc<dyn Tool>>` — swap the whole registry if you need to
/// reconfigure.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        let tools = tools
            .into_iter()
            .map(|t| (t.name().to_string(), t))
            .collect();
        Self { tools }
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Tool>> {
        self.tools.values()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// Decides whether the agent may invoke a given tool.
///
/// The loop treats a denial as a non-fatal `is_error: true` tool_result,
/// so the LLM can observe the block and try an alternative path. This is
/// deliberately consistent with how "tool not found" is handled —
/// guardrails that fail loudly inside the conversation are easier to
/// reason about than ones that explode outward.
pub trait ToolPolicy: Send + Sync {
    fn is_allowed(&self, tool_name: &str) -> bool;
}

/// Default policy: every tool is allowed.
pub struct AllowAll;

impl ToolPolicy for AllowAll {
    fn is_allowed(&self, _tool_name: &str) -> bool {
        true
    }
}

/// Dispatches tool calls against a registry, gated by a policy.
///
/// Cloning `Arc<ToolExecutor>` is cheap and intended: sub-agents share the
/// same executor with their parent so a nested agent automatically inherits
/// the parent's entire toolset (including `SubAgent` itself — multi-level
/// nesting comes for free up to `max_depth`).
pub struct ToolExecutor {
    registry: Arc<ToolRegistry>,
    policy: Arc<dyn ToolPolicy>,
}

impl ToolExecutor {
    pub fn new(registry: Arc<ToolRegistry>, policy: Arc<dyn ToolPolicy>) -> Self {
        Self { registry, policy }
    }

    pub fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }

    /// Execute a single tool call. Always returns a `tool_result` `Content`
    /// block — even on policy denial, missing tool, or tool error (with
    /// `is_error: true`). The loop never aborts on a tool problem; the LLM
    /// sees the error and may adapt.
    pub async fn execute_one(&self, call: ToolCall, ctx: &ToolContext) -> Content {
        if !self.policy.is_allowed(&call.name) {
            return Content::tool_result(
                &call.id,
                format!("Error: tool '{}' is not allowed by policy", call.name),
                true,
            );
        }

        let Some(tool) = self.registry.get(&call.name) else {
            return Content::tool_result(
                &call.id,
                format!("Error: tool '{}' not found", call.name),
                true,
            );
        };

        match tool.execute(call.input, ctx).await {
            Ok(output) => Content::tool_result(&call.id, output.content(), output.is_error()),
            Err(e) => Content::tool_result(&call.id, format!("Error: {e}"), true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ToolError;
    use crate::tool::{ToolClass, ToolOutput};
    use async_trait::async_trait;
    use serde_json::json;
    use std::path::PathBuf;

    struct Echo;
    #[async_trait]
    impl Tool for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echo"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        fn class(&self) -> ToolClass {
            ToolClass::ReadOnly
        }
        async fn execute(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::text(input["msg"].as_str().unwrap_or("")))
        }
    }

    fn ctx() -> ToolContext {
        ToolContext {
            working_dir: PathBuf::from("/tmp"),
            provider: Arc::new(crate::providers::Mock::with_text("")),
            model: "m".into(),
            max_turns: 1,
            max_tokens: 1,
            temperature: None,
            agent_depth: 0,
            max_agent_depth: 1,
        }
    }

    fn call(name: &str, input: Value) -> ToolCall {
        ToolCall {
            id: "id".into(),
            name: name.into(),
            input,
        }
    }

    #[tokio::test]
    async fn allow_all_runs_tool() {
        let reg = Arc::new(ToolRegistry::new(vec![Arc::new(Echo)]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));
        let res = exec
            .execute_one(call("echo", json!({"msg": "hi"})), &ctx())
            .await;
        let Content::ToolResult {
            content, is_error, ..
        } = res
        else {
            panic!("expected tool_result");
        };
        assert!(!is_error);
        assert_eq!(content, "hi");
    }

    #[tokio::test]
    async fn missing_tool_returns_error_result() {
        let reg = Arc::new(ToolRegistry::new(vec![]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));
        let res = exec.execute_one(call("ghost", json!({})), &ctx()).await;
        let Content::ToolResult {
            content, is_error, ..
        } = res
        else {
            panic!("expected tool_result");
        };
        assert!(is_error);
        assert!(content.contains("not found"));
    }

    struct DenyNamed(&'static str);
    impl ToolPolicy for DenyNamed {
        fn is_allowed(&self, name: &str) -> bool {
            name != self.0
        }
    }

    #[tokio::test]
    async fn policy_denial_returns_error_result() {
        let reg = Arc::new(ToolRegistry::new(vec![Arc::new(Echo)]));
        let exec = ToolExecutor::new(reg, Arc::new(DenyNamed("echo")));
        let res = exec
            .execute_one(call("echo", json!({"msg": "hi"})), &ctx())
            .await;
        let Content::ToolResult {
            content, is_error, ..
        } = res
        else {
            panic!("expected tool_result");
        };
        assert!(is_error);
        assert!(content.contains("not allowed"));
    }
}
