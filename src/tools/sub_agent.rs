use async_trait::async_trait;
use serde_json::{Value, json};

use crate::agent::Agent;
use crate::error::ToolError;
use crate::tool::{Tool, ToolContext, ToolOutput};

/// Spawn a sub-agent with its own system prompt and conversation.
pub struct SubAgent;

#[async_trait]
impl Tool for SubAgent {
    fn name(&self) -> &str {
        "agent"
    }

    fn description(&self) -> &str {
        "Spawn a sub-agent to handle a complex task autonomously. \
         The sub-agent gets its own conversation context, system prompt, \
         and access to file/shell tools. Use this for tasks that require \
         multi-step reasoning or focused exploration."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task for the sub-agent to perform"
                },
                "system": {
                    "type": "string",
                    "description": "System prompt for the sub-agent (optional)"
                },
                "model": {
                    "type": "string",
                    "description": "Model to use (optional, defaults to parent's model)"
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Maximum turns for the sub-agent (optional, default: 30)"
                }
            },
            "required": ["prompt"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        if ctx.agent_depth >= ctx.max_agent_depth {
            return Ok(ToolOutput::error(format!(
                "Max sub-agent depth ({}) reached. Cannot spawn further sub-agents.",
                ctx.max_agent_depth
            )));
        }

        let prompt = input["prompt"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("prompt is required".into()))?;

        let system = input["system"].as_str().map(String::from);
        let model = input["model"]
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| ctx.model.clone());
        let max_turns = input["max_turns"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(ctx.max_turns);

        let mut builder = Agent::builder()
            .provider_arc(ctx.provider.clone())
            .model(model)
            .tools(crate::tools::defaults())
            .max_turns(max_turns)
            .max_tokens(ctx.max_tokens)
            .working_dir(&ctx.working_dir)
            .agent_depth(ctx.agent_depth + 1);

        if let Some(sys) = system {
            builder = builder.system(sys);
        }

        if let Some(temp) = ctx.temperature {
            builder = builder.temperature(temp);
        }

        let agent = builder.build();

        match agent.run(prompt).await {
            Ok(result) => Ok(ToolOutput::text(result.text)),
            Err(e) => Ok(ToolOutput::error(format!("Sub-agent error: {e}"))),
        }
    }
}
