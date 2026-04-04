use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::Command;

use crate::error::ToolError;
use crate::tool::{Tool, ToolContext, ToolOutput};

/// Execute shell commands.
pub struct Bash;

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command and return its output (stdout + stderr). \
         The command runs in the agent's working directory."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Timeout in milliseconds. Default: 120000 (2 minutes)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let command = input["command"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("command is required".into()))?;
        let timeout_ms = input["timeout_ms"].as_u64().unwrap_or(120_000);

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(timeout_ms),
            Command::new("bash")
                .arg("-c")
                .arg(command)
                .current_dir(&ctx.working_dir)
                .output(),
        )
        .await;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                let mut result = String::new();
                if !stdout.is_empty() {
                    result.push_str(&stdout);
                }
                if !stderr.is_empty() {
                    if !result.is_empty() {
                        result.push('\n');
                    }
                    result.push_str("[stderr]\n");
                    result.push_str(&stderr);
                }

                if output.status.success() {
                    Ok(ToolOutput::text(if result.is_empty() {
                        "(no output)".to_string()
                    } else {
                        result
                    }))
                } else {
                    let code = output.status.code().unwrap_or(-1);
                    Ok(ToolOutput::error(format!("Exit code: {code}\n{result}")))
                }
            }
            Ok(Err(e)) => Err(ToolError::Io(e)),
            Err(_) => Ok(ToolOutput::error(format!(
                "Command timed out after {timeout_ms}ms"
            ))),
        }
    }
}
