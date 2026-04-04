use std::path::Path;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::ToolError;
use crate::tool::{Tool, ToolContext, ToolOutput};

/// Write content to a file (creates or overwrites).
pub struct Write;

#[async_trait]
impl Tool for Write {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, \
         overwrites if it does. Creates parent directories as needed."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let file_path = input["file_path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("file_path is required".into()))?;
        let content = input["content"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("content is required".into()))?;

        let path = resolve_path(&ctx.working_dir, file_path);

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(ToolError::Io)?;
        }

        tokio::fs::write(&path, content)
            .await
            .map_err(ToolError::Io)?;

        let lines = content.lines().count();
        Ok(ToolOutput::text(format!(
            "Wrote {lines} lines to {}",
            path.display()
        )))
    }
}

fn resolve_path(working_dir: &Path, file_path: &str) -> std::path::PathBuf {
    let p = Path::new(file_path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        working_dir.join(p)
    }
}
