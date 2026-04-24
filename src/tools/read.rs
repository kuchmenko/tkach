use std::path::Path;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::error::ToolError;
use crate::tool::{Tool, ToolClass, ToolContext, ToolOutput};

/// Read file contents with optional offset and limit.
pub struct Read;

#[async_trait]
impl Tool for Read {
    fn name(&self) -> &str {
        "read"
    }

    fn class(&self) -> ToolClass {
        ToolClass::ReadOnly
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Returns numbered lines. \
         Use `offset` and `limit` for large files."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based). Default: 0"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Default: 2000"
                }
            },
            "required": ["file_path"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let file_path = input["file_path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("file_path is required".into()))?;

        let offset = input["offset"].as_u64().unwrap_or(0) as usize;
        let limit = input["limit"].as_u64().unwrap_or(2000) as usize;

        let path = resolve_path(&ctx.working_dir, file_path);

        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(ToolError::Io)?;

        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();

        let selected: String = lines
            .into_iter()
            .skip(offset)
            .take(limit)
            .enumerate()
            .map(|(i, line)| format!("{}\t{}", offset + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");

        let header = if offset > 0 || total > offset + limit {
            format!(
                "[Lines {}-{} of {total}]\n",
                offset + 1,
                (offset + limit).min(total)
            )
        } else {
            String::new()
        };

        Ok(ToolOutput::text(format!("{header}{selected}")))
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
