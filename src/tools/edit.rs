use std::path::Path;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::error::ToolError;
use crate::tool::{Tool, ToolContext, ToolOutput};

/// Edit a file by replacing an exact string match.
pub struct Edit;

#[async_trait]
impl Tool for Edit {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing an exact string match. The old_string must \
         appear exactly once in the file (for safety). Use replace_all: true \
         to replace all occurrences."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find"
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false)"
                }
            },
            "required": ["file_path", "old_string", "new_string"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let file_path = input["file_path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("file_path is required".into()))?;
        let old_string = input["old_string"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("old_string is required".into()))?;
        let new_string = input["new_string"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("new_string is required".into()))?;
        let replace_all = input["replace_all"].as_bool().unwrap_or(false);

        let path = resolve_path(&ctx.working_dir, file_path);

        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(ToolError::Io)?;

        let count = content.matches(old_string).count();

        if count == 0 {
            return Ok(ToolOutput::error(
                "old_string not found in file. Make sure it matches exactly \
                 (including whitespace and indentation).",
            ));
        }

        if count > 1 && !replace_all {
            return Ok(ToolOutput::error(format!(
                "old_string appears {count} times in the file. \
                 Provide more context to make it unique, or set replace_all: true."
            )));
        }

        let new_content = if replace_all {
            content.replace(old_string, new_string)
        } else {
            content.replacen(old_string, new_string, 1)
        };

        tokio::fs::write(&path, &new_content)
            .await
            .map_err(ToolError::Io)?;

        Ok(ToolOutput::text(format!(
            "Replaced {count} occurrence(s) in {}",
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
