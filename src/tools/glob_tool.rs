use async_trait::async_trait;
use serde_json::{Value, json};

use crate::error::ToolError;
use crate::tool::{Tool, ToolClass, ToolContext, ToolOutput};

/// Find files matching a glob pattern.
pub struct Glob;

#[async_trait]
impl Tool for Glob {
    fn name(&self) -> &str {
        "glob"
    }

    fn class(&self) -> ToolClass {
        ToolClass::ReadOnly
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern (e.g. \"**/*.rs\", \"src/**/*.ts\"). \
         Returns matching file paths sorted by modification time."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g. \"**/*.rs\")"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in. Defaults to working directory"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let pattern = input["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("pattern is required".into()))?;

        let base_dir = match input["path"].as_str() {
            Some(p) => {
                let path = std::path::Path::new(p);
                if path.is_absolute() {
                    path.to_path_buf()
                } else {
                    ctx.working_dir.join(path)
                }
            }
            None => ctx.working_dir.clone(),
        };

        let full_pattern = base_dir.join(pattern).to_string_lossy().to_string();

        // Run glob in blocking task (it's filesystem IO)
        let entries = tokio::task::spawn_blocking(move || -> Result<Vec<String>, ToolError> {
            let mut paths: Vec<_> = glob::glob(&full_pattern)
                .map_err(|e| ToolError::InvalidInput(format!("invalid glob pattern: {e}")))?
                .filter_map(|entry| entry.ok())
                .filter(|p| p.is_file())
                .collect();

            // Sort by modification time (newest first)
            paths.sort_by(|a, b| {
                let time_a = a.metadata().and_then(|m| m.modified()).ok();
                let time_b = b.metadata().and_then(|m| m.modified()).ok();
                time_b.cmp(&time_a)
            });

            Ok(paths.into_iter().map(|p| p.display().to_string()).collect())
        })
        .await
        .map_err(|e| ToolError::Execution(format!("task join error: {e}")))??;

        if entries.is_empty() {
            return Ok(ToolOutput::text("No files matched the pattern."));
        }

        let count = entries.len();
        let result = entries.join("\n");
        Ok(ToolOutput::text(format!("{result}\n\n({count} files)")))
    }
}
