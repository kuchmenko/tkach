use std::path::Path;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::error::ToolError;
use crate::tool::{Tool, ToolClass, ToolContext, ToolOutput};

/// Search file contents using regex patterns.
pub struct Grep;

#[async_trait]
impl Tool for Grep {
    fn name(&self) -> &str {
        "grep"
    }

    fn class(&self) -> ToolClass {
        ToolClass::ReadOnly
    }

    fn description(&self) -> &str {
        "Search for a regex pattern in files. Returns matching lines with file paths \
         and line numbers. Walks directories recursively, respects common ignore patterns."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to working directory"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob filter for file names (e.g. \"*.rs\", \"*.{ts,tsx}\")"
                },
                "context": {
                    "type": "integer",
                    "description": "Number of context lines before and after each match"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matching lines to return. Default: 200"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let pattern_str = input["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("pattern is required".into()))?;
        let max_results = input["max_results"].as_u64().unwrap_or(200) as usize;
        let context_lines = input["context"].as_u64().unwrap_or(0) as usize;
        let glob_filter = input["glob"].as_str().map(String::from);

        let search_path = match input["path"].as_str() {
            Some(p) => {
                let path = Path::new(p);
                if path.is_absolute() {
                    path.to_path_buf()
                } else {
                    ctx.working_dir.join(path)
                }
            }
            None => ctx.working_dir.clone(),
        };

        let pattern_owned = pattern_str.to_string();

        let results = tokio::task::spawn_blocking(move || -> Result<Vec<String>, ToolError> {
            let re = regex::Regex::new(&pattern_owned)
                .map_err(|e| ToolError::InvalidInput(format!("invalid regex: {e}")))?;

            let glob_re = glob_filter.as_deref().map(glob_to_regex).transpose()?;

            let mut matches = Vec::new();

            if search_path.is_file() {
                search_file(&search_path, &re, context_lines, max_results, &mut matches)?;
            } else {
                for entry in walkdir::WalkDir::new(&search_path)
                    .follow_links(false)
                    .into_iter()
                    .filter_entry(|e| !is_hidden_or_ignored(e))
                    .filter_map(|e| e.ok())
                {
                    if !entry.file_type().is_file() {
                        continue;
                    }

                    if let Some(ref glob_re) = glob_re {
                        let name = entry.file_name().to_string_lossy();
                        if !glob_re.is_match(&name) {
                            continue;
                        }
                    }

                    search_file(
                        entry.path(),
                        &re,
                        context_lines,
                        max_results - matches.len(),
                        &mut matches,
                    )?;

                    if matches.len() >= max_results {
                        break;
                    }
                }
            }

            Ok(matches)
        })
        .await
        .map_err(|e| ToolError::Execution(format!("task join error: {e}")))??;

        if results.is_empty() {
            return Ok(ToolOutput::text("No matches found."));
        }

        let count = results.len();
        let output = results.join("\n");
        let truncated = if count >= max_results {
            format!("\n\n(results truncated at {max_results})")
        } else {
            String::new()
        };

        Ok(ToolOutput::text(format!("{output}{truncated}")))
    }
}

fn search_file(
    path: &Path,
    re: &regex::Regex,
    context_lines: usize,
    max: usize,
    results: &mut Vec<String>,
) -> Result<(), ToolError> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Ok(()), // Skip binary / unreadable files
    };

    let lines: Vec<&str> = content.lines().collect();
    let path_str = path.display().to_string();

    for (i, line) in lines.iter().enumerate() {
        if re.is_match(line) {
            if context_lines > 0 {
                let start = i.saturating_sub(context_lines);
                let end = (i + context_lines + 1).min(lines.len());
                for (j, line_content) in lines.iter().enumerate().take(end).skip(start) {
                    let marker = if j == i { ">" } else { " " };
                    results.push(format!("{path_str}:{}{marker} {}", j + 1, line_content));
                }
                results.push("--".to_string());
            } else {
                results.push(format!("{path_str}:{}:{}", i + 1, line));
            }

            if results.len() >= max {
                return Ok(());
            }
        }
    }

    Ok(())
}

fn is_hidden_or_ignored(entry: &walkdir::DirEntry) -> bool {
    let name = entry.file_name().to_string_lossy();
    if name.starts_with('.') {
        return true;
    }
    matches!(
        name.as_ref(),
        "node_modules" | "target" | "__pycache__" | ".git" | "dist" | "build" | "vendor"
    )
}

fn glob_to_regex(glob: &str) -> Result<regex::Regex, ToolError> {
    // Simple glob → regex conversion for file name matching
    let mut re = String::from("^");
    for c in glob.chars() {
        match c {
            '*' => re.push_str(".*"),
            '?' => re.push('.'),
            '.' => re.push_str("\\."),
            '{' => re.push('('),
            '}' => re.push(')'),
            ',' => re.push('|'),
            c => re.push(c),
        }
    }
    re.push('$');
    regex::Regex::new(&re).map_err(|e| ToolError::InvalidInput(format!("invalid glob: {e}")))
}
