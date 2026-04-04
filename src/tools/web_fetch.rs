use async_trait::async_trait;
use serde_json::{Value, json};

use crate::error::ToolError;
use crate::tool::{Tool, ToolContext, ToolOutput};

/// Fetch content from a URL.
pub struct WebFetch;

#[async_trait]
impl Tool for WebFetch {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch content from a URL. Returns the response body as text. \
         For HTML pages, returns raw HTML (consider using with an LLM to extract info)."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP headers as key-value pairs",
                    "additionalProperties": { "type": "string" }
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let url = input["url"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("url is required".into()))?;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| ToolError::Execution(format!("failed to create HTTP client: {e}")))?;

        let mut request = client.get(url);

        if let Some(headers) = input["headers"].as_object() {
            for (key, value) in headers {
                if let Some(v) = value.as_str() {
                    request = request.header(key, v);
                }
            }
        }

        let response = request
            .send()
            .await
            .map_err(|e| ToolError::Execution(format!("HTTP request failed: {e}")))?;

        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|e| ToolError::Execution(format!("failed to read response body: {e}")))?;

        if status >= 400 {
            Ok(ToolOutput::error(format!("HTTP {status}\n{body}")))
        } else {
            // Truncate very large responses
            let max_len = 100_000;
            if body.len() > max_len {
                Ok(ToolOutput::text(format!(
                    "{}\n\n[truncated at {max_len} chars, total: {} chars]",
                    &body[..max_len],
                    body.len()
                )))
            } else {
                Ok(ToolOutput::text(body))
            }
        }
    }
}
