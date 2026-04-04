use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("max turns ({0}) reached without completion")]
    MaxTurnsReached(usize),

    #[error("provider error: {0}")]
    Provider(#[from] ProviderError),

    #[error("tool '{tool_name}' failed: {source}")]
    Tool {
        tool_name: String,
        #[source]
        source: ToolError,
    },
}

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("deserialization error: {0}")]
    Deserialization(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("{0}")]
    Execution(String),
}
