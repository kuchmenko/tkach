//! # agent-runtime
//!
//! A provider-independent agent runtime for Rust with built-in tools.
//!
//! ## Quick Start
//!
//! ```ignore
//! use agent_runtime::{Agent, providers::Anthropic, tools};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let agent = Agent::builder()
//!         .provider(Anthropic::from_env())
//!         .model("claude-sonnet-4-6-20250627")
//!         .system("You are a helpful coding assistant.")
//!         .tools(tools::defaults())
//!         .build();
//!
//!     let result = agent.run("What files are in the current directory?").await?;
//!     println!("{}", result.text);
//!     Ok(())
//! }
//! ```

pub mod agent;
pub mod error;
pub mod message;
pub mod provider;
pub mod providers;
pub mod tool;
pub mod tools;

// Re-export core types at the crate root for convenience.
pub use agent::{Agent, AgentBuilder, AgentResult};
pub use error::{AgentError, ProviderError, ToolError};
pub use message::{Content, Message, Role, StopReason, Usage};
pub use provider::{LlmProvider, Request, Response, ToolDefinition};
pub use tool::{Tool, ToolContext, ToolOutput};
