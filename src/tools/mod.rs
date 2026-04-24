mod bash;
mod edit;
mod glob_tool;
mod grep;
mod read;
mod sub_agent;
mod web_fetch;
mod write;

use std::sync::Arc;

pub use bash::Bash;
pub use edit::Edit;
pub use glob_tool::Glob;
pub use grep::Grep;
pub use read::Read;
pub use sub_agent::SubAgent;
pub use web_fetch::WebFetch;
pub use write::Write;

use crate::tool::Tool;

/// Returns the default set of file-system and shell tools as shared
/// trait objects.
///
/// Includes: Read, Write, Edit, Glob, Grep, Bash — the stateless tools
/// that don't need provider/model configuration. Add `WebFetch` and a
/// pre-configured `SubAgent` explicitly (both require caller decisions
/// they can't guess).
pub fn defaults() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(Read),
        Arc::new(Write),
        Arc::new(Edit),
        Arc::new(Glob),
        Arc::new(Grep),
        Arc::new(Bash),
    ]
}
