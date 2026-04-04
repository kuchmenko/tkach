mod bash;
mod edit;
mod glob_tool;
mod grep;
mod read;
mod sub_agent;
mod web_fetch;
mod write;

pub use bash::Bash;
pub use edit::Edit;
pub use glob_tool::Glob;
pub use grep::Grep;
pub use read::Read;
pub use sub_agent::SubAgent;
pub use web_fetch::WebFetch;
pub use write::Write;

use crate::tool::Tool;

/// Returns the default set of file-system and shell tools.
///
/// Includes: Read, Write, Edit, Glob, Grep, Bash.
/// Does NOT include SubAgent or WebFetch (add them explicitly).
pub fn defaults() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(Read),
        Box::new(Write),
        Box::new(Edit),
        Box::new(Glob),
        Box::new(Grep),
        Box::new(Bash),
    ]
}

/// Returns all available built-in tools.
pub fn all() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(Read),
        Box::new(Write),
        Box::new(Edit),
        Box::new(Glob),
        Box::new(Grep),
        Box::new(Bash),
        Box::new(SubAgent),
        Box::new(WebFetch),
    ]
}
