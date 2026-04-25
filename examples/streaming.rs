//! Live token streaming through the Anthropic provider.
//!
//! Demonstrates `Agent::stream`: tokens arrive as `ContentDelta` events
//! and are printed immediately, while the loop also accumulates them
//! into the final `AgentResult` for history extension.
//!
//! Run with:  `ANTHROPIC_API_KEY=sk-... cargo run --example streaming`

use std::io::Write;

use agent_runtime::{Agent, CancellationToken, Message, StreamEvent, providers::Anthropic};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv_override();

    let agent = Agent::builder()
        .provider(Anthropic::from_env())
        .model("claude-haiku-4-5-20251001")
        .system("You are concise. Reply in 2-3 sentences.")
        .max_turns(1)
        .max_tokens(512)
        .build();

    let history = vec![Message::user_text(
        "In one paragraph, explain why streaming responses feel \
         faster than batched ones, even when total time is identical.",
    )];

    let mut stream = agent.stream(history, CancellationToken::new());

    print!("> ");
    std::io::stdout().flush()?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentDelta(text) => {
                print!("{text}");
                std::io::stdout().flush()?;
            }
            StreamEvent::ToolUse { name, .. } => {
                eprintln!("\n[tool: {name}]");
            }
            _ => {}
        }
    }
    println!();

    let result = stream.into_result().await?;
    eprintln!(
        "[tokens: {} in / {} out, stop: {:?}]",
        result.usage.input_tokens, result.usage.output_tokens, result.stop_reason
    );

    Ok(())
}
