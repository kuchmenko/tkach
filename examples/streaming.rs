//! Live token + thinking streaming through the Anthropic provider.
//!
//! Demonstrates the provider-neutral `Agent::stream` consumer shape:
//! visible answer tokens arrive as `ContentDelta` and go to stdout;
//! provider-returned thinking summaries, when the model/API emits them,
//! arrive as typed thinking events and go to stderr.
//!
//! `AgentResult.text` remains visible-answer-only. Finalized thinking
//! blocks are preserved in `AgentResult.new_messages` for replay.
//!
//! Run with:  `ANTHROPIC_API_KEY=sk-... cargo run --example streaming`

use std::io::Write;

use futures::StreamExt;
use tkach::{Agent, CancellationToken, Message, StreamEvent, providers::Anthropic};

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

    let mut thinking_delta_chars = 0usize;
    let mut thinking_blocks = 0usize;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentDelta(text) => {
                print!("{text}");
                std::io::stdout().flush()?;
            }
            StreamEvent::ThinkingDelta { text } => {
                thinking_delta_chars += text.chars().count();
                eprint!("\n[thinking] {text}");
                std::io::stderr().flush()?;
            }
            StreamEvent::ThinkingBlock { text, provider, .. } => {
                thinking_blocks += 1;
                eprintln!(
                    "\n[thinking block: {provider:?}, {} chars; metadata preserved]",
                    text.chars().count()
                );
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
        "[tokens: {} in / {} out, stop: {:?}, thinking: {} chars / {} blocks]",
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.stop_reason,
        thinking_delta_chars,
        thinking_blocks
    );

    Ok(())
}
