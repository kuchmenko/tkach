use agent_runtime::{Agent, providers::Anthropic, tools};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let agent = Agent::builder()
        .provider(Anthropic::from_env())
        .model("claude-sonnet-4-6-20250627")
        .system("You are a helpful coding assistant. Be concise.")
        .tools(tools::defaults())
        .build();

    let result = agent
        .run("What files are in the current directory? List them.")
        .await?;

    println!("{}", result.text);
    println!(
        "\n[tokens: {} in / {} out]",
        result.usage.input_tokens, result.usage.output_tokens
    );

    Ok(())
}
