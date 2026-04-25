# agent-runtime

A provider-independent agent runtime for Rust. Stateless agent loop, pluggable LLM providers, built-in file/shell tools, real SSE streaming, cooperative cancellation, and per-call approval gating.

[![Crates.io](https://img.shields.io/crates/v/agent-runtime.svg)](https://crates.io/crates/agent-runtime)
[![Docs.rs](https://img.shields.io/docsrs/agent-runtime)](https://docs.rs/agent-runtime)
[![CI](https://github.com/kuchmenko/agent-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/kuchmenko/agent-runtime/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Status:** pre-1.0 (`0.2.0`). Breaking changes are signalled via `feat!:` conventional commits and recorded in [`CHANGELOG.md`](./CHANGELOG.md). The core API just stabilised across three milestones — foundation, streaming, approval — and is settling, but expect motion.

## Why this exists

LLM agent runtimes tend to either (a) bake in a single provider and hide the loop, or (b) give you primitives without a working loop. This crate sits in the middle:

- **Stateless `Agent::run`** — caller owns the message history; the agent returns the **delta** of new messages it appended. Resume, multi-turn chat, fork & retry all become composable.
- **Atomic event semantics under streaming** — `ToolUse` events are emitted whole, never as partial JSON, regardless of how the upstream chunks them.
- **Sub-agents inherit the parent's executor** — one `ApprovalHandler`, one `ToolPolicy`, one tool registry gates the whole agent tree without explicit re-plumbing (Model 3).
- **Cooperative cancellation propagates** — a single `CancellationToken` shuts down the loop, the SSE pull, the in-flight HTTP body, and any `bash` child process via `kill_on_drop`.

## Quick start

```toml
[dependencies]
agent-runtime = "0.2"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use agent_runtime::{Agent, CancellationToken, Message, providers::Anthropic, tools};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::builder()
        .provider(Anthropic::from_env())
        .model("claude-haiku-4-5-20251001")
        .system("You are a concise assistant.")
        .tools(tools::defaults())
        .build();

    let mut history = vec![Message::user_text(
        "List the .rs files in this directory and summarise each.",
    )];

    let result = agent.run(history.clone(), CancellationToken::new()).await?;

    history.extend(result.new_messages);   // caller owns history
    println!("{}", result.text);
    println!("[{} in / {} out tokens]", result.usage.input_tokens, result.usage.output_tokens);
    Ok(())
}
```

## Architecture at a glance

```
┌───────────┐  messages + cancel    ┌─────────────────────────────┐
│  caller   │──────────────────────▶│         Agent::run          │
└───────────┘   new_messages,        │     (or ::stream)           │
              text, usage,           │                             │
              stop_reason            └────┬───────────────────────┘
                                          │
                       ┌──────────────────┴────────────┐
                       ▼                               ▼
                ┌────────────┐                 ┌───────────────────┐
                │  Provider  │                 │   ToolExecutor    │
                │            │                 │ ┌───────────────┐ │
                │ Anthropic  │                 │ │  ToolPolicy   │ │
                │ OpenAI-    │                 │ ├───────────────┤ │
                │ compatible │                 │ │ApprovalHandler│ │
                │ Mock       │                 │ ├───────────────┤ │
                │            │                 │ │ ToolRegistry  │ │
                └────────────┘                 │ └───────────────┘ │
                                               └─────────┬─────────┘
                                                         │
                                              read-only batches in
                                              parallel via join_all,
                                              mutating sequentially
```

## Built-in tools

| Tool        | Class      | What it does |
|-------------|------------|--------------|
| `Read`      | ReadOnly   | Read file contents (numbered lines, offset/limit) |
| `Glob`      | ReadOnly   | Find files matching a glob (sorted by mtime) |
| `Grep`      | ReadOnly   | Regex search in files (with context, ignore patterns) |
| `WebFetch`  | ReadOnly   | HTTP GET a URL, returns body text |
| `Write`     | Mutating   | Write a file (creates parents) |
| `Edit`      | Mutating   | Replace exact string in a file |
| `Bash`      | Mutating   | Run shell command (cancel-aware via `kill_on_drop`) |
| `SubAgent`  | Mutating   | Spawn a nested agent that inherits the parent's tools |

`tools::defaults()` returns `Read + Write + Edit + Glob + Grep + Bash`. Add `WebFetch` and `SubAgent::new(provider, model)` explicitly when you want them.

## Providers

```rust
use agent_runtime::providers::{Anthropic, OpenAICompatible};

// Anthropic
let p = Anthropic::from_env();   // ANTHROPIC_API_KEY

// OpenAI itself
let p = OpenAICompatible::from_env();   // OPENAI_API_KEY

// Any OpenAI-compatible endpoint:
//   OpenRouter
let p = OpenAICompatible::new(key)
    .with_base_url("https://openrouter.ai/api/v1");
//   Local Ollama
let p = OpenAICompatible::new("ignored")
    .with_base_url("http://localhost:11434/v1");
//   Moonshot, DeepSeek, Together, Groq — same shape
```

Implementing your own provider: implement `LlmProvider` (one `complete` and one `stream` method).

## Streaming

```rust
use agent_runtime::{Agent, CancellationToken, Message, StreamEvent};
use futures::StreamExt;

let mut stream = agent.stream(history, CancellationToken::new());

while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::ContentDelta(text) => {
            print!("{text}");                    // live tokens
        }
        StreamEvent::ToolUse { id, name, input } => {
            // Atomic: parser accumulated all `input_json_delta` chunks
            // before emitting; you never see partial JSON.
            eprintln!("[tool: {name}({input})]");
        }
        StreamEvent::ToolCallPending { id, name, input, class } => {
            // Agent-emitted: render an "approval pending" prompt in the UI.
            // Fires after ToolUse, before the executor's approval gate runs.
        }
        StreamEvent::Done => break,
        _ => {}                                  // MessageDelta, Usage
    }
}

let result = stream.into_result().await?;        // final AgentResult
```

Backpressure is real: a slow consumer parks the producer task, which closes the SSE read side, which lets the OS shrink the TCP receive window — all the way back to the LLM server. Cancellation works mid-stream too: `cancel.cancel()` aborts the current SSE pull within milliseconds via `tokio::select!`.

See [`examples/streaming_cancel.rs`](./examples/streaming_cancel.rs) for live cancel timing.

## Approval flow

```rust
use agent_runtime::{ApprovalDecision, ApprovalHandler, ToolClass};
use async_trait::async_trait;
use serde_json::Value;

struct MyApproval;

#[async_trait]
impl ApprovalHandler for MyApproval {
    async fn approve(&self, name: &str, input: &Value, class: ToolClass) -> ApprovalDecision {
        if class == ToolClass::ReadOnly {
            return ApprovalDecision::Allow;             // blanket-allow reads
        }
        // Hand off to UI; wait for user click.
        match prompt_user(name, input).await {
            true  => ApprovalDecision::Allow,
            false => ApprovalDecision::Deny("user declined".into()),
        }
    }
}

let agent = Agent::builder()
    .provider(Anthropic::from_env())
    .model("claude-haiku-4-5-20251001")
    .tools(tools::defaults())
    .approval(MyApproval)
    .build();
```

`Deny(reason)` flows back to the model as `is_error: true` tool_result so the LLM can adapt — it is **not** an `AgentError`. The runtime races `approve()` against `cancel.cancelled()`, so an outer cancel always wins over a hung UI handler.

## Custom tools

```rust
use agent_runtime::{Tool, ToolClass, ToolContext, ToolError, ToolOutput};
use serde_json::{Value, json};

struct CurrentTime;

#[async_trait::async_trait]
impl Tool for CurrentTime {
    fn name(&self) -> &str { "current_time" }
    fn description(&self) -> &str { "Returns the current UTC time as ISO 8601." }
    fn class(&self) -> ToolClass { ToolClass::ReadOnly }
    fn input_schema(&self) -> Value { json!({ "type": "object", "properties": {} }) }

    async fn execute(&self, _input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(chrono::Utc::now().to_rfc3339()))
    }
}

let agent = Agent::builder()
    .provider(...)
    .tool(CurrentTime)
    .build();
```

Long-running tools should `tokio::select!` on `ctx.cancel.cancelled()` and return `ToolError::Cancelled` promptly — the loop trusts the contract and does not race tools at the outer level.

## Examples

Each runnable demo also asserts its invariants — `cargo run --example NAME` either prints the demo and exits 0, or panics with a clear message.

| Example | What it shows |
|---|---|
| [`basic.rs`](./examples/basic.rs) | Minimal `agent.run` |
| [`streaming.rs`](./examples/streaming.rs) | Live token streaming |
| [`streaming_multi_tool.rs`](./examples/streaming_multi_tool.rs) | Multi-turn write→edit→read chain via `Agent::stream` |
| [`streaming_subagent.rs`](./examples/streaming_subagent.rs) | Sonnet streams, delegates to a Haiku sub-agent |
| [`streaming_openai_tools.rs`](./examples/streaming_openai_tools.rs) | OpenAI-compatible tool call (works through OpenRouter) |
| [`streaming_cancel.rs`](./examples/streaming_cancel.rs) | Cancel mid-generation, partial text preserved |
| [`streaming_resilience.rs`](./examples/streaming_resilience.rs) | Tool failure + cancel-during-tool + multi-block turns |
| [`approval_flow.rs`](./examples/approval_flow.rs) | Live denial flow with custom `ApprovalHandler` |
| [`parallel_tools.rs`](./examples/parallel_tools.rs) | Read-only tools running in parallel |
| [`custom_tool.rs`](./examples/custom_tool.rs) | Defining your own tool |

Examples that talk to live APIs read `ANTHROPIC_API_KEY` (and optionally `OPENAI_API_KEY` + `OPENAI_BASE_URL` + `OPENAI_SMOKE_MODEL`) from `.env` — see [`.env.example`](./.env.example).

## Testing

```sh
cargo test                       # unit + mock-based integration (no network)
cargo test -- --ignored          # adds real-API smoke tests (needs ANTHROPIC_API_KEY)
cargo run --example streaming    # any of the runnable examples
```

CI runs fmt, clippy (with cognitive-complexity gates), MSRV (1.86), and `cargo deny` on every PR. Real-API smoke runs are gated behind `Actions → Integration Tests → Run workflow → tier=smoke|full`.

## Versioning & releases

Conventional commits + [release-please](https://github.com/googleapis/release-please) drive the version bump and changelog. See [`RELEASING.md`](./RELEASING.md). `feat!:` commits cut a breaking-change release; pre-1.0 those bump the minor version.

## License

[MIT](./LICENSE).
