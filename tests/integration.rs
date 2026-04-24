// Integration tests that call real LLM APIs.
// All tests are #[ignore] — they require ANTHROPIC_API_KEY env var.
//
// Run locally:   ANTHROPIC_API_KEY=sk-... cargo test -- --ignored
// Run in CI:     gh workflow run integration.yml -f tier=smoke

use std::path::Path;

use agent_runtime::message::{Content, Message};
use agent_runtime::providers::Anthropic;
use agent_runtime::{Agent, AgentResult, CancellationToken};

fn prompt(text: &str) -> Vec<Message> {
    vec![Message::user_text(text)]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn require_api_key() -> Anthropic {
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        panic!("ANTHROPIC_API_KEY is required for integration tests");
    }
    Anthropic::from_env()
}

fn haiku_agent(working_dir: &Path) -> Agent {
    Agent::builder()
        .provider(require_api_key())
        .model("claude-haiku-4-5-20251001")
        .system("You are a concise assistant. Use tools when needed. Be brief.")
        .tools(agent_runtime::tools::defaults())
        .max_turns(10)
        .max_tokens(1024)
        .working_dir(working_dir)
        .build()
}

fn sonnet_agent(working_dir: &Path) -> Agent {
    Agent::builder()
        .provider(require_api_key())
        .model("claude-sonnet-4-6")
        .system("You are a concise coding assistant. Use tools when needed. Be brief.")
        .tools(agent_runtime::tools::defaults())
        .tool(agent_runtime::tools::WebFetch)
        .tool(agent_runtime::tools::SubAgent)
        .max_turns(15)
        .max_tokens(4096)
        .working_dir(working_dir)
        .build()
}

fn assert_tool_called(result: &AgentResult, tool_name: &str) {
    let called = result.new_messages.iter().any(|msg| {
        msg.content
            .iter()
            .any(|c| matches!(c, Content::ToolUse { name, .. } if name == tool_name))
    });
    assert!(
        called,
        "Expected tool '{tool_name}' to be called. Tools called: {:?}",
        collect_tool_calls(result)
    );
}

fn assert_no_tool_errors(result: &AgentResult) {
    for msg in &result.new_messages {
        for content in &msg.content {
            if let Content::ToolResult {
                is_error: true,
                content,
                ..
            } = content
            {
                panic!("Unexpected tool error in conversation: {content}");
            }
        }
    }
}

fn collect_tool_calls(result: &AgentResult) -> Vec<String> {
    result
        .new_messages
        .iter()
        .flat_map(|msg| msg.content.iter())
        .filter_map(|c| match c {
            Content::ToolUse { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

fn assert_file_contains(path: &Path, expected: &str) {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    assert!(
        content.contains(expected),
        "File {} does not contain '{expected}'. Content:\n{content}",
        path.display()
    );
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir()
        .join("agent_runtime_integration")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------------------
// Tier 1: Smoke tests (haiku) — provider conversion + basic loop
// ---------------------------------------------------------------------------

/// Raw provider roundtrip: send message, get text back (no tools).
#[tokio::test]
#[ignore]
async fn smoke_provider_roundtrip() {
    let agent = Agent::builder()
        .provider(require_api_key())
        .model("claude-haiku-4-5-20251001")
        .system("Reply with exactly: PONG")
        .max_turns(1)
        .max_tokens(32)
        .build();

    let result = agent
        .run(prompt("PING"), CancellationToken::new())
        .await
        .unwrap();

    assert!(!result.text.is_empty(), "Response should not be empty");
    assert!(
        result.text.contains("PONG"),
        "Expected 'PONG' in response, got: {}",
        result.text
    );
    assert!(result.usage.input_tokens > 0, "Should have input tokens");
    assert!(result.usage.output_tokens > 0, "Should have output tokens");
}

/// Agent reads a known file using the read tool.
#[tokio::test]
#[ignore]
async fn smoke_agent_reads_file() {
    let dir = temp_dir("smoke_read");
    std::fs::write(dir.join("hello.txt"), "The secret code is 42.").unwrap();

    let agent = haiku_agent(&dir);
    let result = agent
        .run(
            prompt("Read the file hello.txt and tell me the secret code."),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "read");
    assert_no_tool_errors(&result);
    assert!(
        result.text.contains("42"),
        "Agent should mention the secret code. Got: {}",
        result.text
    );
}

/// Agent runs a bash command.
#[tokio::test]
#[ignore]
async fn smoke_agent_runs_bash() {
    let dir = temp_dir("smoke_bash");

    let agent = haiku_agent(&dir);
    let result = agent
        .run(
            prompt("Run `echo 'hello_from_bash'` and tell me what it printed."),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "bash");
    assert_no_tool_errors(&result);
    assert!(
        result.text.contains("hello_from_bash"),
        "Agent should report the command output. Got: {}",
        result.text
    );
}

/// Agent creates a new file.
#[tokio::test]
#[ignore]
async fn smoke_agent_writes_file() {
    let dir = temp_dir("smoke_write");

    let agent = haiku_agent(&dir);
    let result = agent
        .run(
            prompt("Create a file called output.txt with the content 'agent was here'."),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "write");
    assert_no_tool_errors(&result);

    let output_file = dir.join("output.txt");
    assert!(output_file.exists(), "File should have been created");
    assert_file_contains(&output_file, "agent was here");
}

/// Agent uses glob to find files.
#[tokio::test]
#[ignore]
async fn smoke_agent_finds_files() {
    let dir = temp_dir("smoke_glob");
    std::fs::write(dir.join("foo.rs"), "fn foo() {}").unwrap();
    std::fs::write(dir.join("bar.rs"), "fn bar() {}").unwrap();
    std::fs::write(dir.join("readme.md"), "# Hello").unwrap();

    let agent = haiku_agent(&dir);
    let result = agent
        .run(
            prompt("How many .rs files are in this directory? Use glob to find them."),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "glob");
    assert_no_tool_errors(&result);
    assert!(
        result.text.contains('2') || result.text.to_lowercase().contains("two"),
        "Agent should find 2 .rs files. Got: {}",
        result.text
    );
}

/// Agent uses grep to search file contents.
#[tokio::test]
#[ignore]
async fn smoke_agent_greps() {
    let dir = temp_dir("smoke_grep");
    std::fs::write(
        dir.join("code.rs"),
        "fn main() {\n    let x = TODO_FIX;\n}\n",
    )
    .unwrap();
    std::fs::write(dir.join("lib.rs"), "pub fn helper() {}\n").unwrap();

    let agent = haiku_agent(&dir);
    let result = agent
        .run(
            prompt("Search for 'TODO_FIX' in the files here. Which file contains it?"),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "grep");
    assert_no_tool_errors(&result);
    assert!(
        result.text.contains("code.rs"),
        "Agent should identify code.rs. Got: {}",
        result.text
    );
}

// ---------------------------------------------------------------------------
// Tier 2: Full tests (sonnet) — complex multi-tool scenarios
// ---------------------------------------------------------------------------

/// Agent reads a file, edits it, then the edit is verified.
#[tokio::test]
#[ignore]
async fn full_agent_edit_chain() {
    let dir = temp_dir("full_edit");
    std::fs::write(
        dir.join("config.toml"),
        "[server]\nhost = \"localhost\"\nport = 8080\n",
    )
    .unwrap();

    let agent = sonnet_agent(&dir);
    let result = agent
        .run(
            prompt("Read config.toml, then change the port from 8080 to 9090."),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "read");
    assert_tool_called(&result, "edit");
    assert_no_tool_errors(&result);
    assert_file_contains(&dir.join("config.toml"), "9090");
}

/// Agent combines glob + grep to find a pattern across files.
#[tokio::test]
#[ignore]
async fn full_agent_multi_tool_search() {
    let dir = temp_dir("full_search");
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(
        dir.join("src/main.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("src/lib.rs"),
        "pub async fn process() {\n    // async work\n}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("src/utils.rs"),
        "pub fn helper() {\n    // sync helper\n}\n",
    )
    .unwrap();

    let agent = sonnet_agent(&dir);
    let result = agent
        .run(
            prompt(
                "Use the grep tool to search for the pattern 'async' in the src/ directory. \
                 Tell me which files contain it.",
            ),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "grep");
    assert_no_tool_errors(&result);
    assert!(
        result.text.contains("lib.rs"),
        "Agent should find lib.rs. Got: {}",
        result.text
    );
    let tools_used = collect_tool_calls(&result);
    assert!(
        !tools_used.is_empty(),
        "Agent should have used at least one tool"
    );
}

/// Agent delegates to a sub-agent.
#[tokio::test]
#[ignore]
async fn full_agent_sub_agent() {
    let dir = temp_dir("full_subagent");
    std::fs::write(dir.join("data.txt"), "The answer is 7.").unwrap();

    let agent = sonnet_agent(&dir);
    let result = agent
        .run(
            prompt(
                "Use a sub-agent to read data.txt and report what it says. \
                 Pass this prompt to the agent tool: 'Read data.txt and return its contents.'",
            ),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "agent");
    assert_no_tool_errors(&result);
    assert!(
        result.text.contains('7') || result.text.contains("seven"),
        "Agent should relay the sub-agent's finding. Got: {}",
        result.text
    );
}

/// Full scenario: create a project structure and make changes.
#[tokio::test]
#[ignore]
async fn full_agent_create_and_modify() {
    let dir = temp_dir("full_create_modify");

    let agent = sonnet_agent(&dir);
    let result = agent
        .run(
            prompt(
                "Create a file called hello.py with a function greet(name) that \
                 prints 'Hello, {name}!'. Then read it back to verify it's correct.",
            ),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_tool_called(&result, "write");
    assert_tool_called(&result, "read");
    assert_no_tool_errors(&result);

    let py_file = dir.join("hello.py");
    assert!(py_file.exists(), "hello.py should have been created");
    assert_file_contains(&py_file, "greet");
    assert_file_contains(&py_file, "Hello");
}
