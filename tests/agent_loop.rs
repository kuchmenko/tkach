use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_runtime::message::{Content, StopReason, Usage};
use agent_runtime::provider::Response;
use agent_runtime::providers::Mock;
use agent_runtime::{Agent, AgentError};
use serde_json::json;

fn test_dir() -> std::path::PathBuf {
    std::env::current_dir().unwrap()
}

// --- Simple text response (no tools, 1 turn) ---

#[tokio::test]
async fn single_turn_text_response() {
    let agent = Agent::builder()
        .provider(Mock::with_text("Hello, world!"))
        .model("test")
        .working_dir(test_dir())
        .build();

    let result = agent.run("Hi").await.unwrap();

    assert_eq!(result.text, "Hello, world!");
    // 2 messages: user prompt + assistant response
    assert_eq!(result.messages.len(), 2);
}

// --- Tool call → result → final text (2 LLM calls) ---

#[tokio::test]
async fn tool_call_then_text_response() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: json!({"command": "echo hello"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => Ok(Response {
                content: vec![Content::text("The command output: hello")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(agent_runtime::tools::defaults())
        .working_dir(test_dir())
        .build();

    let result = agent.run("run echo hello").await.unwrap();

    assert_eq!(result.text, "The command output: hello");
    // user → assistant(tool_use) → user(tool_result) → assistant(text) = 4
    assert_eq!(result.messages.len(), 4);
}

// --- Multiple tool calls in a single response ---

#[tokio::test]
async fn multiple_tool_calls_single_response() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![
                    Content::ToolUse {
                        id: "t1".into(),
                        name: "bash".into(),
                        input: json!({"command": "echo first"}),
                    },
                    Content::ToolUse {
                        id: "t2".into(),
                        name: "bash".into(),
                        input: json!({"command": "echo second"}),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => Ok(Response {
                content: vec![Content::text("Both commands ran successfully")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(agent_runtime::tools::defaults())
        .working_dir(test_dir())
        .build();

    let result = agent.run("run two commands").await.unwrap();
    assert_eq!(result.text, "Both commands ran successfully");

    // Check that tool results message contains 2 results
    let tool_results_msg = &result.messages[2]; // user(tool_results)
    assert_eq!(tool_results_msg.content.len(), 2);
}

// --- Max turns exceeded ---

#[tokio::test]
async fn max_turns_exceeded() {
    // Provider always requests a tool call → infinite loop
    let mock = Mock::new(|_req| {
        Ok(Response {
            content: vec![Content::ToolUse {
                id: "t1".into(),
                name: "bash".into(),
                input: json!({"command": "echo loop"}),
            }],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        })
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(agent_runtime::tools::defaults())
        .max_turns(3)
        .working_dir(test_dir())
        .build();

    let err = agent.run("loop forever").await.unwrap_err();
    assert!(matches!(err, AgentError::MaxTurnsReached(3)));
}

// --- Tool not found ---

#[tokio::test]
async fn tool_not_found_returns_error_result() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "nonexistent_tool".into(),
                    input: json!({}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => {
                // Verify the LLM received the error in tool result
                let last_msg = req.messages.last().unwrap();
                let has_error = last_msg.content.iter().any(|c| match c {
                    Content::ToolResult { is_error, .. } => *is_error,
                    _ => false,
                });
                assert!(has_error, "LLM should receive tool error");

                Ok(Response {
                    content: vec![Content::text("Tool not found, sorry.")],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            }
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .working_dir(test_dir())
        .build();

    let result = agent.run("use a fake tool").await.unwrap();
    assert_eq!(result.text, "Tool not found, sorry.");
}

// --- Usage accumulation across turns ---

#[tokio::test]
async fn usage_accumulates_across_turns() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: json!({"command": "echo hi"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    input_tokens: 100,
                    output_tokens: 50,
                },
            }),
            _ => Ok(Response {
                content: vec![Content::text("done")],
                stop_reason: StopReason::EndTurn,
                usage: Usage {
                    input_tokens: 200,
                    output_tokens: 30,
                },
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(agent_runtime::tools::defaults())
        .working_dir(test_dir())
        .build();

    let result = agent.run("test").await.unwrap();
    assert_eq!(result.usage.input_tokens, 300);
    assert_eq!(result.usage.output_tokens, 80);
}

// --- Read tool integration ---

#[tokio::test]
async fn read_tool_reads_actual_file() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "read".into(),
                    input: json!({"file_path": "Cargo.toml"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => {
                // Verify the tool result contains Cargo.toml content
                let last_msg = req.messages.last().unwrap();
                let content = match &last_msg.content[0] {
                    Content::ToolResult { content, .. } => content.clone(),
                    _ => panic!("expected tool result"),
                };
                assert!(
                    content.contains("agent-runtime"),
                    "should contain package name"
                );

                Ok(Response {
                    content: vec![Content::text("I read the Cargo.toml")],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            }
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(agent_runtime::tools::defaults())
        .working_dir(test_dir())
        .build();

    let result = agent.run("read cargo toml").await.unwrap();
    assert_eq!(result.text, "I read the Cargo.toml");
}

// --- Glob tool integration ---

#[tokio::test]
async fn glob_tool_finds_files() {
    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();

    let mock = Mock::new(move |req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "glob".into(),
                    input: json!({"pattern": "src/**/*.rs"}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            _ => {
                let last_msg = req.messages.last().unwrap();
                let content = match &last_msg.content[0] {
                    Content::ToolResult { content, .. } => content.clone(),
                    _ => panic!("expected tool result"),
                };
                assert!(content.contains("lib.rs"), "should find lib.rs");

                Ok(Response {
                    content: vec![Content::text("Found Rust files")],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                })
            }
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(agent_runtime::tools::defaults())
        .working_dir(test_dir())
        .build();

    let result = agent.run("find rust files").await.unwrap();
    assert_eq!(result.text, "Found Rust files");
}

// --- Multi-turn tool chain (read → edit → read to verify) ---

#[tokio::test]
async fn multi_turn_tool_chain() {
    let tmp_dir = std::env::temp_dir().join("agent_runtime_test");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let test_file = tmp_dir.join("test.txt");
    std::fs::write(&test_file, "hello world").unwrap();

    let call = Arc::new(AtomicUsize::new(0));
    let call_clone = call.clone();
    let file_path = test_file.display().to_string();
    let file_path_clone = file_path.clone();

    let mock = Mock::new(move |_req| {
        let n = call_clone.fetch_add(1, Ordering::SeqCst);
        match n {
            // Turn 1: read the file
            0 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t1".into(),
                    name: "read".into(),
                    input: json!({"file_path": file_path_clone}),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            // Turn 2: edit the file
            1 => Ok(Response {
                content: vec![Content::ToolUse {
                    id: "t2".into(),
                    name: "edit".into(),
                    input: json!({
                        "file_path": file_path_clone,
                        "old_string": "hello world",
                        "new_string": "hello agent"
                    }),
                }],
                stop_reason: StopReason::ToolUse,
                usage: Usage::default(),
            }),
            // Turn 3: done
            _ => Ok(Response {
                content: vec![Content::text("File updated successfully")],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            }),
        }
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(agent_runtime::tools::defaults())
        .working_dir(&tmp_dir)
        .build();

    let result = agent.run("update the file").await.unwrap();
    assert_eq!(result.text, "File updated successfully");

    // Verify the file was actually edited
    let content = std::fs::read_to_string(&test_file).unwrap();
    assert_eq!(content, "hello agent");

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

// --- No tools configured: agent still works (text-only) ---

#[tokio::test]
async fn no_tools_text_only() {
    let agent = Agent::builder()
        .provider(Mock::with_text("I have no tools but I can chat!"))
        .model("test")
        .working_dir(test_dir())
        .build();

    let result = agent.run("hello").await.unwrap();
    assert_eq!(result.text, "I have no tools but I can chat!");
}

// --- EndTurn with tool_use content stops the loop ---

#[tokio::test]
async fn end_turn_stops_even_with_tool_use_content() {
    // Edge case: stop_reason is EndTurn but content has tool_use
    // The loop should respect stop_reason and stop.
    let mock = Mock::new(|_req| {
        Ok(Response {
            content: vec![
                Content::text("Let me think..."),
                Content::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: json!({"command": "echo orphan"}),
                },
            ],
            stop_reason: StopReason::EndTurn, // stop despite tool_use
            usage: Usage::default(),
        })
    });

    let agent = Agent::builder()
        .provider(mock)
        .model("test")
        .tools(agent_runtime::tools::defaults())
        .working_dir(test_dir())
        .build();

    let result = agent.run("test").await.unwrap();
    assert_eq!(result.text, "Let me think...");
    // Should be 2 messages: user + assistant (no tool execution)
    assert_eq!(result.messages.len(), 2);
}
