//! Tool dispatch: registry, policy, and executor.
//!
//! The executor is the single entry point through which the agent loop
//! invokes tools. It handles three failure modes uniformly — returning
//! a `tool_result` `Content` with `is_error: true` so the LLM can observe
//! and adapt, rather than terminating the loop:
//!
//! 1. **Policy denial** — `ToolPolicy::is_allowed` returned false.
//! 2. **Missing tool** — the LLM invoked a name that is not in the registry.
//! 3. **Tool error** — the tool itself returned `Err(ToolError)`.
//!
//! Separating dispatch from the agent loop lets sub-agents (and future
//! orchestration tools) share the same registry via [`ToolContext`].

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;
use serde_json::Value;
use tracing::warn;

use crate::approval::{ApprovalDecision, ApprovalHandler, AutoApprove};
use crate::message::Content;
use crate::tool::{Tool, ToolClass, ToolContext};

/// A single tool invocation decoded from an LLM `tool_use` block.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

/// Name-keyed collection of tools. Construction is one-shot from a
/// `Vec<Arc<dyn Tool>>` — swap the whole registry if you need to
/// reconfigure.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Build a registry from a tool list. If two tools share a `name()`,
    /// the later registration wins (consistent with `HashMap::insert`)
    /// and a `tracing::warn!` records the collision so silent shadowing
    /// — e.g. a custom tool accidentally masking a built-in — is at
    /// least visible in logs.
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        let mut map: HashMap<String, Arc<dyn Tool>> = HashMap::with_capacity(tools.len());
        for t in tools {
            let name = t.name().to_string();
            if map.insert(name.clone(), t).is_some() {
                warn!(
                    tool = %name,
                    "duplicate tool name in registry; later registration overrode earlier"
                );
            }
        }
        Self { tools: map }
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Tool>> {
        self.tools.values()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// Decides whether the agent may invoke a given tool.
///
/// The loop treats a denial as a non-fatal `is_error: true` tool_result,
/// so the LLM can observe the block and try an alternative path. This is
/// deliberately consistent with how "tool not found" is handled —
/// guardrails that fail loudly inside the conversation are easier to
/// reason about than ones that explode outward.
pub trait ToolPolicy: Send + Sync {
    fn is_allowed(&self, tool_name: &str) -> bool;
}

/// Default policy: every tool is allowed.
pub struct AllowAll;

impl ToolPolicy for AllowAll {
    fn is_allowed(&self, _tool_name: &str) -> bool {
        true
    }
}

/// Dispatches tool calls against a registry, gated by a policy and an
/// approval handler.
///
/// Two gates run before every tool invocation:
///
/// 1. [`ToolPolicy::is_allowed`] — *static* gate. Synchronous, no UI
///    interaction; decides whether the tool may run at all based on
///    its name. Denial here surfaces as `is_error: true` tool_result.
/// 2. [`ApprovalHandler::approve`] — *dynamic* gate. Async, may block
///    on a UI prompt. Decides whether *this specific call* with
///    *these specific arguments* may run. Denial also surfaces as
///    `is_error: true` tool_result so the model can adapt.
///
/// The approval call is raced against `ctx.cancel.cancelled()`, so an
/// outer cancel always wins over a hung UI.
///
/// Cloning `Arc<ToolExecutor>` is cheap and intended: sub-agents share
/// the same executor with their parent so nested agents automatically
/// inherit the same registry, policy, AND approval handler (Model 3).
pub struct ToolExecutor {
    registry: Arc<ToolRegistry>,
    policy: Arc<dyn ToolPolicy>,
    approval: Arc<dyn ApprovalHandler>,
}

impl ToolExecutor {
    /// Construct an executor with the default `AutoApprove` handler.
    /// Backwards-compatible with pre-#6 callers — same behaviour.
    pub fn new(registry: Arc<ToolRegistry>, policy: Arc<dyn ToolPolicy>) -> Self {
        Self {
            registry,
            policy,
            approval: Arc::new(AutoApprove),
        }
    }

    /// Construct an executor with an explicit approval handler.
    /// Used by `AgentBuilder::approval(...)`; consumers can also
    /// construct directly when bypassing the builder.
    pub fn with_approval(
        registry: Arc<ToolRegistry>,
        policy: Arc<dyn ToolPolicy>,
        approval: Arc<dyn ApprovalHandler>,
    ) -> Self {
        Self {
            registry,
            policy,
            approval,
        }
    }

    pub fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }

    /// Execute a single tool call. Always returns a `tool_result` `Content`
    /// block — even on policy denial, approval denial, missing tool, or
    /// tool error (with `is_error: true`). The loop never aborts on a
    /// tool problem; the LLM sees the error and may adapt.
    pub async fn execute_one(&self, call: ToolCall, ctx: &ToolContext) -> Content {
        if !self.policy.is_allowed(&call.name) {
            return Content::tool_result(
                &call.id,
                format!("Error: tool '{}' is not allowed by policy", call.name),
                true,
            );
        }

        let Some(tool) = self.registry.get(&call.name) else {
            return Content::tool_result(
                &call.id,
                format!("Error: tool '{}' not found", call.name),
                true,
            );
        };

        // Dynamic gate: ask the approval handler. Race against the
        // outer cancellation token so a hung UI cannot deadlock the
        // agent indefinitely — `cancel.cancel()` always wins.
        let class = tool.class();
        let decision = tokio::select! {
            biased;
            _ = ctx.cancel.cancelled() => {
                return Content::tool_result(
                    &call.id,
                    "Error: cancelled while awaiting approval",
                    true,
                );
            }
            d = self.approval.approve(&call.name, &call.input, class) => d,
        };
        if let ApprovalDecision::Deny(reason) = decision {
            return Content::tool_result(
                &call.id,
                format!("Error: approval denied — {reason}"),
                true,
            );
        }

        match tool.execute(call.input, ctx).await {
            Ok(output) => Content::tool_result(&call.id, output.content(), output.is_error()),
            Err(e) => Content::tool_result(&call.id, format!("Error: {e}"), true),
        }
    }

    /// Execute a batch of tool calls in the LLM-issued order.
    ///
    /// Consecutive [`ToolClass::ReadOnly`] calls run concurrently via
    /// `join_all`; [`ToolClass::Mutating`] calls run strictly sequentially.
    /// Results are returned in the **original input order** regardless of
    /// when the concurrent futures actually resolve. This means:
    ///
    /// ```text
    ///   input:  [Read A, Read B, Write X, Read C, Glob D]
    ///   runs :  [ RO,       RO ] [ Mut ] [ RO,      RO ]
    ///           └─ join_all ─┘  └─ seq ─┘└─ join_all ─┘
    ///   out  :  [ res(A), res(B), res(X), res(C), res(D) ]
    /// ```
    ///
    /// Policy denial and missing-tool errors classify as `Mutating` so
    /// they serialise trivially as error `tool_result` content without
    /// affecting surrounding batches.
    ///
    /// Cancellation: before starting each new run (RO batch or single Mut
    /// call), the executor checks `ctx.cancel.is_cancelled()`. If cancel
    /// has fired, remaining calls receive a synthetic `is_error: true`
    /// `tool_result` with body `Error: cancelled before execution` and
    /// the batch returns. This preserves the 1:1 tool_use → tool_result
    /// invariant the agent loop relies on, and stops mutating tools from
    /// running after a cancel that arrived mid-batch.
    pub async fn execute_batch(&self, calls: Vec<ToolCall>, ctx: &ToolContext) -> Vec<Content> {
        if calls.is_empty() {
            return Vec::new();
        }

        let classes: Vec<ToolClass> = calls.iter().map(|c| self.classify(c)).collect();
        let mut results: Vec<Content> = Vec::with_capacity(calls.len());
        let mut calls_iter = calls.into_iter();
        let mut i = 0;

        while i < classes.len() {
            // Don't START new tools after cancel — already-running tools
            // in this batch will themselves observe ctx.cancel through
            // their cooperative select! and return Cancelled errors. New
            // tools must not begin work after the caller has aborted.
            if ctx.cancel.is_cancelled() {
                for call in calls_iter {
                    results.push(Content::tool_result(
                        &call.id,
                        "Error: cancelled before execution".to_string(),
                        true,
                    ));
                }
                return results;
            }

            let start = i;
            if classes[i] == ToolClass::ReadOnly {
                while i < classes.len() && classes[i] == ToolClass::ReadOnly {
                    i += 1;
                }
                let run: Vec<ToolCall> = (&mut calls_iter).take(i - start).collect();
                let batch = join_all(run.into_iter().map(|c| self.execute_one(c, ctx))).await;
                results.extend(batch);
            } else {
                let call = calls_iter.next().expect("classes and calls in lockstep");
                let res = self.execute_one(call, ctx).await;
                results.push(res);
                i += 1;
            }
        }

        results
    }

    fn classify(&self, call: &ToolCall) -> ToolClass {
        // Denied or missing tools are classified as Mutating so they execute
        // sequentially. Since they produce error results instantly, this has
        // no performance impact; it just keeps the partition logic simple.
        if !self.policy.is_allowed(&call.name) {
            return ToolClass::Mutating;
        }
        self.registry
            .get(&call.name)
            .map(|t| t.class())
            .unwrap_or(ToolClass::Mutating)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ToolError;
    use crate::tool::{ToolClass, ToolOutput};
    use async_trait::async_trait;
    use serde_json::json;
    use std::path::PathBuf;

    struct Echo;
    #[async_trait]
    impl Tool for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echo"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        fn class(&self) -> ToolClass {
            ToolClass::ReadOnly
        }
        async fn execute(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::text(input["msg"].as_str().unwrap_or("")))
        }
    }

    fn empty_executor() -> Arc<ToolExecutor> {
        Arc::new(ToolExecutor::new(
            Arc::new(ToolRegistry::new(vec![])),
            Arc::new(AllowAll),
        ))
    }

    fn ctx() -> ToolContext {
        ToolContext {
            working_dir: PathBuf::from("/tmp"),
            cancel: tokio_util::sync::CancellationToken::new(),
            depth: 0,
            max_depth: 1,
            executor: empty_executor(),
        }
    }

    fn call(name: &str, input: Value) -> ToolCall {
        ToolCall {
            id: "id".into(),
            name: name.into(),
            input,
        }
    }

    #[tokio::test]
    async fn allow_all_runs_tool() {
        let reg = Arc::new(ToolRegistry::new(vec![Arc::new(Echo)]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));
        let res = exec
            .execute_one(call("echo", json!({"msg": "hi"})), &ctx())
            .await;
        let Content::ToolResult {
            content, is_error, ..
        } = res
        else {
            panic!("expected tool_result");
        };
        assert!(!is_error);
        assert_eq!(content, "hi");
    }

    #[tokio::test]
    async fn missing_tool_returns_error_result() {
        let reg = Arc::new(ToolRegistry::new(vec![]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));
        let res = exec.execute_one(call("ghost", json!({})), &ctx()).await;
        let Content::ToolResult {
            content, is_error, ..
        } = res
        else {
            panic!("expected tool_result");
        };
        assert!(is_error);
        assert!(content.contains("not found"));
    }

    struct DenyNamed(&'static str);
    impl ToolPolicy for DenyNamed {
        fn is_allowed(&self, name: &str) -> bool {
            name != self.0
        }
    }

    #[tokio::test]
    async fn policy_denial_returns_error_result() {
        let reg = Arc::new(ToolRegistry::new(vec![Arc::new(Echo)]));
        let exec = ToolExecutor::new(reg, Arc::new(DenyNamed("echo")));
        let res = exec
            .execute_one(call("echo", json!({"msg": "hi"})), &ctx())
            .await;
        let Content::ToolResult {
            content, is_error, ..
        } = res
        else {
            panic!("expected tool_result");
        };
        assert!(is_error);
        assert!(content.contains("not allowed"));
    }

    /// A read-only tool that sleeps for `delay_ms` then echoes its label.
    struct SlowRO {
        label: String,
    }
    #[async_trait]
    impl Tool for SlowRO {
        fn name(&self) -> &str {
            &self.label
        }
        fn description(&self) -> &str {
            "slow"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        fn class(&self) -> ToolClass {
            ToolClass::ReadOnly
        }
        async fn execute(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
            let delay_ms = input["delay_ms"].as_u64().unwrap_or(0);
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            Ok(ToolOutput::text(self.label.clone()))
        }
    }

    /// A mutating tool that records its invocation order in a shared counter.
    struct OrderingMut {
        label: String,
    }
    #[async_trait]
    impl Tool for OrderingMut {
        fn name(&self) -> &str {
            &self.label
        }
        fn description(&self) -> &str {
            "mut"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        // class() defaults to Mutating.
        async fn execute(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::text(self.label.clone()))
        }
    }

    fn extract_text(c: &Content) -> &str {
        match c {
            Content::ToolResult { content, .. } => content.as_str(),
            _ => panic!("expected tool_result"),
        }
    }

    #[tokio::test]
    async fn batch_preserves_order_despite_parallel_ro() {
        // Put a slow RO tool BEFORE a fast RO tool. If RO runs are truly
        // parallel and results are not re-ordered, "b" finishes first but
        // appears second in the output.
        let reg = Arc::new(ToolRegistry::new(vec![
            Arc::new(SlowRO { label: "a".into() }),
            Arc::new(SlowRO { label: "b".into() }),
        ]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));
        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "a".into(),
                input: json!({"delay_ms": 50}),
            },
            ToolCall {
                id: "2".into(),
                name: "b".into(),
                input: json!({"delay_ms": 0}),
            },
        ];

        let start = std::time::Instant::now();
        let results = exec.execute_batch(calls, &ctx()).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 2);
        assert_eq!(extract_text(&results[0]), "a"); // original order kept
        assert_eq!(extract_text(&results[1]), "b");
        // If RO ran sequentially, elapsed would be ≥ 50ms. Parallel ⇒ ~50ms.
        // Sequential would be ~50ms + 0ms = 50ms too, so timing alone is not
        // a strict test; ordering IS the invariant we care about. Keep the
        // timing check loose — just assert it didn't balloon to double.
        assert!(
            elapsed < std::time::Duration::from_millis(150),
            "unexpected slowdown: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn batch_partitions_ro_and_mut_runs() {
        // Pattern: [RO a, RO b, MUT m, RO c] — two RO runs bracketing a
        // Mut call. Each run must execute, results must be in input order.
        let reg = Arc::new(ToolRegistry::new(vec![
            Arc::new(SlowRO { label: "a".into() }),
            Arc::new(SlowRO { label: "b".into() }),
            Arc::new(OrderingMut { label: "m".into() }),
            Arc::new(SlowRO { label: "c".into() }),
        ]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));
        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "a".into(),
                input: json!({"delay_ms": 10}),
            },
            ToolCall {
                id: "2".into(),
                name: "b".into(),
                input: json!({"delay_ms": 10}),
            },
            ToolCall {
                id: "3".into(),
                name: "m".into(),
                input: json!({}),
            },
            ToolCall {
                id: "4".into(),
                name: "c".into(),
                input: json!({"delay_ms": 10}),
            },
        ];

        let results = exec.execute_batch(calls, &ctx()).await;
        assert_eq!(results.len(), 4);
        assert_eq!(extract_text(&results[0]), "a");
        assert_eq!(extract_text(&results[1]), "b");
        assert_eq!(extract_text(&results[2]), "m");
        assert_eq!(extract_text(&results[3]), "c");
    }

    /// A mutating tool that toggles a shared flag — used to detect whether
    /// a tool ran. If `execute_batch` correctly stops dispatching after
    /// cancel fires, the second mutating tool's flag stays unset.
    struct FlagSetter(Arc<std::sync::atomic::AtomicBool>, &'static str);
    #[async_trait]
    impl Tool for FlagSetter {
        fn name(&self) -> &str {
            self.1
        }
        fn description(&self) -> &str {
            "flag"
        }
        fn input_schema(&self) -> Value {
            json!({})
        }
        async fn execute(
            &self,
            _input: Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(ToolOutput::text("ran"))
        }
    }

    #[tokio::test]
    async fn batch_stops_dispatching_after_cancel() {
        let m1_ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let m2_ran = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let reg = Arc::new(ToolRegistry::new(vec![
            Arc::new(FlagSetter(Arc::clone(&m1_ran), "m1")),
            Arc::new(FlagSetter(Arc::clone(&m2_ran), "m2")),
        ]));
        let exec = ToolExecutor::new(reg, Arc::new(AllowAll));

        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext {
            working_dir: PathBuf::from("/tmp"),
            cancel: cancel.clone(),
            depth: 0,
            max_depth: 1,
            executor: empty_executor(),
        };

        // Pre-cancel before invocation: NEITHER tool should run, both
        // should produce synthetic cancelled errors.
        cancel.cancel();
        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "m1".into(),
                input: json!({}),
            },
            ToolCall {
                id: "2".into(),
                name: "m2".into(),
                input: json!({}),
            },
        ];
        let results = exec.execute_batch(calls, &ctx).await;

        assert_eq!(results.len(), 2, "result count must match input count");
        for r in &results {
            let Content::ToolResult {
                content, is_error, ..
            } = r
            else {
                panic!("expected tool_result");
            };
            assert!(*is_error, "cancelled-before-execution should be is_error");
            assert!(
                content.contains("cancelled before execution"),
                "got: {content}"
            );
        }
        assert!(
            !m1_ran.load(std::sync::atomic::Ordering::SeqCst),
            "m1 must not have run after cancel"
        );
        assert!(
            !m2_ran.load(std::sync::atomic::Ordering::SeqCst),
            "m2 must not have run after cancel"
        );
    }

    // --- Approval-gate tests -----------------------------------------------

    /// Approval handler that always denies with a fixed reason.
    struct AlwaysDeny(&'static str);
    #[async_trait]
    impl ApprovalHandler for AlwaysDeny {
        async fn approve(&self, _: &str, _: &Value, _: ToolClass) -> ApprovalDecision {
            ApprovalDecision::Deny(self.0.to_string())
        }
    }

    /// Approval handler that takes 10s before answering — used to
    /// prove cancellation interrupts a hung approval.
    struct SlowApproval;
    #[async_trait]
    impl ApprovalHandler for SlowApproval {
        async fn approve(&self, _: &str, _: &Value, _: ToolClass) -> ApprovalDecision {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            ApprovalDecision::Allow
        }
    }

    #[tokio::test]
    async fn approval_deny_emits_error_tool_result_and_skips_execution() {
        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ran_clone = Arc::clone(&ran);

        struct ObservingTool(Arc<std::sync::atomic::AtomicBool>);
        #[async_trait]
        impl Tool for ObservingTool {
            fn name(&self) -> &str {
                "observe"
            }
            fn description(&self) -> &str {
                "observes whether it ran"
            }
            fn input_schema(&self) -> Value {
                json!({})
            }
            async fn execute(&self, _: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(ToolOutput::text("ran"))
            }
        }

        let reg = Arc::new(ToolRegistry::new(vec![Arc::new(ObservingTool(ran_clone))]));
        let exec = ToolExecutor::with_approval(
            reg,
            Arc::new(AllowAll),
            Arc::new(AlwaysDeny("blocked by user")),
        );
        let res = exec.execute_one(call("observe", json!({})), &ctx()).await;
        let Content::ToolResult {
            content, is_error, ..
        } = res
        else {
            panic!("expected tool_result");
        };

        assert!(is_error, "denied call should yield is_error: true");
        assert!(
            content.contains("approval denied"),
            "content should mark approval denial, got: {content}"
        );
        assert!(
            content.contains("blocked by user"),
            "content should preserve the deny reason, got: {content}"
        );
        assert!(
            !ran.load(std::sync::atomic::Ordering::SeqCst),
            "tool must NOT have executed after approval denial"
        );
    }

    #[tokio::test]
    async fn approval_cancel_during_approve_short_circuits() {
        let reg = Arc::new(ToolRegistry::new(vec![Arc::new(Echo)]));
        let exec = ToolExecutor::with_approval(reg, Arc::new(AllowAll), Arc::new(SlowApproval));

        let cancel = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext {
            working_dir: PathBuf::from("/tmp"),
            cancel: cancel.clone(),
            depth: 0,
            max_depth: 1,
            executor: empty_executor(),
        };

        // Fire cancel after 50ms; SlowApproval would take 10s otherwise.
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            cancel_clone.cancel();
        });

        let started = std::time::Instant::now();
        let res = exec
            .execute_one(call("echo", json!({"msg": "x"})), &ctx)
            .await;
        let elapsed = started.elapsed();

        let Content::ToolResult {
            content, is_error, ..
        } = res
        else {
            panic!("expected tool_result");
        };
        assert!(is_error, "cancel during approval should yield is_error");
        assert!(
            content.contains("cancelled"),
            "content should mention cancellation, got: {content}"
        );
        // Critical: the 10s SlowApproval future was racing the 50ms
        // cancel. With biased select! on cancel-first, we must beat
        // 10s by an order of magnitude. 1s is comfortable slack.
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "cancel should win the race against approve(); took {elapsed:?}"
        );
    }
}
