//! Streaming primitives.
//!
//! `complete()` returns a buffered final answer; `stream()` returns the
//! same answer as a sequence of events as the model produces them. The
//! event types are deliberately coarser than the wire format — we
//! aggregate provider-side delta JSON into atomic `ToolUse` events
//! before emitting, so consumers never have to parse partial JSON.
//!
//! The stream item type is `Result<StreamEvent, ProviderError>`. Errors
//! that surface mid-stream (parser failures, mid-body HTTP glitches)
//! arrive as `Err`. Errors that happen before the stream starts
//! (auth, malformed request, connect refused) are returned by the
//! `stream(...)` async fn itself.

use std::pin::Pin;

use futures::stream::Stream;
use serde_json::Value;

use crate::error::ProviderError;
use crate::message::{StopReason, Usage};

/// One unit of progress from a streaming provider.
///
/// `ContentDelta` arrives in many small chunks during generation.
/// `ToolUse` is **atomic** — provider-side `input_json_delta` fragments
/// are accumulated by our parser and emitted only when the block
/// closes, so the consumer never has to handle partial JSON.
/// `MessageDelta` carries the final `stop_reason`. `Usage` typically
/// arrives once at the end. `Done` is the terminal marker.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A piece of assistant text. Concatenate in order to reconstruct
    /// the full reply.
    ContentDelta(String),

    /// A complete tool invocation request. The agent loop will execute
    /// it and feed the result back on the next turn.
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },

    /// Final stop reason from the provider for this turn.
    MessageDelta { stop_reason: StopReason },

    /// Token usage for this turn. Some providers emit it only at the
    /// end (Anthropic), others split input/output across events.
    Usage(Usage),

    /// Stream terminated normally. No more events will follow.
    Done,
}

/// Boxed, object-safe stream of provider events. Used as the return
/// payload of `LlmProvider::stream`.
pub type ProviderEventStream =
    Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>;
