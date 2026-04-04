use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::error::ProviderError;
use crate::message::{Content, StopReason, Usage};
use crate::provider::{LlmProvider, Request, Response};

type ResponseFn = dyn Fn(&Request) -> Result<Response, ProviderError> + Send + Sync;

/// Mock LLM provider for testing.
///
/// # Example
///
/// ```ignore
/// use agent_runtime::providers::Mock;
/// use agent_runtime::message::{Content, StopReason, Usage};
/// use agent_runtime::provider::Response;
///
/// let mock = Mock::new(|_req| {
///     Ok(Response {
///         content: vec![Content::text("Hello from mock!")],
///         stop_reason: StopReason::EndTurn,
///         usage: Usage::default(),
///     })
/// });
/// ```
pub struct Mock {
    handler: Arc<ResponseFn>,
    call_count: Arc<Mutex<usize>>,
}

impl Mock {
    /// Create a mock provider with a custom response function.
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(&Request) -> Result<Response, ProviderError> + Send + Sync + 'static,
    {
        Self {
            handler: Arc::new(handler),
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Create a mock that always returns a text response.
    pub fn with_text(text: impl Into<String>) -> Self {
        let text = text.into();
        Self::new(move |_| {
            Ok(Response {
                content: vec![Content::text(&text)],
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            })
        })
    }

    /// Create a mock from a sequence of responses.
    ///
    /// Each call to `complete()` returns the next response in order.
    /// After the sequence is exhausted, returns the last response repeatedly.
    pub fn with_responses(responses: Vec<Response>) -> Self {
        let responses = Arc::new(responses);
        let index = Arc::new(Mutex::new(0usize));
        Self::new(move |_| {
            let mut i = index.lock().unwrap();
            let resp_idx = (*i).min(responses.len() - 1);
            *i += 1;
            let r = &responses[resp_idx];
            Ok(Response {
                content: r.content.clone(),
                stop_reason: r.stop_reason,
                usage: r.usage.clone(),
            })
        })
    }

    /// Returns how many times `complete()` was called.
    pub fn call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

#[async_trait]
impl LlmProvider for Mock {
    async fn complete(&self, request: Request) -> Result<Response, ProviderError> {
        {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
        }
        (self.handler)(&request)
    }
}
