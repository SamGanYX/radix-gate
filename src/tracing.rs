use rand::RngCore;

/// W3C Distributed Trace Context (traceparent specification).
/// Enables distributed trace correlation between client, RadixGate proxy, and upstream inference engines (vLLM / SGLang).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub sampled: bool,
}

impl TraceContext {
    /// Generates a new root trace context with cryptographic pseudo-randomness.
    pub fn new_root(sampled: bool) -> Self {
        let mut rng = rand::thread_rng();
        let mut trace_bytes = [0u8; 16];
        let mut span_bytes = [0u8; 8];
        rng.fill_bytes(&mut trace_bytes);
        rng.fill_bytes(&mut span_bytes);

        Self {
            trace_id: hex::encode(trace_bytes),
            span_id: hex::encode(span_bytes),
            parent_span_id: None,
            sampled,
        }
    }

    /// Derives a child span context within the current distributed trace.
    pub fn child_span(&self) -> Self {
        let mut rng = rand::thread_rng();
        let mut span_bytes = [0u8; 8];
        rng.fill_bytes(&mut span_bytes);

        Self {
            trace_id: self.trace_id.clone(),
            span_id: hex::encode(span_bytes),
            parent_span_id: Some(self.span_id.clone()),
            sampled: self.sampled,
        }
    }

    /// Parses a standard W3C `traceparent` header (`00-{trace_id}-{parent_id}-{trace_flags}`).
    pub fn from_traceparent(header_val: &str) -> Option<Self> {
        let parts: Vec<&str> = header_val.trim().split('-').collect();
        if parts.len() < 4 {
            return None;
        }

        let version = parts[0];
        if version != "00" {
            // Unsupported version
            return None;
        }

        let trace_id = parts[1];
        let parent_id = parts[2];
        let flags = parts[3];

        if trace_id.len() != 32 || parent_id.len() != 16 || flags.len() != 2 {
            return None;
        }

        // Validate hex
        if !trace_id.chars().all(|c| c.is_ascii_hexdigit())
            || !parent_id.chars().all(|c| c.is_ascii_hexdigit())
            || !flags.chars().all(|c| c.is_ascii_hexdigit())
        {
            return None;
        }

        let sampled = flags == "01";

        // Generate a new span ID for RadixGate execution, marking the received ID as parent
        let mut rng = rand::thread_rng();
        let mut span_bytes = [0u8; 8];
        rng.fill_bytes(&mut span_bytes);

        Some(Self {
            trace_id: trace_id.to_lowercase(),
            span_id: hex::encode(span_bytes),
            parent_span_id: Some(parent_id.to_lowercase()),
            sampled,
        })
    }

    /// Formats as standard W3C `traceparent` string.
    pub fn to_traceparent(&self) -> String {
        let flag = if self.sampled { "01" } else { "00" };
        format!("00-{}-{}-{}", self.trace_id, self.span_id, flag)
    }
}

/// Simple hex encoder without external hex crate dependency.
mod hex {
    pub fn encode<T: AsRef<[u8]>>(data: T) -> String {
        data.as_ref().iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_context_generation() {
        let ctx = TraceContext::new_root(true);
        assert_eq!(ctx.trace_id.len(), 32);
        assert_eq!(ctx.span_id.len(), 16);
        assert!(ctx.sampled);
        assert!(ctx.parent_span_id.is_none());

        let tp = ctx.to_traceparent();
        assert!(tp.starts_with("00-"));
        assert!(tp.ends_with("-01"));
    }

    #[test]
    fn test_parse_valid_traceparent() {
        let header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let parsed = TraceContext::from_traceparent(header).unwrap();
        assert_eq!(parsed.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(parsed.parent_span_id.as_deref(), Some("00f067aa0ba902b7"));
        assert!(parsed.sampled);
        assert_eq!(parsed.span_id.len(), 16);
        assert_ne!(parsed.span_id, "00f067aa0ba902b7");
    }

    #[test]
    fn test_parse_invalid_traceparent() {
        assert!(TraceContext::from_traceparent("invalid").is_none());
        assert!(TraceContext::from_traceparent("01-abc-def-00").is_none());
        assert!(TraceContext::from_traceparent("00-short-00f067aa0ba902b7-01").is_none());
    }

    #[test]
    fn test_child_span_derivation() {
        let root = TraceContext::new_root(true);
        let child = root.child_span();
        assert_eq!(child.trace_id, root.trace_id);
        assert_ne!(child.span_id, root.span_id);
        assert_eq!(child.parent_span_id.as_deref(), Some(root.span_id.as_str()));
    }
}
