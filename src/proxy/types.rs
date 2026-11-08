use serde::{Deserialize, Serialize};

/// Represents an explicit prompt segment with optional prompt cache control.
#[derive(Debug, Clone)]
pub struct PromptSegment {
    pub text: String,
    pub has_cache_control: bool,
}

/// Generic message block supporting both string and structured content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

impl ChatMessage {
    pub fn content_as_str(&self) -> String {
        if let Some(s) = self.content.as_str() {
            s.to_string()
        } else if let Some(arr) = self.content.as_array() {
            let mut out = String::new();
            for item in arr {
                if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                    out.push_str(text);
                    out.push(' ');
                }
            }
            out
        } else {
            self.content.to_string()
        }
    }

    /// Extracts granular segments and detects Anthropic `cache_control: {"type": "ephemeral"}` markers.
    pub fn extract_segments(&self) -> Vec<PromptSegment> {
        let mut segments = Vec::new();

        if let Some(s) = self.content.as_str() {
            segments.push(PromptSegment {
                text: s.to_string(),
                has_cache_control: false,
            });
        } else if let Some(arr) = self.content.as_array() {
            for item in arr {
                let text = item.get("text").and_then(|t| t.as_str()).unwrap_or("");
                let has_cache_control = item.get("cache_control").is_some();
                segments.push(PromptSegment {
                    text: text.to_string(),
                    has_cache_control,
                });
            }
        } else {
            segments.push(PromptSegment {
                text: self.content.to_string(),
                has_cache_control: false,
            });
        }

        segments
    }
}

/// Incoming chat completion request (OpenAI / Anthropic unified model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    #[serde(default)]
    pub system: Option<serde_json::Value>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
}

impl ChatRequest {
    /// Extracts canonical text for prompt prefix matching.
    pub fn extract_prompt_text(&self) -> String {
        let mut full_prompt = String::new();

        if let Some(sys) = &self.system {
            full_prompt.push_str("<|system|>\n");
            if let Some(s) = sys.as_str() {
                full_prompt.push_str(s);
            } else if let Some(arr) = sys.as_array() {
                for item in arr {
                    if let Some(t) = item.get("text").and_then(|x| x.as_str()) {
                        full_prompt.push_str(t);
                        full_prompt.push(' ');
                    }
                }
            } else {
                full_prompt.push_str(&sys.to_string());
            }
            full_prompt.push_str("\n<|end|>\n");
        }

        for msg in &self.messages {
            full_prompt.push_str(&format!("<|{}|>\n", msg.role));
            full_prompt.push_str(&msg.content_as_str());
            full_prompt.push_str("\n<|end|>\n");
        }

        full_prompt
    }

    /// Checks if the request contains explicit Anthropic prompt caching breakpoints.
    pub fn has_explicit_cache_breakpoints(&self) -> bool {
        if let Some(arr) = self.system.as_ref().and_then(|s| s.as_array()) {
            for item in arr {
                if item.get("cache_control").is_some() {
                    return true;
                }
            }
        }

        for msg in &self.messages {
            for seg in msg.extract_segments() {
                if seg.has_cache_control {
                    return true;
                }
            }
        }

        false
    }
}

/// Proxy routing response metadata headers and wrapper.
#[derive(Debug, Serialize, Deserialize)]
pub struct GatewayMetricsResponse {
    pub cluster_workers: usize,
    pub radix_tree_nodes: usize,
    pub routing_decision: Option<String>,
    pub matched_blocks: usize,
    pub total_blocks: usize,
    pub cache_hit_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_anthropic_cache_control_detection() {
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "claude-3-5-sonnet-20241022",
            "system": [{
                "type": "text",
                "text": "Large documentation prefix...",
                "cache_control": {"type": "ephemeral"}
            }],
            "messages": [{
                "role": "user",
                "content": "Hello"
            }]
        }))
        .unwrap();

        assert!(request.has_explicit_cache_breakpoints());
        assert!(
            request
                .extract_prompt_text()
                .contains("Large documentation prefix")
        );
    }
}
