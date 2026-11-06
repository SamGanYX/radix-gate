use serde::{Deserialize, Serialize};

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
