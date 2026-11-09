use crate::trie::BlockHash;
use tiktoken_rs::{CoreBPE, cl100k_base};
use xxhash_rust::xxh3::{Xxh3, xxh3_64};

/// Utility for chunking text and prompt messages into deterministic 64-bit block hashes.
/// Supports both exact BPE token-level chunking and fast byte streaming.
#[derive(Clone)]
pub struct BlockChunker {
    /// Number of tokens per KV block (e.g., 64 for PagedAttention, 1024 for Anthropic breakpoints).
    pub block_size: usize,
    /// Core BPE tokenizer instance (lazy/cached).
    bpe: Option<CoreBPE>,
}

impl std::fmt::Debug for BlockChunker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockChunker")
            .field("block_size", &self.block_size)
            .field("bpe_enabled", &self.bpe.is_some())
            .finish()
    }
}

impl Default for BlockChunker {
    fn default() -> Self {
        Self::new(64)
    }
}

impl BlockChunker {
    /// Creates a new chunker with BPE tokenization initialized.
    pub fn new(block_size: usize) -> Self {
        let bpe = cl100k_base().ok();
        Self { block_size, bpe }
    }

    /// Creates a lightweight chunker without BPE (fast word-slice approximation).
    pub fn new_fast(block_size: usize) -> Self {
        Self {
            block_size,
            bpe: None,
        }
    }

    /// Chunks a raw sequence of u32 token IDs into deterministic 64-bit block hashes.
    pub fn chunk_tokens(&self, tokens: &[u32]) -> Vec<BlockHash> {
        tokens
            .chunks(self.block_size)
            .map(|chunk| {
                let bytes: &[u8] = bytemuck_or_slice_u32_to_bytes(chunk);
                xxh3_64(bytes)
            })
            .collect()
    }

    /// Chunks text using exact BPE tokenization (byte-for-byte identical to model input).
    /// If BPE tokenizer is available, produces exact token block hashes; otherwise falls back to fast stream.
    pub fn chunk_text(&self, text: &str) -> Vec<BlockHash> {
        if let Some(ref bpe) = self.bpe {
            let tokens = bpe.encode_with_special_tokens(text);
            self.chunk_tokens(&tokens)
        } else {
            self.chunk_text_fast(text)
        }
    }

    /// Fast zero-allocation fallback chunker without BPE overhead.
    pub fn chunk_text_fast(&self, text: &str) -> Vec<BlockHash> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }

        words
            .chunks(self.block_size)
            .map(|chunk| {
                let mut hasher = Xxh3::new();
                for (i, word) in chunk.iter().enumerate() {
                    if i > 0 {
                        hasher.update(b" ");
                    }
                    hasher.update(word.as_bytes());
                }
                hasher.digest()
            })
            .collect()
    }

    /// Combines multiple role-based messages into a canonical prompt prefix and chunks it.
    pub fn chunk_messages(&self, messages: &[(&str, &str)]) -> Vec<BlockHash> {
        let mut full_repr = String::new();
        for (role, content) in messages {
            full_repr.push_str("<|im_start|>");
            full_repr.push_str(role);
            full_repr.push('\n');
            full_repr.push_str(content);
            full_repr.push_str("<|im_end|>\n");
        }
        self.chunk_text(&full_repr)
    }
}

fn bytemuck_or_slice_u32_to_bytes(slice: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_exact_token_chunking() {
        let chunker = BlockChunker::new(4);
        let prompt = "Anthropic prompt caching reduces prefill latency.";
        let hashes = chunker.chunk_text(prompt);
        assert!(!hashes.is_empty());

        // Repeated chunking must be deterministic
        let hashes_repeat = chunker.chunk_text(prompt);
        assert_eq!(hashes, hashes_repeat);
    }
}
