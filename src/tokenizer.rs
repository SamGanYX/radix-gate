use crate::trie::BlockHash;
use tiktoken_rs::{CoreBPE, cl100k_base};
use xxhash_rust::xxh3::{Xxh3, xxh3_64};

/// Utility for chunking text and prompt messages into deterministic 64-bit block hashes.
/// Supports both exact BPE token-level chunking and fast byte streaming.
#[derive(Clone)]
pub struct BlockChunker {
    /// Number of tokens per KV block (e.g., 64 for PagedAttention, 1024 for Anthropic breakpoints).
    pub block_size: usize,
    /// Maximum number of tokens to encode from the prompt prefix to protect gateway CPU on massive prompts.
    pub max_prefix_tokens: Option<usize>,
    /// Core BPE tokenizer instance (lazy/cached).
    bpe: Option<CoreBPE>,
}

impl std::fmt::Debug for BlockChunker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockChunker")
            .field("block_size", &self.block_size)
            .field("max_prefix_tokens", &self.max_prefix_tokens)
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
    /// Creates a new chunker with BPE tokenization initialized and default 4,096 max prefix cutoff.
    pub fn new(block_size: usize) -> Self {
        let bpe = cl100k_base().ok();
        Self {
            block_size,
            max_prefix_tokens: Some(4096),
            bpe,
        }
    }

    /// Sets the maximum number of prefix tokens to tokenize before early truncation.
    pub fn with_max_prefix_tokens(mut self, max: usize) -> Self {
        self.max_prefix_tokens = Some(max);
        self
    }

    /// Creates a lightweight chunker without BPE (fast word-slice approximation).
    pub fn new_fast(block_size: usize) -> Self {
        Self {
            block_size,
            max_prefix_tokens: Some(4096),
            bpe: None,
        }
    }

    /// Chunks a raw sequence of u32 token IDs into deterministic 64-bit block hashes.
    pub fn chunk_tokens(&self, tokens: &[u32]) -> Vec<BlockHash> {
        let (hashes, _) = self.chunk_tokens_with_residual(tokens);
        hashes
    }

    /// Chunks a raw sequence of u32 token IDs into deterministic 64-bit block hashes,
    /// also returning an optional residual hash for trailing tokens < block_size.
    pub fn chunk_tokens_with_residual(
        &self,
        tokens: &[u32],
    ) -> (Vec<BlockHash>, Option<BlockHash>) {
        let full_blocks = tokens.len() / self.block_size;
        let mut hashes = Vec::with_capacity(full_blocks);

        for chunk in tokens[..full_blocks * self.block_size].chunks_exact(self.block_size) {
            let bytes: &[u8] = bytemuck_or_slice_u32_to_bytes(chunk);
            hashes.push(xxh3_64(bytes));
        }

        let residual = if !tokens.len().is_multiple_of(self.block_size) {
            let rem = &tokens[full_blocks * self.block_size..];
            let bytes = bytemuck_or_slice_u32_to_bytes(rem);
            let seed = hashes.last().copied().unwrap_or(0);
            Some(xxhash_rust::xxh3::xxh3_64_with_seed(bytes, seed))
        } else {
            None
        };

        (hashes, residual)
    }

    /// Chunks text using exact BPE tokenization (byte-for-byte identical to model input).
    /// Employs early prefix cutoff to prevent gateway CPU saturation on massive prompts.
    pub fn chunk_text(&self, text: &str) -> Vec<BlockHash> {
        if let Some(ref bpe) = self.bpe {
            let input_slice = if let Some(max_tokens) = self.max_prefix_tokens {
                let char_cutoff = max_tokens.saturating_mul(6);
                if text.len() > char_cutoff {
                    let safe_idx = text[..char_cutoff]
                        .rfind(|c: char| c.is_whitespace() || c == '\n')
                        .unwrap_or(char_cutoff);
                    &text[..safe_idx]
                } else {
                    text
                }
            } else {
                text
            };

            let mut tokens = bpe.encode_with_special_tokens(input_slice);
            if let Some(max_tokens) = self.max_prefix_tokens {
                tokens.truncate(max_tokens);
            }
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

    #[test]
    fn test_residual_token_hashing() {
        let chunker = BlockChunker::new(4);
        let tokens = vec![1, 2, 3, 4, 5, 6]; // 1 full block of 4, residual of 2
        let (blocks, residual) = chunker.chunk_tokens_with_residual(&tokens);
        assert_eq!(blocks.len(), 1);
        assert!(residual.is_some());

        let exact_tokens = vec![1, 2, 3, 4];
        let (exact_blocks, exact_residual) = chunker.chunk_tokens_with_residual(&exact_tokens);
        assert_eq!(exact_blocks.len(), 1);
        assert!(exact_residual.is_none());
    }

    #[test]
    fn test_early_prefix_cutoff() {
        let chunker = BlockChunker::new(4).with_max_prefix_tokens(8);
        let massive_text = "word ".repeat(500); // 500 words
        let blocks = chunker.chunk_text(&massive_text);
        // With max 8 tokens and block_size 4, should produce at most 2 blocks!
        assert_eq!(blocks.len(), 2);
    }
}
