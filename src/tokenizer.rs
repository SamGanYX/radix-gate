use crate::trie::BlockHash;
use xxhash_rust::xxh3::xxh3_64;

/// Utility for chunking text and prompt messages into deterministic 64-bit block hashes.
#[derive(Debug, Clone)]
pub struct BlockChunker {
    /// Number of tokens (or word-approximations) per KV block.
    pub block_size: usize,
}

impl Default for BlockChunker {
    fn default() -> Self {
        Self { block_size: 64 }
    }
}

impl BlockChunker {
    pub fn new(block_size: usize) -> Self {
        Self { block_size }
    }

    /// Chunks a raw sequence of token IDs into 64-bit block hashes.
    pub fn chunk_tokens(&self, tokens: &[u32]) -> Vec<BlockHash> {
        tokens
            .chunks(self.block_size)
            .map(|chunk| {
                let bytes: &[u8] = bytemuck_or_slice_u32_to_bytes(chunk);
                xxh3_64(bytes)
            })
            .collect()
    }

    /// Fast, deterministic chunker for string prompts.
    /// Splits on whitespace boundaries with window size approximation.
    pub fn chunk_text(&self, text: &str) -> Vec<BlockHash> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }

        // Each word is ~1.3 tokens on average. We chunk by block_size words
        words
            .chunks(self.block_size)
            .map(|chunk| {
                let joined = chunk.join(" ");
                xxh3_64(joined.as_bytes())
            })
            .collect()
    }

    /// Combines multiple role-based messages (system, user, assistant) into a canonical prefix.
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
    // Safe alignment-preserving cast of u32 slice to byte slice
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<u32>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunker_deterministic() {
        let chunker = BlockChunker::new(4);
        let text = "You are a helpful coding assistant specialized in high performance distributed systems and ML inference.";
        let hashes1 = chunker.chunk_text(text);
        let hashes2 = chunker.chunk_text(text);

        assert!(!hashes1.is_empty());
        assert_eq!(hashes1, hashes2);
    }
}
