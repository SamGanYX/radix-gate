use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::net::UdpSocket;
use tracing::{debug, error, info, warn};

use crate::trie::{BlockHash, RadixTree};

pub const GOSSIP_MAGIC: u32 = 0x52474154; // 'RGAT' in ASCII

/// Compact network packet broadcast across peer gateways to maintain cluster-wide cache coherence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrefixGossipMessage {
    pub magic: u32,
    pub sender_id: String,
    pub worker_id: String,
    pub block_hashes: Vec<BlockHash>,
    pub sequence: u64,
    pub timestamp_ms: u64,
}

impl PrefixGossipMessage {
    pub fn new(
        sender_id: String,
        worker_id: String,
        block_hashes: Vec<BlockHash>,
        seq: u64,
    ) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            magic: GOSSIP_MAGIC,
            sender_id,
            worker_id,
            block_hashes,
            sequence: seq,
            timestamp_ms,
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

/// Configuration parameters for the Distributed Gateway Mesh.
#[derive(Debug, Clone)]
pub struct MeshConfig {
    pub node_id: String,
    pub bind_addr: SocketAddr,
    pub peer_addrs: Vec<SocketAddr>,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            node_id: format!("gateway-{:04x}", rand::random::<u16>()),
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            peer_addrs: Vec::new(),
        }
    }
}

/// Handles peer discovery, state broadcasting, and background packet ingestion.
pub struct GatewayMesh {
    config: MeshConfig,
    socket: Arc<UdpSocket>,
    radix_tree: Arc<RwLock<RadixTree>>,
    sequence: AtomicU64,
    seen_messages: Arc<RwLock<HashSet<(String, u64)>>>,
}

impl GatewayMesh {
    pub async fn bind(
        config: MeshConfig,
        radix_tree: Arc<RwLock<RadixTree>>,
    ) -> Result<Self, std::io::Error> {
        let socket = UdpSocket::bind(config.bind_addr).await?;
        let local_addr = socket.local_addr()?;
        info!(
            node_id = %config.node_id,
            bind_addr = %local_addr,
            peers = ?config.peer_addrs,
            "[GatewayMesh] Initialized UDP gossip mesh node"
        );

        Ok(Self {
            config,
            socket: Arc::new(socket),
            radix_tree,
            sequence: AtomicU64::new(1),
            seen_messages: Arc::new(RwLock::new(HashSet::new())),
        })
    }

    pub fn local_addr(&self) -> Result<SocketAddr, std::io::Error> {
        self.socket.local_addr()
    }

    /// Broadcasts a newly cached prefix to all configured peer gateways.
    pub async fn broadcast_prefix(&self, worker_id: &str, blocks: &[BlockHash]) {
        if self.config.peer_addrs.is_empty() || blocks.is_empty() {
            return;
        }

        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);
        let msg = PrefixGossipMessage::new(
            self.config.node_id.clone(),
            worker_id.to_string(),
            blocks.to_vec(),
            seq,
        );

        let Ok(bytes) = msg.to_bytes() else {
            return;
        };

        for peer in &self.config.peer_addrs {
            if let Err(e) = self.socket.send_to(&bytes, peer).await {
                debug!(peer = %peer, error = %e, "[GatewayMesh] Failed to broadcast gossip packet to peer");
            }
        }
    }

    /// Spawns the background receiver task continuously ingesting gossip updates from peers.
    pub fn start_listener(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut buf = vec![0u8; 65535];
            loop {
                match self.socket.recv_from(&mut buf).await {
                    Ok((len, src)) => {
                        let packet = &buf[..len];
                        match PrefixGossipMessage::from_bytes(packet) {
                            Ok(msg) => {
                                if msg.magic != GOSSIP_MAGIC {
                                    continue;
                                }
                                if msg.sender_id == self.config.node_id {
                                    continue; // Ignore loopback packets
                                }

                                let msg_key = (msg.sender_id.clone(), msg.sequence);
                                {
                                    let mut seen = self.seen_messages.write();
                                    if seen.contains(&msg_key) {
                                        continue;
                                    }
                                    seen.insert(msg_key);
                                    if seen.len() > 10_000 {
                                        seen.clear();
                                    }
                                }

                                debug!(
                                    from = %src,
                                    sender = %msg.sender_id,
                                    worker = %msg.worker_id,
                                    blocks = msg.block_hashes.len(),
                                    "[GatewayMesh] Received prefix gossip announcement"
                                );

                                // Synchronize remote prefix into local Radix Tree shard
                                {
                                    let mut tree = self.radix_tree.write();
                                    tree.insert(&msg.block_hashes, &msg.worker_id);
                                }
                            }
                            Err(e) => {
                                warn!(from = %src, error = %e, "[GatewayMesh] Corrupt packet discarded");
                            }
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "[GatewayMesh] Socket read failure");
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mesh_cross_gateway_cache_sync() {
        let tree_a = Arc::new(RwLock::new(RadixTree::new(64)));
        let tree_b = Arc::new(RwLock::new(RadixTree::new(64)));

        // Bind Node B first on ephemeral port
        let config_b = MeshConfig {
            node_id: "node-b".to_string(),
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            peer_addrs: Vec::new(),
        };
        let mesh_b = Arc::new(GatewayMesh::bind(config_b, tree_b.clone()).await.unwrap());
        let addr_b = mesh_b.local_addr().unwrap();
        let _listener_b = mesh_b.clone().start_listener();

        // Bind Node A pointing to Node B as peer
        let config_a = MeshConfig {
            node_id: "node-a".to_string(),
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            peer_addrs: vec![addr_b],
        };
        let mesh_a = Arc::new(GatewayMesh::bind(config_a, tree_a.clone()).await.unwrap());

        // Node A inserts a prefix locally and broadcasts to Node B
        let prefix = vec![0x1111, 0x2222, 0x3333];
        tree_a.write().insert(&prefix, "worker-gpu-01");
        mesh_a.broadcast_prefix("worker-gpu-01", &prefix).await;

        // Allow async UDP packet transit
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Verify Node B now knows that worker-gpu-01 holds the prefix without ever touching worker-gpu-01!
        let match_b = tree_b.read().find_longest_prefix(&prefix);
        assert_eq!(match_b.max_matched_blocks, 3);
        assert_eq!(match_b.worker_matches.get("worker-gpu-01"), Some(&3));
    }
}
