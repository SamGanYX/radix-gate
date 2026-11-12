use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tracing::{info, warn};

use crate::shadow_cache::{WorkerPool, WorkerReplica, WorkerRole};
use crate::trie::RadixTree;

/// Configuration for dynamic inference replica service discovery.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Target service hostname and port to resolve (e.g. "vllm-headless.default.svc.cluster.local:8000")
    pub service_address: String,
    /// Polling interval for DNS / endpoint changes
    pub poll_interval: Duration,
    /// Default KV capacity blocks for newly discovered workers
    pub default_kv_blocks: usize,
    /// Default max concurrency per discovered replica
    pub default_max_concurrency: usize,
    /// Default role assigned to discovered replicas
    pub default_role: WorkerRole,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            service_address: "127.0.0.1:8000".to_string(),
            poll_interval: Duration::from_secs(5),
            default_kv_blocks: 20_000,
            default_max_concurrency: 32,
            default_role: WorkerRole::Unified,
        }
    }
}

/// Watches for dynamic cluster scaling, discovering new GPU worker pods
/// and evicting terminated pods from the active Radix Tree without gateway restarts.
pub struct ServiceDiscoveryWatcher {
    pub config: DiscoveryConfig,
    pub pool: Arc<WorkerPool>,
    pub tree: Arc<RwLock<RadixTree>>,
    pub known_addrs: Arc<RwLock<HashSet<SocketAddr>>>,
}

impl ServiceDiscoveryWatcher {
    pub fn new(
        config: DiscoveryConfig,
        pool: Arc<WorkerPool>,
        tree: Arc<RwLock<RadixTree>>,
    ) -> Self {
        Self {
            config,
            pool,
            tree,
            known_addrs: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Performs one synchronization tick against the upstream service name.
    pub async fn sync_once(&self) -> Result<(usize, usize), std::io::Error> {
        let addrs: HashSet<SocketAddr> = tokio::net::lookup_host(&self.config.service_address)
            .await?
            .collect();

        let (added, removed) = self.reconcile_addresses(&addrs);
        Ok((added, removed))
    }

    /// Reconciles a newly resolved set of addresses against the active worker pool and radix tree.
    pub fn reconcile_addresses(&self, addrs: &HashSet<SocketAddr>) -> (usize, usize) {
        let mut added = 0;
        let mut removed = 0;

        let mut known = self.known_addrs.write();

        // 1. Identify newly discovered replicas
        for addr in addrs {
            if !known.contains(addr) {
                let id = format!("worker-{}", addr.to_string().replace(['.', ':'], "-"));
                let url = format!("http://{}", addr);

                info!(
                    worker = %id,
                    endpoint = %url,
                    role = ?self.config.default_role,
                    "[ServiceDiscovery] Discovered new inference replica pod"
                );

                self.pool.add_worker(WorkerReplica::new_with_role(
                    id,
                    url,
                    self.config.default_kv_blocks,
                    self.config.default_max_concurrency,
                    self.config.default_role,
                ));
                added += 1;
            }
        }

        // 2. Identify and evict terminated replicas
        for addr in known.iter() {
            if !addrs.contains(addr) {
                let id = format!("worker-{}", addr.to_string().replace(['.', ':'], "-"));
                warn!(
                    worker = %id,
                    endpoint = %addr,
                    "[ServiceDiscovery] Inference replica terminated/scaled down; evicting from Radix Tree"
                );

                if let Some(w) = self.pool.get_worker(&id) {
                    w.write().is_healthy = false;
                }
                self.tree.write().evict_worker(&id);
                removed += 1;
            }
        }

        *known = addrs.clone();
        (added, removed)
    }

    /// Starts the background discovery loop continuously watching for cluster scale up/down events.
    pub fn start_watcher(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self.config.poll_interval);
            loop {
                interval.tick().await;
                if let Err(e) = self.sync_once().await {
                    tracing::debug!(
                        target = %self.config.service_address,
                        error = %e,
                        "[ServiceDiscovery] DNS resolution probe failed (retrying)"
                    );
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_discovery_scale_up_and_scale_down() {
        let pool = Arc::new(WorkerPool::new());
        let tree = Arc::new(RwLock::new(RadixTree::new(64)));
        let watcher =
            ServiceDiscoveryWatcher::new(DiscoveryConfig::default(), pool.clone(), tree.clone());

        // 1. Initial scale up: 2 worker pods discovered
        let mut initial_addrs = HashSet::new();
        initial_addrs.insert("10.0.1.10:8000".parse().unwrap());
        initial_addrs.insert("10.0.1.11:8000".parse().unwrap());

        let (added, removed) = watcher.reconcile_addresses(&initial_addrs);
        assert_eq!(added, 2);
        assert_eq!(removed, 0);
        assert_eq!(pool.len(), 2);

        // Pre-warm a prefix on worker 1
        let w1_id = "worker-10-0-1-10-8000";
        tree.write().insert(&[12345, 67890], w1_id);
        assert!(
            tree.read()
                .find_longest_prefix(&[12345, 67890])
                .worker_matches
                .contains_key(w1_id)
        );

        // 2. Scale up: 1 additional pod joins (total 3)
        let mut scaled_up = initial_addrs.clone();
        scaled_up.insert("10.0.1.12:8000".parse().unwrap());
        let (added, removed) = watcher.reconcile_addresses(&scaled_up);
        assert_eq!(added, 1);
        assert_eq!(removed, 0);
        assert_eq!(pool.len(), 3);

        // 3. Scale down: Pod 10.0.1.10 is terminated
        let mut scaled_down = HashSet::new();
        scaled_down.insert("10.0.1.11:8000".parse().unwrap());
        scaled_down.insert("10.0.1.12:8000".parse().unwrap());
        let (added, removed) = watcher.reconcile_addresses(&scaled_down);
        assert_eq!(added, 0);
        assert_eq!(removed, 1);

        // Terminated worker must be circuit-broken and evicted from Radix Tree!
        let w1 = pool.get_worker(w1_id).unwrap();
        assert!(!w1.read().is_healthy);
        assert!(
            !tree
                .read()
                .find_longest_prefix(&[12345, 67890])
                .worker_matches
                .contains_key(w1_id)
        );
    }
}
