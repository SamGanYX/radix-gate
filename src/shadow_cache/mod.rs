pub mod worker;

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
pub use worker::WorkerReplica;

/// Manages the pool of active inference worker replicas and their shadow cache states.
#[derive(Debug, Default)]
pub struct WorkerPool {
    pub workers: RwLock<HashMap<String, Arc<RwLock<WorkerReplica>>>>,
    pub worker_order: RwLock<Vec<String>>,
}

impl WorkerPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a new worker to the cluster.
    pub fn add_worker(&self, worker: WorkerReplica) {
        let id = worker.id.clone();
        let mut workers = self.workers.write();
        let mut order = self.worker_order.write();
        if !order.contains(&id) {
            order.push(id.clone());
        }
        workers.insert(id, Arc::new(RwLock::new(worker)));
    }

    /// Retrieves an Arc to a worker by ID.
    pub fn get_worker(&self, id: &str) -> Option<Arc<RwLock<WorkerReplica>>> {
        let workers = self.workers.read();
        workers.get(id).cloned()
    }

    /// Returns a list of all active worker IDs.
    pub fn get_worker_ids(&self) -> Vec<String> {
        self.worker_order.read().clone()
    }

    /// Returns a list of all currently healthy worker IDs.
    pub fn get_healthy_worker_ids(&self) -> Vec<String> {
        let order = self.worker_order.read();
        let workers = self.workers.read();
        order
            .iter()
            .filter(|id| {
                workers
                    .get(*id)
                    .map(|w| w.read().is_healthy)
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// Number of workers currently registered.
    pub fn len(&self) -> usize {
        self.worker_order.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
