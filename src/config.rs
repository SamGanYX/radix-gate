use std::fs;
use std::net::SocketAddr;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Master configuration schema for RadixGate.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GatewayConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub balancer: BalancerConfig,
    #[serde(default)]
    pub mesh: MeshConfigYaml,
    #[serde(default)]
    pub discovery: DiscoveryConfigYaml,
    #[serde(default)]
    pub workers: Vec<WorkerConfigYaml>,
    #[serde(default)]
    pub models: Vec<ModelRouteConfig>,
}

impl GatewayConfig {
    /// Loads and validates configuration from a YAML file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Validates configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.server.block_size == 0 {
            return Err("server.block_size must be greater than 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.balancer.alpha) {
            return Err("balancer.alpha must be between 0.0 and 1.0".to_string());
        }
        if !(0.0..=1.0).contains(&self.balancer.beta) {
            return Err("balancer.beta must be between 0.0 and 1.0".to_string());
        }
        if !(0.0..=1.0).contains(&self.balancer.saturation_threshold) {
            return Err("balancer.saturation_threshold must be between 0.0 and 1.0".to_string());
        }
        for w in &self.workers {
            if w.id.trim().is_empty() {
                return Err("Worker id cannot be empty".to_string());
            }
            if w.endpoint_url.trim().is_empty() {
                return Err(format!(
                    "Worker '{}' must define a non-empty endpoint_url",
                    w.id
                ));
            }
            if w.max_concurrency == 0 {
                return Err(format!(
                    "Worker '{}' max_concurrency must be greater than 0",
                    w.id
                ));
            }
        }
        Ok(())
    }
}

/// HTTP server options.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_block_size")]
    pub block_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            block_size: default_block_size(),
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8080
}
fn default_block_size() -> usize {
    64
}

/// Routing balancer options.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BalancerConfig {
    #[serde(default = "default_policy")]
    pub policy: String,
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    #[serde(default = "default_beta")]
    pub beta: f64,
    #[serde(default = "default_saturation")]
    pub saturation_threshold: f64,
}

impl Default for BalancerConfig {
    fn default() -> Self {
        Self {
            policy: default_policy(),
            alpha: default_alpha(),
            beta: default_beta(),
            saturation_threshold: default_saturation(),
        }
    }
}

fn default_policy() -> String {
    "prefix_aware".to_string()
}
fn default_alpha() -> f64 {
    0.75
}
fn default_beta() -> f64 {
    0.25
}
fn default_saturation() -> f64 {
    0.85
}

/// UDP gossip cluster mesh options.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MeshConfigYaml {
    #[serde(default)]
    pub enabled: bool,
    pub node_id: Option<String>,
    pub bind_addr: Option<SocketAddr>,
    #[serde(default)]
    pub peers: Vec<SocketAddr>,
}

/// Dynamic Kubernetes / DNS service discovery options.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscoveryConfigYaml {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_service_address")]
    pub service_address: String,
    #[serde(default = "default_discovery_poll_interval_secs")]
    pub poll_interval_secs: u64,
    #[serde(default = "default_kv_capacity")]
    pub default_kv_capacity_blocks: usize,
    #[serde(default = "default_max_concurrency")]
    pub default_max_concurrency: usize,
    #[serde(default = "default_role")]
    pub default_role: String,
}

impl Default for DiscoveryConfigYaml {
    fn default() -> Self {
        Self {
            enabled: false,
            service_address: default_service_address(),
            poll_interval_secs: default_discovery_poll_interval_secs(),
            default_kv_capacity_blocks: default_kv_capacity(),
            default_max_concurrency: default_max_concurrency(),
            default_role: default_role(),
        }
    }
}

fn default_service_address() -> String {
    "vllm-headless.default.svc.cluster.local:8000".to_string()
}
fn default_discovery_poll_interval_secs() -> u64 {
    5
}

/// Declarative worker replica configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerConfigYaml {
    pub id: String,
    pub endpoint_url: String,
    #[serde(default = "default_kv_capacity")]
    pub kv_capacity_blocks: usize,
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,
    #[serde(default = "default_role")]
    pub role: String,
}

fn default_kv_capacity() -> usize {
    10_000
}
fn default_max_concurrency() -> usize {
    32
}
fn default_role() -> String {
    "unified".to_string()
}

/// Model routing rules.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelRouteConfig {
    pub model: String,
    #[serde(default)]
    pub allowed_workers: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let cfg = GatewayConfig::default();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.server.port, 8080);
        assert_eq!(cfg.balancer.alpha, 0.75);
    }

    #[test]
    fn test_parse_valid_yaml() {
        let yaml = r#"
server:
  host: "127.0.0.1"
  port: 9000
  block_size: 128

balancer:
  policy: "p2c_prefix"
  alpha: 0.80
  beta: 0.20
  saturation_threshold: 0.90

mesh:
  enabled: true
  node_id: "gw-test-01"
  bind_addr: "127.0.0.1:9090"
  peers:
    - "127.0.0.1:9091"

workers:
  - id: "worker-01"
    endpoint_url: "http://10.0.0.1:8000"
    kv_capacity_blocks: 20000
    max_concurrency: 64
    role: "prefill"
  - id: "worker-02"
    endpoint_url: "http://10.0.0.2:8000"
    role: "decode"
"#;
        let cfg: GatewayConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.server.port, 9000);
        assert_eq!(cfg.server.block_size, 128);
        assert_eq!(cfg.balancer.policy, "p2c_prefix");
        assert_eq!(cfg.balancer.alpha, 0.80);
        assert_eq!(cfg.workers.len(), 2);
        assert_eq!(cfg.workers[0].role, "prefill");
        assert_eq!(cfg.workers[1].role, "decode");
        assert_eq!(cfg.workers[1].max_concurrency, 32);
    }

    #[test]
    fn test_invalid_config_validation() {
        let mut cfg = GatewayConfig::default();
        cfg.server.block_size = 0;
        assert!(cfg.validate().is_err());

        let mut cfg2 = GatewayConfig::default();
        cfg2.balancer.alpha = 1.5;
        assert!(cfg2.validate().is_err());

        let mut cfg3 = GatewayConfig::default();
        cfg3.workers.push(WorkerConfigYaml {
            id: "".into(),
            endpoint_url: "http://localhost:8000".into(),
            kv_capacity_blocks: 100,
            max_concurrency: 10,
            role: "unified".into(),
        });
        assert!(cfg3.validate().is_err());
    }
}
