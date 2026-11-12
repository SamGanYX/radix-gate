use std::io;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Gauge, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use serde_json::Value;

/// Runtime snapshot of cluster state parsed from /metrics and /api/v1/tree.
#[derive(Debug, Default, Clone)]
pub struct ClusterSnapshot {
    pub is_connected: bool,
    pub hit_ratio_pct: f64,
    pub total_requests: u64,
    pub matched_blocks: u64,
    pub prompt_blocks: u64,
    pub total_tree_nodes: usize,
    pub tree_ascii: String,
    pub workers: Vec<WorkerSnapshot>,
    pub last_updated: Option<Instant>,
    pub error_msg: Option<String>,
}

#[derive(Debug, Clone)]
pub struct WorkerSnapshot {
    pub id: String,
    pub is_healthy: bool,
    pub load: usize,
    pub max_concurrency: usize,
    pub kv_utilization: f64,
    pub role: String,
    pub role_badge: String,
}

/// Core interactive TUI runner.
pub struct TuiApp {
    target_url: String,
    client: reqwest::Client,
    snapshot: ClusterSnapshot,
    should_quit: bool,
}

impl TuiApp {
    pub fn new(target_url: String) -> Self {
        Self {
            target_url: target_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_millis(800))
                .build()
                .unwrap_or_default(),
            snapshot: ClusterSnapshot::default(),
            should_quit: false,
        }
    }

    /// Fetches telemetry from target gateway.
    pub async fn update_metrics(&mut self) {
        let metrics_url = format!("{}/metrics", self.target_url);
        let tree_url = format!("{}/api/v1/tree", self.target_url);

        let metrics_res = self
            .client
            .get(&metrics_url)
            .header("accept", "application/json")
            .send()
            .await;

        let tree_res = self.client.get(&tree_url).send().await;

        match metrics_res {
            Ok(resp) => {
                if resp.status().is_success() {
                    if let Ok(json) = resp.json::<Value>().await {
                        self.parse_metrics_json(&json);
                        self.snapshot.is_connected = true;
                        self.snapshot.error_msg = None;
                    }
                } else {
                    self.snapshot.is_connected = false;
                    self.snapshot.error_msg =
                        Some(format!("HTTP Error {}", resp.status().as_u16()));
                }
            }
            Err(e) => {
                self.snapshot.is_connected = false;
                self.snapshot.error_msg = Some(format!("Connection error: {}", e));
            }
        }

        if let Ok(resp) = tree_res
            && resp.status().is_success()
            && let Ok(text) = resp.text().await
        {
            self.snapshot.tree_ascii = text;
        }

        self.snapshot.last_updated = Some(Instant::now());
    }

    fn parse_metrics_json(&mut self, json: &Value) {
        if let Some(perf) = json.get("performance") {
            if let Some(ratio_str) = perf.get("cluster_cache_hit_ratio").and_then(|v| v.as_str()) {
                let clean = ratio_str.trim_end_matches('%');
                self.snapshot.hit_ratio_pct = clean.parse::<f64>().unwrap_or(0.0);
            }
            self.snapshot.total_requests = perf
                .get("total_requests")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            self.snapshot.matched_blocks = perf
                .get("total_matched_blocks")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            self.snapshot.prompt_blocks = perf
                .get("total_prompt_blocks")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
        }

        if let Some(tree) = json.get("radix_tree") {
            self.snapshot.total_tree_nodes = tree
                .get("total_nodes")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
        }

        if let Some(workers_arr) = json.get("workers").and_then(|v| v.as_array()) {
            let mut list = Vec::new();
            for w in workers_arr {
                let id = w
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let load = w.get("load").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let is_healthy = w
                    .get("is_healthy")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let max_concurrency = w
                    .get("max_concurrency")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32) as usize;
                let kv_utilization = w
                    .get("cache_utilization")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let role = w
                    .get("role")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unified")
                    .to_string();
                let role_badge = w
                    .get("role_badge")
                    .and_then(|v| v.as_str())
                    .unwrap_or("[U]")
                    .to_string();

                list.push(WorkerSnapshot {
                    id,
                    is_healthy,
                    load,
                    max_concurrency,
                    kv_utilization,
                    role,
                    role_badge,
                });
            }
            list.sort_by(|a, b| a.id.cmp(&b.id));
            self.snapshot.workers = list;
        }
    }

    /// Launches the full terminal dashboard event loop.
    pub async fn run(&mut self) -> Result<(), io::Error> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let mut last_poll = Instant::now() - Duration::from_secs(10);
        let poll_interval = Duration::from_millis(300);

        while !self.should_quit {
            if last_poll.elapsed() >= poll_interval {
                self.update_metrics().await;
                last_poll = Instant::now();
            }

            terminal.draw(|f| self.render(f))?;

            if event::poll(Duration::from_millis(50))?
                && let Event::Key(key) = event::read()?
            {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        self.should_quit = true;
                    }
                    KeyCode::Char('r') => {
                        self.update_metrics().await;
                        last_poll = Instant::now();
                    }
                    _ => {}
                }
            }
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        Ok(())
    }

    fn render(&self, f: &mut Frame) {
        let size = f.area();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(3), // Header banner
                Constraint::Length(5), // High-level KPI cards
                Constraint::Min(10),   // Split view: Radix Tree vs Worker Fleet
                Constraint::Length(3), // Footer / Keybindings
            ])
            .split(size);

        self.render_header(f, chunks[0]);
        self.render_kpis(f, chunks[1]);
        self.render_main_split(f, chunks[2]);
        self.render_footer(f, chunks[3]);
    }

    fn render_header(&self, f: &mut Frame, area: Rect) {
        let (status_text, status_style) = if self.snapshot.is_connected {
            (
                "● CONNECTED",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            (
                "○ DISCONNECTED",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            )
        };

        let title_spans = vec![
            Span::styled(
                "RADIX-GATE ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(":: Distributed KV-Cache Gateway & Topology Inspector  ["),
            Span::styled(&self.target_url, Style::default().fg(Color::Yellow)),
            Span::raw("]  Status: "),
            Span::styled(status_text, status_style),
        ];

        let header_p = Paragraph::new(Line::from(title_spans))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .alignment(Alignment::Left);

        f.render_widget(header_p, area);
    }

    fn render_kpis(&self, f: &mut Frame, area: Rect) {
        let kpi_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(area);

        // 1. Cluster Cache Hit Ratio Gauge
        let ratio = self.snapshot.hit_ratio_pct;
        let gauge_color = if ratio >= 75.0 {
            Color::Green
        } else if ratio >= 40.0 {
            Color::Yellow
        } else {
            Color::Red
        };

        let gauge = Gauge::default()
            .block(
                Block::default()
                    .title(" Global KV Hit Ratio ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            )
            .gauge_style(
                Style::default()
                    .fg(gauge_color)
                    .add_modifier(Modifier::BOLD),
            )
            .percent(ratio.clamp(0.0, 100.0) as u16)
            .label(format!("{:.1}%", ratio));
        f.render_widget(gauge, kpi_chunks[0]);

        // 2. Routed Request Counter
        let req_text = vec![
            Line::from(vec![
                Span::styled(
                    format!("{}", self.snapshot.total_requests),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" reqs"),
            ]),
            Line::from(Span::styled(
                "Total Routed",
                Style::default().fg(Color::DarkGray),
            )),
        ];
        let p_req = Paragraph::new(req_text)
            .block(
                Block::default()
                    .title(" Throughput ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            )
            .alignment(Alignment::Center);
        f.render_widget(p_req, kpi_chunks[1]);

        // 3. Matched vs Total Blocks
        let blocks_text = vec![
            Line::from(vec![
                Span::styled(
                    format!("{}", self.snapshot.matched_blocks),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" / "),
                Span::raw(format!("{}", self.snapshot.prompt_blocks)),
            ]),
            Line::from(Span::styled(
                "Matched Blocks",
                Style::default().fg(Color::DarkGray),
            )),
        ];
        let p_blocks = Paragraph::new(blocks_text)
            .block(
                Block::default()
                    .title(" Block Cache Match ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            )
            .alignment(Alignment::Center);
        f.render_widget(p_blocks, kpi_chunks[2]);

        // 4. Radix Tree Topology Nodes
        let nodes_text = vec![
            Line::from(vec![
                Span::styled(
                    format!("{}", self.snapshot.total_tree_nodes),
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" active branches"),
            ]),
            Line::from(Span::styled(
                "Prefix Tree Depth",
                Style::default().fg(Color::DarkGray),
            )),
        ];
        let p_nodes = Paragraph::new(nodes_text)
            .block(
                Block::default()
                    .title(" Radix Topology ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            )
            .alignment(Alignment::Center);
        f.render_widget(p_nodes, kpi_chunks[3]);
    }

    fn render_main_split(&self, f: &mut Frame, area: Rect) {
        let main_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
            .split(area);

        // Left: Live ASCII Radix Tree
        let tree_content = if self.snapshot.tree_ascii.is_empty() {
            "Waiting for active prefix requests...".to_string()
        } else {
            self.snapshot.tree_ascii.clone()
        };

        let tree_p = Paragraph::new(tree_content)
            .block(
                Block::default()
                    .title(" Live Radix Tree Topology ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .wrap(Wrap { trim: false });
        f.render_widget(tree_p, main_chunks[0]);

        // Right: GPU Worker Replicas Fleet
        let mut worker_lines = Vec::new();
        if self.snapshot.workers.is_empty() {
            worker_lines.push(Line::from(Span::raw("No workers detected.")));
        } else {
            for w in &self.snapshot.workers {
                let health_span = if w.is_healthy {
                    Span::styled("● OK ", Style::default().fg(Color::Green))
                } else {
                    Span::styled(
                        "▲ TRIP",
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    )
                };

                let load_pct = (w.load as f64 / w.max_concurrency.max(1) as f64 * 100.0).min(100.0);
                let kv_pct = (w.kv_utilization * 100.0).min(100.0);

                let role_style = match w.role.as_str() {
                    "Prefill" => Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                    "Decode" => Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                    _ => Style::default().fg(Color::DarkGray),
                };

                worker_lines.push(Line::from(vec![
                    health_span,
                    Span::styled(format!("{:<3} ", w.role_badge), role_style),
                    Span::styled(
                        format!("{:<10} ", w.id),
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(format!(
                        "InFlight: {:>2}/{} ({:.0}%)  ",
                        w.load, w.max_concurrency, load_pct
                    )),
                    Span::styled(
                        format!("KV: {:.1}%", kv_pct),
                        Style::default().fg(Color::Yellow),
                    ),
                ]));

                // Visual miniature bar
                let bar_width: usize = 24;
                let filled = ((kv_pct / 100.0) * bar_width as f64).round() as usize;
                let empty = bar_width.saturating_sub(filled);
                let bar_str = format!("[{}{}]", "■".repeat(filled), " ".repeat(empty));
                worker_lines.push(Line::from(vec![
                    Span::raw("   "),
                    Span::styled(bar_str, Style::default().fg(Color::Green)),
                ]));
                worker_lines.push(Line::raw(""));
            }
        }

        let worker_p = Paragraph::new(worker_lines)
            .block(
                Block::default()
                    .title(" Inference Worker Fleet (KV-Cache & In-Flight Load) ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(Color::Green)),
            )
            .wrap(Wrap { trim: false });
        f.render_widget(worker_p, main_chunks[1]);
    }

    fn render_footer(&self, f: &mut Frame, area: Rect) {
        let footer_spans = vec![
            Span::styled(
                " [q] ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("Quit  | "),
            Span::styled(
                " [r] ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("Force Refresh  | "),
            Span::styled(" RadixGate v0.1.0 ", Style::default().fg(Color::Cyan)),
        ];

        let footer_p = Paragraph::new(Line::from(footer_spans))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            )
            .alignment(Alignment::Center);

        f.render_widget(footer_p, area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tui_metrics_parsing() {
        let mut app = TuiApp::new("http://localhost:8080".to_string());
        let mock_json = json!({
            "performance": {
                "cluster_cache_hit_ratio": "86.5%",
                "total_matched_blocks": 1500,
                "total_prompt_blocks": 2000,
                "total_requests": 150
            },
            "radix_tree": {
                "total_nodes": 42
            },
            "workers": [
                {
                    "id": "worker-01",
                    "load": 4,
                    "max_concurrency": 32,
                    "is_healthy": true,
                    "role": "Prefill",
                    "role_badge": "[P]",
                    "cache_utilization": 0.65
                },
                {
                    "id": "worker-02",
                    "load": 0,
                    "max_concurrency": 32,
                    "is_healthy": false,
                    "role": "Decode",
                    "role_badge": "[D]",
                    "cache_utilization": 0.10
                }
            ]
        });

        app.parse_metrics_json(&mock_json);

        assert!((app.snapshot.hit_ratio_pct - 86.5).abs() < 1e-4);
        assert_eq!(app.snapshot.total_requests, 150);
        assert_eq!(app.snapshot.matched_blocks, 1500);
        assert_eq!(app.snapshot.total_tree_nodes, 42);
        assert_eq!(app.snapshot.workers.len(), 2);
        assert_eq!(app.snapshot.workers[0].id, "worker-01");
        assert!(app.snapshot.workers[0].is_healthy);
        assert_eq!(app.snapshot.workers[1].id, "worker-02");
        assert!(!app.snapshot.workers[1].is_healthy);
    }
}
