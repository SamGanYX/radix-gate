"""
Empirical Multi-Turn Agent Benchmark: NGINX Round-Robin vs. RadixGate on Real vLLM Clusters.

Measures:
1. Client-perceived Time-To-First-Token (TTFT) via true SSE streaming (`stream=True`).
2. P50, P95, P99 TTFT and End-to-End latency.
3. Actual physical GPU KV-cache prefix hit rates scraped directly from each vLLM instance's `/metrics` endpoint.
"""

import argparse
import asyncio
import json
import statistics
import time
from typing import Dict, List, Optional, Tuple
import httpx


def build_agent_system_prompt(agent_idx: int, approx_words: int = 2200) -> str:
    """
    Constructs a realistic ~3,000-token system prompt + tool schema unique to an agent persona,
    large enough to span ~45+ 64-token PagedAttention blocks on vLLM.
    """
    header = (
        f"You are Agent Persona #{agent_idx}, a principal distributed systems engineer "
        "specializing in storage engines, consensus protocols, and GPU inference runtimes.\n"
        "Available Tools:\n"
        "- read_file(path: str, start_line: int, end_line: int) -> str\n"
        "- run_profiler(pid: int, duration_sec: int, flamegraph: bool) -> dict\n"
        "- query_telemetry(metric: str, window_sec: int) -> list\n\n"
        "Reference Architecture Specification & Runbook:\n"
    )
    paragraph = (
        f"[Context Block Agent-{agent_idx:02d}] In replicated log systems and disaggregated KV-cache "
        "serving architectures, deterministic prefix placement minimizes redundant HBM bandwidth "
        "consumption during prefill matrix multiplication. Always verify lock ordering between "
        "worker pool state and radix trie nodes, inspect P99 tail latency spikes under memory "
        "eviction pressure, and validate that streaming connection guards release only upon EOF. "
    )
    repeats = max(1, approx_words // len(paragraph.split()))
    body = "\n".join(f"Section {i} (Agent {agent_idx}): {paragraph}" for i in range(repeats))
    return header + body


def parse_vllm_prefix_metrics(metrics_text: str) -> Tuple[float, float]:
    """
    Parses `vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total`
    (or `vllm:gpu_prefix_cache_queries` / `hits` across vLLM versions) from Prometheus text.
    """
    queries = 0.0
    hits = 0.0
    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{")[0]
        try:
            val = float(parts[-1])
        except ValueError:
            continue
        if name in ("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total", "vllm:prefix_cache_queries"):
            queries += val
        elif name in ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total", "vllm:prefix_cache_hits"):
            hits += val
    return queries, hits


async def scrape_cluster_vllm_counters(worker_urls: List[str]) -> Tuple[float, float]:
    """Sums prefix cache queries and hits across all physical vLLM workers."""
    total_q = 0.0
    total_h = 0.0
    async with httpx.AsyncClient(timeout=5.0) as client:
        for url in worker_urls:
            try:
                resp = await client.get(f"{url.rstrip('/')}/metrics")
                if resp.status_code == 200:
                    q, h = parse_vllm_prefix_metrics(resp.text)
                    total_q += q
                    total_h += h
            except Exception as exc:
                print(f"[warn] Could not scrape {url}/metrics: {exc}")
    return total_q, total_h


async def send_streaming_turn(
    client: httpx.AsyncClient,
    target_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> Tuple[float, float, str]:
    """
    Sends a streaming `/v1/chat/completions` request and returns:
    (ttft_ms, total_ms, assistant_text)
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t_start = time.perf_counter()
    ttft_ms: Optional[float] = None
    chunks: List[str] = []

    async with client.stream(
        "POST",
        f"{target_url.rstrip('/')}/v1/chat/completions",
        json=payload,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            if content:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000.0
                chunks.append(content)

    total_ms = (time.perf_counter() - t_start) * 1000.0
    if ttft_ms is None:
        ttft_ms = total_ms
    return ttft_ms, total_ms, "".join(chunks)


async def run_target_benchmark(
    label: str,
    target_url: str,
    worker_urls: List[str],
    model: str,
    num_sessions: int,
    turns_per_session: int,
    num_agents: int,
    max_tokens: int,
    concurrency: int,
    salt: str,
) -> Dict:
    print(f"\n--- Running workload against {label} ({target_url}) ---")
    system_prompts = [
        f"[Run:{salt}]\n" + build_agent_system_prompt(i)
        for i in range(num_agents)
    ]

    q_before, h_before = await scrape_cluster_vllm_counters(worker_urls)
    sem = asyncio.Semaphore(concurrency)
    ttfts: List[float] = []
    totals: List[float] = []
    errors = 0
    total_expected = num_sessions * turns_per_session

    async def run_session(session_id: int, client: httpx.AsyncClient):
        nonlocal errors
        sys_prompt = system_prompts[session_id % num_agents]
        history = [{"role": "system", "content": sys_prompt}]
        for turn in range(turns_per_session):
            history.append({
                "role": "user",
                "content": f"Session {session_id} turn {turn + 1}: summarize the key invariant concisely.",
            })
            async with sem:
                try:
                    ttft, total, reply = await send_streaming_turn(
                        client, target_url, model, history, max_tokens
                    )
                    ttfts.append(ttft)
                    totals.append(total)
                    done = len(ttfts)
                    if done % 20 == 0 or done == total_expected:
                        print(
                            f"  [{label}] {done}/{total_expected} requests "
                            f"(running mean TTFT: {statistics.mean(ttfts):.1f} ms)",
                            flush=True,
                        )
                    history.append({"role": "assistant", "content": reply or "Acknowledged."})
                except Exception as exc:
                    errors += 1
                    print(f"[error] {label} session {session_id} turn {turn + 1}: {exc}", flush=True)

    t_wall_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        await asyncio.gather(*(run_session(s, client) for s in range(num_sessions)))
    wall_sec = time.perf_counter() - t_wall_start

    q_after, h_after = await scrape_cluster_vllm_counters(worker_urls)
    delta_q = max(0.0, q_after - q_before)
    delta_h = max(0.0, h_after - h_before)
    vllm_hit_pct = (delta_h / delta_q * 100.0) if delta_q > 0 else 0.0

    ttfts_sorted = sorted(ttfts)

    def pct(vals: List[float], p: float) -> float:
        if not vals:
            return 0.0
        idx = min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1))))
        return vals[idx]

    result = {
        "label": label,
        "requests": len(ttfts),
        "errors": errors,
        "wall_sec": round(wall_sec, 2),
        "rps": round(len(ttfts) / wall_sec, 2) if wall_sec > 0 else 0.0,
        "vllm_prefix_hit_pct": round(vllm_hit_pct, 2),
        "mean_ttft_ms": round(statistics.mean(ttfts), 2) if ttfts else 0.0,
        "p50_ttft_ms": round(pct(ttfts_sorted, 50), 2),
        "p95_ttft_ms": round(pct(ttfts_sorted, 95), 2),
        "p99_ttft_ms": round(pct(ttfts_sorted, 99), 2),
    }
    print(json.dumps(result, indent=2))
    return result


async def main_async(args):
    workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    # Distinct salts ensure neither router inherits warm prefix blocks from the other's run
    nginx_res = await run_target_benchmark(
        label="NGINX (Round-Robin)",
        target_url=args.nginx_url,
        worker_urls=workers,
        model=args.model,
        num_sessions=args.sessions,
        turns_per_session=args.turns,
        num_agents=args.agents,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        salt=f"nginx-{int(time.time())}",
    )
    radix_res = await run_target_benchmark(
        label="RadixGate (Prefix-Aware)",
        target_url=args.radix_url,
        worker_urls=workers,
        model=args.model,
        num_sessions=args.sessions,
        turns_per_session=args.turns,
        num_agents=args.agents,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        salt=f"radix-{int(time.time())}",
    )

    speedup = (
        round(nginx_res["mean_ttft_ms"] / radix_res["mean_ttft_ms"], 2)
        if radix_res["mean_ttft_ms"] > 0
        else 0.0
    )

    print("\n### Real vLLM Cluster Results (Copy into README.md)\n")
    print("| Router | Requests | vLLM GPU Cache Hit Rate | Mean TTFT | P50 TTFT | P95 TTFT | P99 TTFT | Throughput |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in (nginx_res, radix_res):
        print(
            f"| **{r['label']}** | {r['requests']} | {r['vllm_prefix_hit_pct']:.1f}% | "
            f"{r['mean_ttft_ms']:.1f} ms | {r['p50_ttft_ms']:.1f} ms | "
            f"{r['p95_ttft_ms']:.1f} ms | {r['p99_ttft_ms']:.1f} ms | {r['rps']:.1f} req/s |"
        )
    print(f"\nObserved Mean TTFT Speedup: **{speedup:.2f}x**")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"nginx": nginx_res, "radixgate": radix_res, "speedup": speedup}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Benchmark NGINX vs RadixGate on real vLLM workers")
    parser.add_argument("--nginx-url", default="http://127.0.0.1:8090")
    parser.add_argument("--radix-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--workers",
        default="http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003,http://127.0.0.1:8004",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--out", default="vllm_benchmark_results.json")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
