"""
A/B Prompt Evaluation — tests 15 prompt configurations and tracks results in LangSmith.
Measures accuracy, latency, and insight quality across prompt variants.
"""

import os
import json
import time
import statistics
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# ── 15 Prompt Configurations ──────────────────────────────────────────────────

PROMPT_CONFIGS = {
    "v01_baseline": {
        "description": "Baseline: direct instruction, no CoT",
        "system_suffix": "",
        "cot": False,
        "tool_order": ["sql", "rag", "calc"],
    },
    "v02_cot": {
        "description": "Chain-of-Thought: explicit step numbering",
        "system_suffix": "\nThink step by step: (1) identify needed data, (2) retrieve, (3) calculate, (4) synthesize.",
        "cot": True,
        "tool_order": ["sql", "rag", "calc"],
    },
    "v03_persona": {
        "description": "Expert persona injection",
        "system_suffix": "\nYou have 10 years of paid media experience at a Fortune 500 retailer.",
        "cot": False,
        "tool_order": ["sql", "rag", "calc"],
    },
    "v04_cot_persona": {
        "description": "CoT + expert persona",
        "system_suffix": "\nYou have 10 years of paid media experience. Think step by step before answering.",
        "cot": True,
        "tool_order": ["sql", "rag", "calc"],
    },
    "v05_sql_first": {
        "description": "Force SQL retrieval before RAG",
        "system_suffix": "\nAlways start with sql_query_tool to ground your answer in data before using rag_context_tool.",
        "cot": False,
        "tool_order": ["sql", "rag", "calc"],
    },
    "v06_rag_first": {
        "description": "Force RAG context before SQL",
        "system_suffix": "\nAlways retrieve context with rag_context_tool before querying data.",
        "cot": False,
        "tool_order": ["rag", "sql", "calc"],
    },
    "v07_concise": {
        "description": "Conciseness constraint",
        "system_suffix": "\nAnswer in 3 sentences max. Lead with the key metric, then the insight, then the recommendation.",
        "cot": False,
        "tool_order": ["sql", "rag", "calc"],
    },
    "v08_structured_output": {
        "description": "Force structured JSON-like output",
        "system_suffix": "\nFormat your answer as: METRIC: <value> | INSIGHT: <1 sentence> | ACTION: <1 recommendation>",
        "cot": False,
        "tool_order": ["sql", "rag", "calc"],
    },
    "v09_cot_structured": {
        "description": "CoT + structured output",
        "system_suffix": "\nThink step by step, then format: METRIC: <value> | INSIGHT: <1 sentence> | ACTION: <1 recommendation>",
        "cot": True,
        "tool_order": ["sql", "rag", "calc"],
    },
    "v10_few_shot": {
        "description": "Few-shot examples in system prompt",
        "system_suffix": (
            "\nExample Q: 'Which campaign has the highest ROAS this month?'"
            "\nExample A: METRIC: Campaign B, ROAS 5.2x | INSIGHT: 2.1x above portfolio average | ACTION: Increase budget by 20%."
        ),
        "cot": False,
        "tool_order": ["sql", "rag", "calc"],
    },
    "v11_cot_few_shot": {
        "description": "CoT + few-shot",
        "system_suffix": (
            "\nThink step by step."
            "\nExample Q: 'Why did CTR drop last week?'"
            "\nExample A: Step 1: Query daily CTR. Step 2: Compare week-over-week. Step 3: Check industry benchmarks. Conclusion: 1.2% vs 2.1% prior week — ad fatigue, recommend creative refresh."
        ),
        "cot": True,
        "tool_order": ["sql", "rag", "calc"],
    },
    "v12_negative": {
        "description": "Negative constraints (what NOT to do)",
        "system_suffix": "\nDo NOT speculate without data. Do NOT omit concrete numbers. Do NOT exceed 150 words.",
        "cot": False,
        "tool_order": ["sql", "rag", "calc"],
    },
    "v13_multi_tool": {
        "description": "Explicit multi-tool reasoning chain",
        "system_suffix": "\nFor every answer: (1) sql_query_tool for metrics, (2) rag_context_tool for benchmarks, (3) calculate_metric_tool for derived KPIs.",
        "cot": True,
        "tool_order": ["sql", "rag", "calc"],
    },
    "v14_role_play": {
        "description": "Adversarial role-play (Devil's Advocate)",
        "system_suffix": "\nAfter giving your insight, briefly challenge your own conclusion: what could make it wrong?",
        "cot": True,
        "tool_order": ["sql", "rag", "calc"],
    },
    "v15_optimal": {
        "description": "Best-of-breed: CoT + persona + structured + negative constraints",
        "system_suffix": (
            "\nYou are a senior paid media analyst. Think step by step."
            "\nDo NOT speculate. Always cite numeric evidence."
            "\nFormat: METRIC: <value> | INSIGHT: <1 sentence> | ACTION: <1 recommendation>"
        ),
        "cot": True,
        "tool_order": ["sql", "rag", "calc"],
    },
}


# ── Evaluation Dataset ────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    {
        "id": "eval-001",
        "query": "Which campaign had the highest ROAS last month and by how much did it beat the portfolio average?",
        "expected_contains": ["roas", "campaign", "average"],
        "category": "performance_ranking",
    },
    {
        "id": "eval-002",
        "query": "Why might CTR have dropped for the Search channel last week compared to the previous week?",
        "expected_contains": ["ctr", "search", "week"],
        "category": "root_cause",
    },
    {
        "id": "eval-003",
        "query": "What is the total spend and revenue breakdown by channel this quarter?",
        "expected_contains": ["spend", "revenue", "channel"],
        "category": "aggregation",
    },
    {
        "id": "eval-004",
        "query": "Which ad groups have CPC above the industry benchmark and what action should we take?",
        "expected_contains": ["cpc", "benchmark", "action"],
        "category": "anomaly_detection",
    },
    {
        "id": "eval-005",
        "query": "What is the conversion rate trend over the last 30 days and what does it indicate?",
        "expected_contains": ["conversion", "trend", "30"],
        "category": "trend_analysis",
    },
]


# ── Result Types ──────────────────────────────────────────────────────────────

@dataclass
class PromptEvalResult:
    prompt_version: str
    eval_id: str
    latency_ms: float
    answer_length: int
    contains_score: float        # 0–1: fraction of expected_contains keywords found
    tool_calls: int
    error: Optional[str] = None


@dataclass
class PromptVersionSummary:
    prompt_version: str
    description: str
    n_evals: int
    avg_latency_ms: float
    p95_latency_ms: float
    avg_contains_score: float
    avg_tool_calls: float
    error_rate: float
    overall_score: float         # weighted composite


# ── Evaluator ─────────────────────────────────────────────────────────────────

class ABPromptEvaluator:
    """
    Runs the 15 prompt configs against the eval dataset and reports winners.
    Integrates with LangSmith for experiment tracking when LANGCHAIN_API_KEY is set.
    """

    def __init__(self, run_fn=None):
        """
        Args:
            run_fn: Callable(query, prompt_config) -> dict with keys {answer, latency_ms, tool_calls_made}
                    Defaults to mock runner for offline testing.
        """
        self._run = run_fn or self._mock_run
        self._results: List[PromptEvalResult] = []
        self._langsmith_enabled = bool(os.environ.get("LANGCHAIN_API_KEY"))

    # ── LangSmith integration ────────────────────────────────────────────────

    def _log_to_langsmith(self, run_data: dict):
        if not self._langsmith_enabled:
            return
        try:
            from langsmith import Client
            client = Client()
            client.create_run(
                name=f"ab_eval_{run_data['prompt_version']}",
                run_type="chain",
                inputs={"query": run_data["query"], "prompt_config": run_data["config"]},
                outputs={"answer": run_data["answer"]},
                extra={"latency_ms": run_data["latency_ms"], "contains_score": run_data["contains_score"]},
            )
        except Exception as e:
            logger.warning(f"LangSmith logging failed: {e}")

    # ── Mock runner (used when LLM backend is unavailable) ───────────────────

    def _mock_run(self, query: str, config: dict) -> dict:
        """Simulates LLM responses with realistic variance for offline testing."""
        import random
        rng = random.Random(hash(config["description"] + query))

        base_latency = {
            "v01_baseline": 85, "v02_cot": 120, "v03_persona": 90,
            "v04_cot_persona": 130, "v05_sql_first": 95, "v06_rag_first": 100,
            "v07_concise": 80, "v08_structured_output": 88, "v09_cot_structured": 125,
            "v10_few_shot": 105, "v11_cot_few_shot": 140, "v12_negative": 92,
            "v13_multi_tool": 145, "v14_role_play": 160, "v15_optimal": 135,
        }.get(config.get("version_key", ""), 100)

        latency = base_latency + rng.gauss(0, 12)
        words = query.lower().split()
        answer_tokens = rng.randint(40, 180)
        tool_calls = rng.randint(1, 3)

        return {
            "answer": f"[Mock] Insight for '{query[:40]}...' using {config['description']}. Metric: ROAS 4.2x. Insight: Above 3.8x avg. Action: Increase budget 15%.",
            "latency_ms": max(50, latency),
            "tool_calls_made": tool_calls,
        }

    # ── Evaluation loop ───────────────────────────────────────────────────────

    def evaluate_version(self, version_key: str, config: dict) -> List[PromptEvalResult]:
        config_with_key = {**config, "version_key": version_key}
        version_results = []

        for item in EVAL_QUESTIONS:
            try:
                result = self._run(item["query"], config_with_key)
                answer = result.get("answer", "")
                contains = sum(
                    1 for kw in item["expected_contains"]
                    if kw.lower() in answer.lower()
                ) / len(item["expected_contains"])

                er = PromptEvalResult(
                    prompt_version=version_key,
                    eval_id=item["id"],
                    latency_ms=result.get("latency_ms", 999),
                    answer_length=len(answer),
                    contains_score=contains,
                    tool_calls=result.get("tool_calls_made", 0),
                )
                self._results.append(er)
                version_results.append(er)

                self._log_to_langsmith({
                    "prompt_version": version_key,
                    "query": item["query"],
                    "config": config,
                    "answer": answer,
                    "latency_ms": er.latency_ms,
                    "contains_score": contains,
                })

            except Exception as e:
                er = PromptEvalResult(
                    prompt_version=version_key,
                    eval_id=item["id"],
                    latency_ms=999,
                    answer_length=0,
                    contains_score=0.0,
                    tool_calls=0,
                    error=str(e),
                )
                self._results.append(er)
                version_results.append(er)

        return version_results

    def run_all(self) -> List[PromptVersionSummary]:
        logger.info(f"Running A/B eval across {len(PROMPT_CONFIGS)} prompt configs × {len(EVAL_QUESTIONS)} questions")
        summaries = []
        for vkey, config in PROMPT_CONFIGS.items():
            self.evaluate_version(vkey, config)
            summaries.append(self._summarize(vkey, config["description"]))

        summaries.sort(key=lambda s: s.overall_score, reverse=True)
        return summaries

    def _summarize(self, version_key: str, description: str) -> PromptVersionSummary:
        rows = [r for r in self._results if r.prompt_version == version_key]
        valid = [r for r in rows if not r.error]
        n = len(rows)

        latencies = [r.latency_ms for r in valid] or [0]
        latencies_sorted = sorted(latencies)
        p95 = latencies_sorted[int(0.95 * len(latencies_sorted))] if latencies_sorted else 0

        avg_contains = statistics.mean(r.contains_score for r in valid) if valid else 0
        avg_tools = statistics.mean(r.tool_calls for r in valid) if valid else 0
        error_rate = sum(1 for r in rows if r.error) / n if n else 0

        # Composite: 50% accuracy, 30% latency (inverted), 20% tool efficiency
        latency_score = max(0, 1 - (statistics.mean(latencies) / 200))  # 200 ms = target
        tool_score = min(1, avg_tools / 2)  # 2 tools = ideal
        overall = 0.5 * avg_contains + 0.3 * latency_score + 0.2 * tool_score

        return PromptVersionSummary(
            prompt_version=version_key,
            description=description,
            n_evals=n,
            avg_latency_ms=round(statistics.mean(latencies), 1),
            p95_latency_ms=round(p95, 1),
            avg_contains_score=round(avg_contains, 3),
            avg_tool_calls=round(avg_tools, 2),
            error_rate=round(error_rate, 3),
            overall_score=round(overall, 4),
        )

    def print_leaderboard(self, summaries: List[PromptVersionSummary]):
        print("\n" + "=" * 90)
        print(f"{'PROMPT A/B EVALUATION LEADERBOARD':^90}")
        print("=" * 90)
        print(f"{'Rank':<5} {'Version':<22} {'Score':>7} {'Acc':>7} {'P95 ms':>8} {'Tools':>7} {'Errors':>7}")
        print("-" * 90)
        for i, s in enumerate(summaries, 1):
            print(
                f"{i:<5} {s.prompt_version:<22} {s.overall_score:>7.4f}"
                f" {s.avg_contains_score:>7.3f} {s.p95_latency_ms:>8.1f}"
                f" {s.avg_tool_calls:>7.2f} {s.error_rate:>7.3f}"
            )
        print("=" * 90)
        winner = summaries[0]
        print(f"\n✅ WINNER: {winner.prompt_version} — {winner.description}")
        print(f"   Score: {winner.overall_score:.4f} | Accuracy: {winner.avg_contains_score:.1%} | P95: {winner.p95_latency_ms:.0f} ms")

    def save_results(self, path: str = "results/ab_eval_results.json"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_configs": len(PROMPT_CONFIGS),
            "n_questions": len(EVAL_QUESTIONS),
            "raw_results": [asdict(r) for r in self._results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved to {path}")
