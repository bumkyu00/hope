"""Day simulator: single day interaction + sleep cycle."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from dreamlora.config import ExperimentConfig
from dreamlora.data.formats import KST
from dreamlora.data.memory_store import MemoryStore
from dreamlora.dream.mixer import DreamPool
from dreamlora.eval.benchmark import run_benchmark, BenchmarkResults
from dreamlora.simulation.scenario_bank import DayScenario, get_scenario
from dreamlora.sleep.orchestrator import SleepOrchestrator
from dreamlora.sleep.state import StateManager

logger = logging.getLogger(__name__)


class DaySimulator:
    """Simulates a single day: add spans from scenario, run sleep cycle, evaluate."""

    def __init__(
        self,
        model,
        tokenizer,
        config: ExperimentConfig,
        memory_store: MemoryStore,
        state_manager: StateManager,
        dream_pool: DreamPool,
        base_date: datetime | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.memory_store = memory_store
        self.state_manager = state_manager
        self.dream_pool = dream_pool
        self.base_date = base_date or datetime(2026, 3, 1, tzinfo=KST)

    def simulate_day(self, day: int, evaluate: bool = True) -> dict:
        """Simulate a single day.

        Args:
            day: day number (1-30)
            evaluate: whether to run evaluation after sleep

        Returns:
            dict with day results
        """
        scenario = get_scenario(day)
        if scenario is None:
            logger.warning(f"No scenario for day {day}")
            return {"day": day, "status": "skipped"}

        logger.info(f"=== Day {day}: {scenario.description} ({scenario.phase}) ===")

        # 1. Add new spans
        day_time = self.base_date + timedelta(days=day - 1, hours=10)
        new_span_ids = []

        for i, span_data in enumerate(scenario.new_spans):
            ts = day_time + timedelta(hours=i)
            span = self.memory_store.add(
                content=span_data["content"],
                level=span_data["level"],
                sentiment=span_data["sentiment"],
                tags=span_data.get("tags", []),
                timestamp=ts,
                span_id=f"mem_day{day:02d}_{i:02d}",
            )
            new_span_ids.append(span.span_id)
            logger.info(f"  Added span: {span.span_id} (level={span_data['level']})")

        # 2. Sleep cycle
        orchestrator = SleepOrchestrator(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            memory_store=self.memory_store,
            state_manager=self.state_manager,
            dream_pool=self.dream_pool,
        )
        cycle_result = orchestrator.run_cycle(new_span_ids=new_span_ids)

        # 3. Evaluate (optional)
        eval_result = None
        if evaluate:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            benchmark = run_benchmark(
                self.model, self.tokenizer, self.memory_store,
                mmlu_num_questions=self.config.eval.mmlu_num_questions,
                mmlu_seed=self.config.eval.mmlu_seed,
                device=device,
            )
            eval_result = benchmark.summary()
            logger.info(f"  Eval: {eval_result}")

        return {
            "day": day,
            "phase": scenario.phase,
            "description": scenario.description,
            "new_spans": new_span_ids,
            "cycle_result": cycle_result,
            "eval_result": eval_result,
            "total_spans": len(self.memory_store),
        }


def run_30day_simulation(
    model,
    tokenizer,
    config: ExperimentConfig,
    output_dir: str | Path,
    evaluate_every: int = 5,
) -> list[dict]:
    """Run the full 30-day simulation.

    Args:
        evaluate_every: run full evaluation every N days (default 5)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    memory_store = MemoryStore(output_dir / "memories.jsonl")
    state_manager = StateManager(output_dir / "state.json")
    dream_pool = DreamPool()

    simulator = DaySimulator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        memory_store=memory_store,
        state_manager=state_manager,
        dream_pool=dream_pool,
    )

    all_results = []
    for day in range(1, 31):
        evaluate = (day % evaluate_every == 0) or day == 30
        result = simulator.simulate_day(day, evaluate=evaluate)
        all_results.append(result)

        # Save incremental results
        with open(output_dir / "simulation_log.jsonl", "a") as f:
            f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    # Save final summary
    with open(output_dir / "simulation_results.json", "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    return all_results
