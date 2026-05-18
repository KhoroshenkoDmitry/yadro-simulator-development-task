import argparse
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).parent))
 
from src import HOURS_PER_DAY, EnergyScheduler  # noqa: E402
from src.io_utils import load_problem, save_schedule  # noqa: E402
from src.models import ProblemInstance, Schedule  # noqa: E402
 
# --- Table formatting constants -----------------------------------------------
 
HOUR_COLUMN_WIDTH = 4
COST_COLUMN_WIDTH = 6
ACTIVE_COLUMN_WIDTH = 27
TABLE_TOTAL_WIDTH = 78
NO_ACTIVE_PLACEHOLDER = "(none)"
NO_UNSERVED_PLACEHOLDER = "-"
 
 
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Energy generator scheduler")
    parser.add_argument(
        "input",
        type=Path,
        help="Path to JSON file with the problem instance",
    )
    parser.add_argument(
        "--backend",
        choices=["scipy", "pulp"],
        default="scipy",
        help="MILP solver backend (default: scipy)",
    )
    parser.add_argument(
        "--strategy",
        choices=["per_hour", "monolithic"],
        default="per_hour",
        help="Decomposition strategy (default: per_hour)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed hourly output",
    )
    return parser
 
 
def _print_summary(
    input_path: Path,
    backend: str,
    strategy: str,
    problem: ProblemInstance,
    schedule: Schedule,
) -> None:
    total_possible = HOURS_PER_DAY * problem.num_consumers
    served_pct = 100 * schedule.served_count / total_possible if total_possible else 0.0
    print(f"=== Solved {input_path} ===")
    print(f"Backend: {backend}, strategy: {strategy}")
    print(f"Total cost: {schedule.total_cost:.2f}")
    print(
        f"Served: {schedule.served_count}/{total_possible} ({served_pct:.1f}%)"
    )
 
 
def _print_hourly_table(problem: ProblemInstance, schedule: Schedule) -> None:
    print()
    header = (
        f"{'Hour':>{HOUR_COLUMN_WIDTH}} | "
        f"{'Cost':>{COST_COLUMN_WIDTH}} | "
        f"{'Active generators':<{ACTIVE_COLUMN_WIDTH}} | Unserved"
    )
    print(header)
    print("-" * TABLE_TOTAL_WIDTH)
 
    for hour in range(HOURS_PER_DAY):
        active_names = [
            problem.generators[generator_idx].name
            for generator_idx, is_active in enumerate(schedule.active[hour])
            if is_active
        ]
        unserved_names = [
            problem.consumers[consumer_idx].name
            for consumer_idx in schedule.hourly_unserved[hour]
        ]
        active_cell = ", ".join(active_names) or NO_ACTIVE_PLACEHOLDER
        unserved_cell = ", ".join(unserved_names) if unserved_names else NO_UNSERVED_PLACEHOLDER
        print(
            f"{hour:>{HOUR_COLUMN_WIDTH}d} | "
            f"{schedule.hourly_cost[hour]:>{COST_COLUMN_WIDTH}.1f} | "
            f"{active_cell:<{ACTIVE_COLUMN_WIDTH}s} | "
            f"{unserved_cell}"
        )
 
 
def main() -> int:
    args = _build_arg_parser().parse_args()
 
    problem = load_problem(args.input)
    scheduler = EnergyScheduler(backend=args.backend, strategy=args.strategy)
    schedule = scheduler.solve(problem)
 
    _print_summary(args.input, args.backend, args.strategy, problem, schedule)
    if not args.quiet:
        _print_hourly_table(problem, schedule)
 
    if args.output is not None:
        save_schedule(problem, schedule, args.output)
        print(f"\nSchedule saved to {args.output}")
 
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
 
