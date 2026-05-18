"""I/O utilities: load problem instances from JSON, write schedules to JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    HOURS_PER_DAY,
    Consumer,
    DieselGenerator,
    Generator,
    ProblemInstance,
    Schedule,
    SolarGenerator,
)

# JSON discriminator values for the "kind" field on generator entries.
_KIND_DIESEL = "diesel"
_KIND_SOLAR = "solar"


def load_problem(path: str | Path) -> ProblemInstance:
    """Load a :class:`ProblemInstance` from a JSON file.

    Expected JSON structure::

        {
            "consumers": [
                {"name": "...", "demand": [d_0, ..., d_23]},
                ...
            ],
            "generators": [
                {"name": "...", "kind": "diesel",
                 "hourly_output": 100.0, "hourly_cost": 50.0},
                {"name": "...", "kind": "solar",
                 "hourly_output_profile": [o_0, ..., o_23],
                 "hourly_cost_profile":   [c_0, ..., c_23]},
                ...
            ]
        }
    """
    with open(path, encoding="utf-8") as handle:
        raw_data = json.load(handle)

    consumers = tuple(_parse_consumer(entry) for entry in raw_data["consumers"])
    generators = tuple(_parse_generator(entry) for entry in raw_data["generators"])
    return ProblemInstance(consumers=consumers, generators=generators)


def _parse_consumer(entry: dict[str, Any]) -> Consumer:
    return Consumer(
        name=entry["name"],
        demand=tuple(float(value) for value in entry["demand"]),
    )


def _parse_generator(entry: dict[str, Any]) -> Generator:
    kind = entry["kind"]
    name = entry["name"]
    if kind == _KIND_DIESEL:
        return DieselGenerator(
            name=name,
            hourly_output=float(entry["hourly_output"]),
            hourly_cost=float(entry["hourly_cost"]),
        )
    if kind == _KIND_SOLAR:
        return SolarGenerator(
            name=name,
            hourly_output_profile=tuple(float(v) for v in entry["hourly_output_profile"]),
            hourly_cost_profile=tuple(float(v) for v in entry["hourly_cost_profile"]),
        )
    raise ValueError(f"Unknown generator kind: {kind!r} (for generator {name!r})")


def schedule_to_dict(
    problem: ProblemInstance,
    schedule: Schedule,
) -> dict[str, Any]:
    """Convert a :class:`Schedule` to a JSON-serializable dict."""
    return {
        "summary": {
            "total_cost": schedule.total_cost,
            "served_count": schedule.served_count,
            "total_possible": HOURS_PER_DAY * problem.num_consumers,
        },
        "hourly_schedule": [
            _hour_to_dict(hour, problem, schedule) for hour in range(HOURS_PER_DAY)
        ],
    }


def _hour_to_dict(
    hour: int,
    problem: ProblemInstance,
    schedule: Schedule,
) -> dict[str, Any]:
    active_generator_names = [
        problem.generators[generator_idx].name
        for generator_idx, is_active in enumerate(schedule.active[hour])
        if is_active
    ]
    served_consumer_names = [
        problem.consumers[consumer_idx].name
        for consumer_idx, is_served in enumerate(schedule.served[hour])
        if is_served
    ]
    unserved_consumer_names = [
        problem.consumers[consumer_idx].name for consumer_idx in schedule.hourly_unserved[hour]
    ]
    return {
        "hour": hour,
        "active_generators": active_generator_names,
        "served_consumers": served_consumer_names,
        "unserved_consumers": unserved_consumer_names,
        "cost": schedule.hourly_cost[hour],
    }


def save_schedule(
    problem: ProblemInstance,
    schedule: Schedule,
    path: str | Path,
) -> None:
    """Serialize a :class:`Schedule` to a JSON file."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            schedule_to_dict(problem, schedule),
            handle,
            indent=2,
            ensure_ascii=False,
        )
