"""Generate two test cases per the task statement: surplus and deficit."""

import json
import math
from pathlib import Path
from typing import Any

HOURS_PER_DAY = 24

# --- Solar profile parameters -------------------------------------------------

DEFAULT_SOLAR_PEAK_HOUR = 13
DEFAULT_SOLAR_WIDTH = 4.0
# Output below this fraction of the peak is clamped to zero (night).
SOLAR_NIGHT_THRESHOLD = 0.05
# Cost per kWh of solar generation (very cheap relative to diesel).
SOLAR_COST_PER_KWH = 0.1

# --- Consumption profile parameters -------------------------------------------
# Each tuple: (start_hour_inclusive, end_hour_exclusive, multiplier_of_base).
# Multipliers chosen to mimic a residential load curve: low at night,
# two peaks (morning and evening), modest daytime usage.

NIGHT_MULTIPLIER = 0.4
DAYTIME_MULTIPLIER = 0.7
LATE_EVENING_MULTIPLIER = 0.6

NIGHT_END_HOUR = 6
MORNING_PEAK_END_HOUR = 9
DAYTIME_END_HOUR = 17
EVENING_PEAK_END_HOUR = 22


def daylight_profile(
    peak: float,
    peak_hour: int = DEFAULT_SOLAR_PEAK_HOUR,
    width: float = DEFAULT_SOLAR_WIDTH,
) -> list[float]:
    """Smooth bell-shaped solar profile: zero at night, maximum around ``peak_hour``."""
    night_cutoff = peak * SOLAR_NIGHT_THRESHOLD
    profile: list[float] = []
    for hour in range(HOURS_PER_DAY):
        offset = (hour - peak_hour) / width
        value = peak * math.exp(-offset * offset)
        profile.append(round(value, 1) if value > night_cutoff else 0.0)
    return profile


def consumption_profile(
    base: float,
    evening_peak: float = 1.6,
    morning_peak: float = 1.2,
) -> list[float]:
    """Realistic residential consumption: low at night, peaks morning and evening."""
    intervals: list[tuple[int, int, float]] = [
        (0, NIGHT_END_HOUR, NIGHT_MULTIPLIER),
        (NIGHT_END_HOUR, MORNING_PEAK_END_HOUR, morning_peak),
        (MORNING_PEAK_END_HOUR, DAYTIME_END_HOUR, DAYTIME_MULTIPLIER),
        (DAYTIME_END_HOUR, EVENING_PEAK_END_HOUR, evening_peak),
        (EVENING_PEAK_END_HOUR, HOURS_PER_DAY, LATE_EVENING_MULTIPLIER),
    ]
    profile: list[float] = []
    for hour in range(HOURS_PER_DAY):
        multiplier = next(m for start, end, m in intervals if start <= hour < end)
        profile.append(round(base * multiplier, 1))
    return profile


def _solar_generator(
    name: str,
    peak: float,
    peak_hour: int = DEFAULT_SOLAR_PEAK_HOUR,
) -> dict[str, Any]:
    """Build a solar generator entry in the new JSON schema."""
    output = daylight_profile(peak=peak, peak_hour=peak_hour)
    cost = [round(produced * SOLAR_COST_PER_KWH, 2) for produced in output]
    return {
        "name": name,
        "kind": "solar",
        "hourly_output_profile": output,
        "hourly_cost_profile": cost,
    }


def _diesel_generator(name: str, hourly_output: float, hourly_cost: float) -> dict[str, Any]:
    """Build a diesel generator entry in the new JSON schema (scalar params)."""
    return {
        "name": name,
        "kind": "diesel",
        "hourly_output": hourly_output,
        "hourly_cost": hourly_cost,
    }


def _make_consumers(bases: list[float], evening_peak: float = 1.6) -> list[dict[str, Any]]:
    return [
        {
            "name": f"house_{idx + 1}",
            "demand": consumption_profile(base, evening_peak=evening_peak),
        }
        for idx, base in enumerate(bases)
    ]


def make_surplus_case() -> dict[str, Any]:
    """Case 1: 10 consumers, 4 generators; total capacity comfortably exceeds peak demand."""
    consumer_bases = [3.0, 4.5, 5.0, 3.5, 6.0, 4.0, 5.5, 3.0, 4.5, 5.0]
    # Peak demand: ~10 * 5 * 1.6 = 80 kWh in the evening. Generators below exceed that.
    generators = [
        _solar_generator("solar_main", peak=80.0),
        _solar_generator("solar_aux", peak=40.0, peak_hour=12),
        _diesel_generator("diesel_main", hourly_output=60.0, hourly_cost=120.0),
        _diesel_generator("diesel_aux", hourly_output=40.0, hourly_cost=80.0),
    ]
    return {
        "name": "surplus_case",
        "description": (
            "10 consumers, 4 generators; total capacity comfortably exceeds peak demand."
        ),
        "consumers": _make_consumers(consumer_bases),
        "generators": generators,
    }


def make_deficit_case() -> dict[str, Any]:
    """Case 2: 10 consumers, 3 generators; insufficient generation at evening/night peaks."""
    consumer_bases = [6.0, 7.0, 5.5, 8.0, 6.5, 7.5, 6.0, 5.5, 7.0, 8.5]
    # Peak demand: ~10 * 7 * 1.8 = 126 kWh in evening; only one small diesel covers nights.
    generators = [
        _solar_generator("solar_main", peak=90.0),
        _solar_generator("solar_aux", peak=30.0, peak_hour=11),
        _diesel_generator("diesel_small", hourly_output=30.0, hourly_cost=90.0),
    ]
    return {
        "name": "deficit_case",
        "description": (
            "10 consumers, 3 generators; insufficient generation at evening/night peaks."
        ),
        "consumers": _make_consumers(consumer_bases, evening_peak=1.8),
        "generators": generators,
    }


CASES: list[tuple[Any, str]] = [
    (make_surplus_case, "surplus.json"),
    (make_deficit_case, "deficit.json"),
]


def main() -> None:
    output_dir = Path(__file__).parent
    for builder, filename in CASES:
        case = builder()
        path = output_dir / filename
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(case, handle, indent=2, ensure_ascii=False)
        print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
