"""Unit tests for the energy scheduler.

Run with: ``pytest tests/unit/ -v``
"""

import json
from pathlib import Path

import pytest
from src import (
    HOURS_PER_DAY,
    Consumer,
    DieselGenerator,
    EnergyScheduler,
    ProblemInstance,
    Schedule,
    SolarGenerator,
)
from src.io_utils import load_problem, save_schedule, schedule_to_dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
CASES_DIR = PROJECT_ROOT / "tests" / "cases"

ALL_BACKENDS = ["scipy", "pulp"]
CASE_FILES = ["surplus.json", "deficit.json"]

# Tolerances for floating-point comparisons.
FEASIBILITY_TOLERANCE = 1e-6
COST_EQUALITY_TOLERANCE = 1e-4


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _constant_profile(value: float) -> tuple[float, ...]:
    """Build a 24-hour profile with the same value in every hour."""
    return tuple([value] * HOURS_PER_DAY)


@pytest.fixture
def trivial_problem() -> ProblemInstance:
    """One consumer, one diesel generator with more than enough capacity."""
    return ProblemInstance(
        consumers=(Consumer(name="user", demand=_constant_profile(10.0)),),
        generators=(
            DieselGenerator(name="gen", hourly_output=20.0, hourly_cost=5.0),
        ),
    )


@pytest.fixture
def deficit_problem() -> ProblemInstance:
    """Two consumers of 15 each, one 20-unit generator: only one consumer fits per hour."""
    return ProblemInstance(
        consumers=(
            Consumer(name="big_1", demand=_constant_profile(15.0)),
            Consumer(name="big_2", demand=_constant_profile(15.0)),
        ),
        generators=(
            DieselGenerator(name="small_gen", hourly_output=20.0, hourly_cost=10.0),
        ),
    )


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_trivial_full_service(trivial_problem: ProblemInstance, backend: str) -> None:
    """All consumers should be served when generation is sufficient."""
    schedule = EnergyScheduler(backend=backend).solve(trivial_problem)
    assert schedule.served_count == HOURS_PER_DAY
    assert all(schedule.served[hour][0] for hour in range(HOURS_PER_DAY))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_deficit_partial_service(deficit_problem: ProblemInstance, backend: str) -> None:
    """With capacity 20 and two consumers of 15, exactly one is served each hour."""
    schedule = EnergyScheduler(backend=backend).solve(deficit_problem)
    for hour in range(HOURS_PER_DAY):
        served_this_hour = sum(schedule.served[hour])
        assert served_this_hour == 1, (
            f"hour {hour}: served {served_this_hour}, expected 1"
        )


def test_empty_problem() -> None:
    """An empty problem returns a zero-cost, zero-service schedule."""
    problem = ProblemInstance(consumers=(), generators=())
    schedule = EnergyScheduler().solve(problem)
    assert schedule.served_count == 0
    assert schedule.total_cost == 0.0


# ---------------------------------------------------------------------------
# Cross-validation between backends and strategies
# ---------------------------------------------------------------------------


@pytest.mark.demo
@pytest.mark.parametrize("case_file", CASE_FILES)
def test_backends_agree(case_file: str) -> None:
    """scipy and pulp must produce equal served_count and total_cost."""
    problem = load_problem(CASES_DIR / case_file)
    scipy_schedule = EnergyScheduler(backend="scipy").solve(problem)
    pulp_schedule = EnergyScheduler(backend="pulp").solve(problem)

    assert scipy_schedule.served_count == pulp_schedule.served_count, (
        f"{case_file}: served_count differs"
    )
    assert (
        abs(scipy_schedule.total_cost - pulp_schedule.total_cost) < COST_EQUALITY_TOLERANCE
    ), f"{case_file}: total_cost differs"


def test_per_hour_vs_monolithic_pulp() -> None:
    """per_hour and monolithic strategies must agree on served_count and total cost.

    The two may select different (equally-optimal) consumers/generators, but the
    aggregate values must match.
    """
    problem = load_problem(CASES_DIR / "surplus.json")
    per_hour_schedule = EnergyScheduler(backend="pulp", strategy="per_hour").solve(problem)
    monolithic_schedule = EnergyScheduler(backend="pulp", strategy="monolithic").solve(problem)

    assert per_hour_schedule.served_count == monolithic_schedule.served_count
    assert (
        abs(per_hour_schedule.total_cost - monolithic_schedule.total_cost)
        < COST_EQUALITY_TOLERANCE
    )


# ---------------------------------------------------------------------------
# Lexicographic property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_no_trivial_zero_solution(backend: str) -> None:
    """If demand can be served, the scheduler must not return the all-zeros solution.

    Guards against the lexicographic trap: minimizing cost alone would return
    zero served at zero cost. Phase 1 must dominate phase 2.
    """
    problem = ProblemInstance(
        consumers=(Consumer(name="u", demand=_constant_profile(5.0)),),
        generators=(
            DieselGenerator(name="g", hourly_output=10.0, hourly_cost=100.0),
        ),
    )
    schedule = EnergyScheduler(backend=backend).solve(problem)
    # Despite high cost, the consumer must be served (phase 1 priority).
    assert schedule.served_count == HOURS_PER_DAY
    assert schedule.total_cost > 0


# ---------------------------------------------------------------------------
# Feasibility: schedule must respect the energy balance
# ---------------------------------------------------------------------------


@pytest.mark.demo
@pytest.mark.parametrize("case_file", CASE_FILES)
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_schedule_feasibility(case_file: str, backend: str) -> None:
    """For every hour: consumed energy must not exceed produced energy."""
    problem = load_problem(CASES_DIR / case_file)
    schedule = EnergyScheduler(backend=backend).solve(problem)

    for hour in range(HOURS_PER_DAY):
        consumed = sum(
            problem.consumers[consumer_idx].demand[hour]
            for consumer_idx, is_served in enumerate(schedule.served[hour])
            if is_served
        )
        produced = sum(
            problem.generators[generator_idx].output[hour]
            for generator_idx, is_active in enumerate(schedule.active[hour])
            if is_active
        )
        assert consumed <= produced + FEASIBILITY_TOLERANCE, (
            f"hour {hour}: consumed {consumed} > produced {produced}"
        )


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------


def test_consumer_wrong_length() -> None:
    with pytest.raises(ValueError, match="hours"):
        Consumer(name="bad", demand=(1.0, 2.0, 3.0))


def test_consumer_negative_demand() -> None:
    with pytest.raises(ValueError, match="negative"):
        Consumer(name="bad", demand=tuple([-1.0] + [0.0] * 23))


def test_diesel_negative_output_rejected() -> None:
    with pytest.raises(ValueError, match="negative output"):
        DieselGenerator(name="bad", hourly_output=-1.0, hourly_cost=1.0)


def test_diesel_negative_cost_rejected() -> None:
    with pytest.raises(ValueError, match="negative cost"):
        DieselGenerator(name="bad", hourly_output=1.0, hourly_cost=-1.0)


# --- SolarGenerator validation -----------------------------------------------


def _zero_profile() -> tuple[float, ...]:
    return tuple([0.0] * HOURS_PER_DAY)


def test_solar_wrong_output_length() -> None:
    with pytest.raises(ValueError, match="output has"):
        SolarGenerator(
            name="bad",
            hourly_output_profile=(1.0, 2.0, 3.0),
            hourly_cost_profile=_zero_profile(),
        )


def test_solar_wrong_cost_length() -> None:
    with pytest.raises(ValueError, match="cost has"):
        SolarGenerator(
            name="bad",
            hourly_output_profile=_zero_profile(),
            hourly_cost_profile=(1.0, 2.0),
        )


def test_solar_negative_output_rejected() -> None:
    bad_output = tuple([-1.0] + [0.0] * (HOURS_PER_DAY - 1))
    with pytest.raises(ValueError, match="negative output"):
        SolarGenerator(
            name="bad",
            hourly_output_profile=bad_output,
            hourly_cost_profile=_zero_profile(),
        )


def test_solar_negative_cost_rejected() -> None:
    bad_cost = tuple([-1.0] + [0.0] * (HOURS_PER_DAY - 1))
    with pytest.raises(ValueError, match="negative cost"):
        SolarGenerator(
            name="bad",
            hourly_output_profile=_zero_profile(),
            hourly_cost_profile=bad_cost,
        )


# --- io_utils: serialization round-trip --------------------------------------


def _make_trivial_problem_and_schedule() -> tuple[ProblemInstance, Schedule]:
    """Solve a tiny problem so downstream tests have a real schedule to inspect."""
    problem = ProblemInstance(
        consumers=(Consumer(name="user", demand=_constant_profile(10.0)),),
        generators=(DieselGenerator(name="gen", hourly_output=20.0, hourly_cost=5.0),),
    )
    schedule = EnergyScheduler(backend="scipy").solve(problem)
    return problem, schedule


def test_schedule_to_dict_structure() -> None:
    problem, schedule = _make_trivial_problem_and_schedule()
    dumped = schedule_to_dict(problem, schedule)

    assert set(dumped) == {"summary", "hourly_schedule"}
    summary = dumped["summary"]
    assert summary["served_count"] == HOURS_PER_DAY
    assert summary["total_possible"] == HOURS_PER_DAY
    assert summary["total_cost"] == schedule.total_cost
    assert len(dumped["hourly_schedule"]) == HOURS_PER_DAY


def test_schedule_to_dict_hour_fields() -> None:
    problem, schedule = _make_trivial_problem_and_schedule()
    dumped = schedule_to_dict(problem, schedule)

    first_hour = dumped["hourly_schedule"][0]
    assert first_hour["hour"] == 0
    assert first_hour["active_generators"] == ["gen"]
    assert first_hour["served_consumers"] == ["user"]
    assert first_hour["unserved_consumers"] == []
    assert first_hour["cost"] == 5.0


def test_save_schedule_round_trip(tmp_path: Path) -> None:
    problem, schedule = _make_trivial_problem_and_schedule()
    output_path = tmp_path / "schedule.json"

    save_schedule(problem, schedule, output_path)

    assert output_path.exists()
    with open(output_path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded == schedule_to_dict(problem, schedule)


# --- io_utils: error paths ---------------------------------------------------


def test_load_problem_rejects_unknown_generator_kind(tmp_path: Path) -> None:
    bad_input = {
        "consumers": [],
        "generators": [{"name": "x", "kind": "nuclear"}],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad_input), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown generator kind"):
        load_problem(path)


# --- Monolithic strategy is gated to PuLP only -------------------------------


def test_monolithic_strategy_rejects_scipy_backend() -> None:
    """scipy + monolithic is unimplemented; constructor allows it but solve raises."""
    problem = ProblemInstance(
        consumers=(Consumer(name="u", demand=_constant_profile(1.0)),),
        generators=(DieselGenerator(name="g", hourly_output=1.0, hourly_cost=1.0),),
    )
    scheduler = EnergyScheduler(backend="scipy", strategy="monolithic")
    with pytest.raises(NotImplementedError, match="PuLP"):
        scheduler.solve(problem)