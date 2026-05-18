"""Energy scheduler: two-phase lexicographic optimization.

Phase 1: maximize number of (consumer, hour) servings.
Phase 2: minimize total generation cost while preserving the phase-1 maximum.

Supports two solver backends (``scipy.optimize.milp`` / PuLP) and two
decomposition strategies (per-hour / monolithic).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np

from .models import HOURS_PER_DAY, ProblemInstance, Schedule

# 1-D float64 numpy array — used throughout for demands / outputs / costs.
FloatArray = np.ndarray[Any, np.dtype[np.float64]]

# Numerical tolerance for feasibility / integrality checks.
NUMERICAL_TOLERANCE = 1e-6


class Backend(str, Enum):
    SCIPY = "scipy"
    PULP = "pulp"


class Strategy(str, Enum):
    PER_HOUR = "per_hour"  # 24 independent subproblems
    MONOLITHIC = "monolithic"  # single problem indexed by hour


@dataclass
class _HourSolution:
    """Internal: solution for a single hour."""

    served_mask: list[bool]  # length == num_consumers
    active_mask: list[bool]  # length == num_generators
    cost: float
    served_target: int  # the S*_t value from phase 1


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _verify_hour_feasibility(
    served_mask: list[bool],
    active_mask: list[bool],
    consumer_demands: FloatArray,
    generator_outputs: FloatArray,
    served_target: int,
) -> None:
    """Validate solver output. Guards against rare solver bugs (e.g. HiGHS #24141)."""
    produced = sum(
        output
        for output, is_active in zip(generator_outputs, active_mask, strict=True)
        if is_active
    )
    consumed = sum(
        demand
        for demand, is_served in zip(consumer_demands, served_mask, strict=True)
        if is_served
    )
    if consumed > produced + NUMERICAL_TOLERANCE:
        raise RuntimeError(
            f"Infeasible solution returned: consumed {consumed} > produced {produced}"
        )
    if sum(served_mask) + NUMERICAL_TOLERANCE < served_target:
        raise RuntimeError(
            f"Infeasible solution returned: served {sum(served_mask)} < target {served_target}"
        )


def _hour_solution_from_masks(
    served_mask: list[bool],
    active_mask: list[bool],
    generator_costs: FloatArray,
    consumer_demands: FloatArray,
    generator_outputs: FloatArray,
    served_target: int,
) -> _HourSolution:
    """Verify feasibility and assemble the per-hour solution record."""
    _verify_hour_feasibility(
        served_mask, active_mask, consumer_demands, generator_outputs, served_target
    )
    actual_cost = sum(
        cost for cost, is_active in zip(generator_costs, active_mask, strict=True) if is_active
    )
    return _HourSolution(
        served_mask=served_mask,
        active_mask=active_mask,
        cost=actual_cost,
        served_target=served_target,
    )


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class _SolverBackend(ABC):
    """Abstract MILP solver for a single hour."""

    @abstractmethod
    def solve_phase1(
        self,
        consumer_demands: FloatArray,
        generator_outputs: FloatArray,
    ) -> int:
        """Return ``S*_t`` — maximum number of consumers that can be served."""

    @abstractmethod
    def solve_phase2(
        self,
        consumer_demands: FloatArray,
        generator_outputs: FloatArray,
        generator_costs: FloatArray,
        served_target: int,
    ) -> _HourSolution:
        """Return optimal schedule minimizing cost s.t. ``served >= target``."""


# ---------------------------------------------------------------------------
# scipy.optimize.milp backend
# ---------------------------------------------------------------------------


class _ScipyBackend(_SolverBackend):
    """MILP via ``scipy.optimize.milp`` (HiGHS).

    Variable ordering for a single hour:
    ``[S_0, ..., S_{num_consumers-1}, Q_0, ..., Q_{num_generators-1}]``.
    All variables are binary.
    """

    def __init__(self) -> None:
        from scipy.optimize import (  # type: ignore[import-untyped]
            Bounds,
            LinearConstraint,
            milp,
        )

        self._milp = milp
        self._LinearConstraint = LinearConstraint
        self._Bounds = Bounds

    def _build_balance_constraint(
        self,
        consumer_demands: FloatArray,
        generator_outputs: FloatArray,
    ) -> Any:
        """Build the constraint ``sum_j A_j * S_j - sum_k E_k * Q_k <= 0``."""
        row = np.concatenate([consumer_demands, -generator_outputs])
        return self._LinearConstraint(row.reshape(1, -1), -np.inf, 0.0)

    def _binary_bounds(self, num_vars: int) -> Any:
        return self._Bounds(lb=np.zeros(num_vars), ub=np.ones(num_vars))

    def solve_phase1(
        self,
        consumer_demands: FloatArray,
        generator_outputs: FloatArray,
    ) -> int:
        num_consumers = len(consumer_demands)
        num_generators = len(generator_outputs)
        num_vars = num_consumers + num_generators

        # Objective: max sum(S) == min sum(-S)
        objective = np.concatenate([-np.ones(num_consumers), np.zeros(num_generators)])
        constraints = self._build_balance_constraint(consumer_demands, generator_outputs)
        integrality = np.ones(num_vars)

        result = self._milp(
            c=objective,
            constraints=constraints,
            integrality=integrality,
            bounds=self._binary_bounds(num_vars),
        )
        if not result.success:
            raise RuntimeError(f"Phase 1 failed: {result.message}")

        max_served = int(round(-result.fun))

        # Sanity check: each S_j is 0/1 and the sum matches the objective.
        served_values = result.x[:num_consumers]
        non_integer = [
            value for value in served_values if abs(value - round(value)) >= NUMERICAL_TOLERANCE
        ]
        if non_integer:
            raise RuntimeError(f"Phase 1 returned non-integer S values: {non_integer}")
        rounded_sum = sum(round(value) for value in served_values)
        if rounded_sum != max_served:
            raise RuntimeError(
                f"Phase 1 objective mismatch: sum(S)={rounded_sum}, S*={max_served}"
            )
        return max_served

    def solve_phase2(
        self,
        consumer_demands: FloatArray,
        generator_outputs: FloatArray,
        generator_costs: FloatArray,
        served_target: int,
    ) -> _HourSolution:
        num_consumers = len(consumer_demands)
        num_generators = len(generator_outputs)
        num_vars = num_consumers + num_generators

        # Objective: min sum_k cost_k * Q_k
        objective = np.concatenate([np.zeros(num_consumers), generator_costs])

        # Row 1: demand @ S - output @ Q <= 0           (energy balance)
        # Row 2: -sum(S)                <= -served_target  (i.e. sum(S) >= target)
        balance_row = np.concatenate([consumer_demands, -generator_outputs])
        target_row = np.concatenate([-np.ones(num_consumers), np.zeros(num_generators)])
        constraint_matrix = np.vstack([balance_row, target_row])
        upper_bounds = np.array([0.0, -float(served_target)])
        lower_bounds = np.full(2, -np.inf)
        constraints = self._LinearConstraint(constraint_matrix, lower_bounds, upper_bounds)

        result = self._milp(
            c=objective,
            constraints=constraints,
            integrality=np.ones(num_vars),
            bounds=self._binary_bounds(num_vars),
        )
        if not result.success:
            raise RuntimeError(f"Phase 2 failed: {result.message}")

        served_mask = [round(v) == 1 for v in result.x[:num_consumers]]
        active_mask = [round(v) == 1 for v in result.x[num_consumers:]]
        return _hour_solution_from_masks(
            served_mask,
            active_mask,
            generator_costs,
            consumer_demands,
            generator_outputs,
            served_target,
        )


# ---------------------------------------------------------------------------
# PuLP backend
# ---------------------------------------------------------------------------


class _PulpBackend(_SolverBackend):
    """MILP via PuLP (default CBC solver)."""

    def __init__(self) -> None:
        import pulp  # type: ignore[import-untyped]

        self._pulp = pulp

    def _build_binary_vars(self, name: str, count: int) -> list[Any]:
        return [self._pulp.LpVariable(f"{name}_{idx}", cat="Binary") for idx in range(count)]

    def solve_phase1(
        self,
        consumer_demands: FloatArray,
        generator_outputs: FloatArray,
    ) -> int:
        pulp = self._pulp
        num_consumers = len(consumer_demands)
        num_generators = len(generator_outputs)

        problem = pulp.LpProblem("phase1", pulp.LpMaximize)
        served_vars = self._build_binary_vars("S", num_consumers)
        active_vars = self._build_binary_vars("Q", num_generators)

        problem += pulp.lpSum(served_vars)
        problem += pulp.lpSum(
            consumer_demands[j] * served_vars[j] for j in range(num_consumers)
        ) <= pulp.lpSum(generator_outputs[k] * active_vars[k] for k in range(num_generators))

        status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"Phase 1 failed: {pulp.LpStatus[status]}")
        return int(round(pulp.value(problem.objective)))

    def solve_phase2(
        self,
        consumer_demands: FloatArray,
        generator_outputs: FloatArray,
        generator_costs: FloatArray,
        served_target: int,
    ) -> _HourSolution:
        pulp = self._pulp
        num_consumers = len(consumer_demands)
        num_generators = len(generator_outputs)

        problem = pulp.LpProblem("phase2", pulp.LpMinimize)
        served_vars = self._build_binary_vars("S", num_consumers)
        active_vars = self._build_binary_vars("Q", num_generators)

        problem += pulp.lpSum(
            generator_costs[k] * active_vars[k] for k in range(num_generators)
        )
        problem += pulp.lpSum(
            consumer_demands[j] * served_vars[j] for j in range(num_consumers)
        ) <= pulp.lpSum(generator_outputs[k] * active_vars[k] for k in range(num_generators))
        problem += pulp.lpSum(served_vars) >= served_target

        status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"Phase 2 failed: {pulp.LpStatus[status]}")

        served_mask = [(pulp.value(var) or 0) > 0.5 for var in served_vars]
        active_mask = [(pulp.value(var) or 0) > 0.5 for var in active_vars]
        return _hour_solution_from_masks(
            served_mask,
            active_mask,
            generator_costs,
            consumer_demands,
            generator_outputs,
            served_target,
        )


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------


def _make_backend(backend: Backend) -> _SolverBackend:
    if backend == Backend.SCIPY:
        return _ScipyBackend()
    if backend == Backend.PULP:
        return _PulpBackend()
    raise ValueError(f"Unknown backend: {backend}")


class EnergyScheduler:
    """Two-phase lexicographic scheduler for the energy unit-commitment problem.

    Usage::

        scheduler = EnergyScheduler(backend="scipy", strategy="per_hour")
        schedule = scheduler.solve(problem)

    The per-hour strategy decomposes the problem into 24 independent
    subproblems (matches the problem statement: maximize served *per hour*).
    The monolithic strategy solves a single MILP with the hour index — kept
    for cross-validation and as a base for future extensions that couple
    hours together (batteries, ramp-up).
    """

    def __init__(
        self,
        backend: Backend | Literal["scipy", "pulp"] = Backend.SCIPY,
        strategy: Strategy | Literal["per_hour", "monolithic"] = Strategy.PER_HOUR,
    ) -> None:
        self.backend = Backend(backend)
        self.strategy = Strategy(strategy)

    def solve(self, problem: ProblemInstance) -> Schedule:
        if self.strategy == Strategy.PER_HOUR:
            return self._solve_per_hour(problem)
        return self._solve_monolithic(problem)

    # ------------------------------------------------------------------ #
    # Per-hour strategy
    # ------------------------------------------------------------------ #

    def _solve_per_hour(self, problem: ProblemInstance) -> Schedule:
        if problem.num_consumers == 0 and problem.num_generators == 0:
            return self._empty_schedule()

        backend = _make_backend(self.backend)

        hour_solutions: list[_HourSolution] = []
        for hour in range(HOURS_PER_DAY):
            consumer_demands = np.array(
                [consumer.demand[hour] for consumer in problem.consumers], dtype=float
            )
            generator_outputs = np.array(
                [generator.output[hour] for generator in problem.generators], dtype=float
            )
            generator_costs = np.array(
                [generator.cost[hour] for generator in problem.generators], dtype=float
            )
            max_served = backend.solve_phase1(consumer_demands, generator_outputs)
            hour_solutions.append(
                backend.solve_phase2(
                    consumer_demands, generator_outputs, generator_costs, max_served
                )
            )

        return self._assemble_schedule(hour_solutions)

    @staticmethod
    def _empty_schedule() -> Schedule:
        return Schedule(
            served=tuple(tuple() for _ in range(HOURS_PER_DAY)),
            active=tuple(tuple() for _ in range(HOURS_PER_DAY)),
            hourly_cost=tuple(0.0 for _ in range(HOURS_PER_DAY)),
            hourly_unserved=tuple(tuple() for _ in range(HOURS_PER_DAY)),
        )

    @staticmethod
    def _assemble_schedule(hour_solutions: list[_HourSolution]) -> Schedule:
        served = tuple(tuple(hour.served_mask) for hour in hour_solutions)
        active = tuple(tuple(hour.active_mask) for hour in hour_solutions)
        hourly_cost = tuple(hour.cost for hour in hour_solutions)
        hourly_unserved = tuple(
            tuple(idx for idx, is_served in enumerate(hour.served_mask) if not is_served)
            for hour in hour_solutions
        )
        return Schedule(
            served=served,
            active=active,
            hourly_cost=hourly_cost,
            hourly_unserved=hourly_unserved,
        )

    # ------------------------------------------------------------------ #
    # Monolithic strategy
    # ------------------------------------------------------------------ #

    def _solve_monolithic(self, problem: ProblemInstance) -> Schedule:
        """Solve all 24 hours as a single MILP.

        Provided for cross-validation and as a foundation for extensions
        that introduce inter-hour coupling (batteries, ramp constraints).

        Currently implemented only for the PuLP backend due to scipy's
        somewhat awkward sparse-matrix interface for problems of this shape.
        """
        if self.backend != Backend.PULP:
            raise NotImplementedError(
                "Monolithic strategy currently requires the PuLP backend"
            )

        import pulp

        num_consumers = problem.num_consumers
        num_generators = problem.num_generators

        def make_variable_grids() -> tuple[list[list[Any]], list[list[Any]]]:
            served_grid = [
                [
                    pulp.LpVariable(f"S_{hour}_{consumer_idx}", cat="Binary")
                    for consumer_idx in range(num_consumers)
                ]
                for hour in range(HOURS_PER_DAY)
            ]
            active_grid = [
                [
                    pulp.LpVariable(f"Q_{hour}_{generator_idx}", cat="Binary")
                    for generator_idx in range(num_generators)
                ]
                for hour in range(HOURS_PER_DAY)
            ]
            return served_grid, active_grid

        def add_hourly_balance(
            prob: Any,
            served_grid: list[list[Any]],
            active_grid: list[list[Any]],
        ) -> None:
            for hour in range(HOURS_PER_DAY):
                demands = [consumer.demand[hour] for consumer in problem.consumers]
                outputs = [generator.output[hour] for generator in problem.generators]
                prob += pulp.lpSum(
                    demands[consumer_idx] * served_grid[hour][consumer_idx]
                    for consumer_idx in range(num_consumers)
                ) <= pulp.lpSum(
                    outputs[generator_idx] * active_grid[hour][generator_idx]
                    for generator_idx in range(num_generators)
                )

        # ----- Phase 1: maximize total served -----
        phase1 = pulp.LpProblem("phase1_monolithic", pulp.LpMaximize)
        served_grid_p1, active_grid_p1 = make_variable_grids()
        phase1 += pulp.lpSum(
            served_grid_p1[hour][consumer_idx]
            for hour in range(HOURS_PER_DAY)
            for consumer_idx in range(num_consumers)
        )
        add_hourly_balance(phase1, served_grid_p1, active_grid_p1)

        status = phase1.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"Phase 1 (monolithic) failed: {pulp.LpStatus[status]}")
        total_served_target = int(round(pulp.value(phase1.objective)))

        # ----- Phase 2: minimize cost subject to total-served floor -----
        phase2 = pulp.LpProblem("phase2_monolithic", pulp.LpMinimize)
        served_grid_p2, active_grid_p2 = make_variable_grids()
        phase2 += pulp.lpSum(
            problem.generators[generator_idx].cost[hour]
            * active_grid_p2[hour][generator_idx]
            for hour in range(HOURS_PER_DAY)
            for generator_idx in range(num_generators)
        )
        add_hourly_balance(phase2, served_grid_p2, active_grid_p2)
        phase2 += pulp.lpSum(
            served_grid_p2[hour][consumer_idx]
            for hour in range(HOURS_PER_DAY)
            for consumer_idx in range(num_consumers)
        ) >= total_served_target

        status = phase2.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"Phase 2 (monolithic) failed: {pulp.LpStatus[status]}")

        # ----- Assemble per-hour solutions -----
        hour_solutions: list[_HourSolution] = []
        for hour in range(HOURS_PER_DAY):
            served_mask = [
                (pulp.value(served_grid_p2[hour][consumer_idx]) or 0) > 0.5
                for consumer_idx in range(num_consumers)
            ]
            active_mask = [
                (pulp.value(active_grid_p2[hour][generator_idx]) or 0) > 0.5
                for generator_idx in range(num_generators)
            ]
            hour_cost = sum(
                problem.generators[generator_idx].cost[hour]
                for generator_idx in range(num_generators)
                if active_mask[generator_idx]
            )
            hour_solutions.append(
                _HourSolution(
                    served_mask=served_mask,
                    active_mask=active_mask,
                    cost=hour_cost,
                    served_target=sum(served_mask),
                )
            )

        return self._assemble_schedule(hour_solutions)
