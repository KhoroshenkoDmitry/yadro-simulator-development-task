"""Data models for the energy scheduling problem."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

HOURS_PER_DAY = 24


@dataclass(frozen=True)
class Consumer:
    """A power consumer with an hourly demand profile.

    Attributes:
        name: Human-readable identifier.
        demand: Hourly demand profile in kWh. Length must equal ``HOURS_PER_DAY``.
    """

    name: str
    demand: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.demand) != HOURS_PER_DAY:
            raise ValueError(
                f"Consumer '{self.name}' demand has {len(self.demand)} hours, "
                f"expected {HOURS_PER_DAY}"
            )
        if any(value < 0 for value in self.demand):
            raise ValueError(f"Consumer '{self.name}' has negative demand")


@dataclass(frozen=True)
class Generator(ABC):
    """Abstract power generator.

    Subclasses must expose two indexable sequences of length ``HOURS_PER_DAY``:

    * ``output`` — energy produced in each hour while the generator is on (kWh).
    * ``cost``   — cost incurred in each hour while the generator is on.

    These are exposed as properties so that the storage representation
    (e.g. a single scalar for diesel vs. a full hourly profile for solar)
    is decoupled from the consumer-facing interface used by the scheduler.
    """

    name: str

    @property
    @abstractmethod
    def output(self) -> tuple[float, ...]:
        """Hourly output profile of length ``HOURS_PER_DAY``."""

    @property
    @abstractmethod
    def cost(self) -> tuple[float, ...]:
        """Hourly cost profile of length ``HOURS_PER_DAY``."""


@dataclass(frozen=True)
class DieselGenerator(Generator):
    """Diesel generator: constant output and constant cost across all hours.

    Attributes:
        name: Human-readable identifier.
        hourly_output: Energy produced per hour while running (kWh).
        hourly_cost: Cost incurred per hour while running.
    """

    hourly_output: float
    hourly_cost: float

    def __post_init__(self) -> None:
        if self.hourly_output < 0:
            raise ValueError(
                f"Diesel generator '{self.name}' has negative output {self.hourly_output}"
            )
        if self.hourly_cost < 0:
            raise ValueError(f"Diesel generator '{self.name}' has negative cost {self.hourly_cost}")

    @property
    def output(self) -> tuple[float, ...]:
        return (self.hourly_output,) * HOURS_PER_DAY

    @property
    def cost(self) -> tuple[float, ...]:
        return (self.hourly_cost,) * HOURS_PER_DAY


@dataclass(frozen=True)
class SolarGenerator(Generator):
    """Solar generator: hourly-varying output (zero at night) and hourly cost.

    Attributes:
        name: Human-readable identifier.
        hourly_output_profile: Energy produced in each hour, length ``HOURS_PER_DAY``.
        hourly_cost_profile: Cost incurred in each hour, length ``HOURS_PER_DAY``.
    """

    hourly_output_profile: tuple[float, ...]
    hourly_cost_profile: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.hourly_output_profile) != HOURS_PER_DAY:
            raise ValueError(
                f"Solar generator '{self.name}' output has "
                f"{len(self.hourly_output_profile)} hours, expected {HOURS_PER_DAY}"
            )
        if len(self.hourly_cost_profile) != HOURS_PER_DAY:
            raise ValueError(
                f"Solar generator '{self.name}' cost has "
                f"{len(self.hourly_cost_profile)} hours, expected {HOURS_PER_DAY}"
            )
        if any(value < 0 for value in self.hourly_output_profile):
            raise ValueError(f"Solar generator '{self.name}' has negative output")
        if any(value < 0 for value in self.hourly_cost_profile):
            raise ValueError(f"Solar generator '{self.name}' has negative cost")

    @property
    def output(self) -> tuple[float, ...]:
        return self.hourly_output_profile

    @property
    def cost(self) -> tuple[float, ...]:
        return self.hourly_cost_profile


@dataclass(frozen=True)
class ProblemInstance:
    """Full input data for the scheduling problem.

    Attributes:
        consumers: Ordered tuple of consumers.
        generators: Ordered tuple of generators (any mix of subclasses).
    """

    consumers: tuple[Consumer, ...]
    generators: tuple[Generator, ...]

    @property
    def num_consumers(self) -> int:
        return len(self.consumers)

    @property
    def num_generators(self) -> int:
        return len(self.generators)


@dataclass(frozen=True)
class Schedule:
    """Solution to the scheduling problem.

    Attributes:
        served: ``served[hour][consumer_idx]`` is True iff that consumer
                is served in that hour. Shape: ``HOURS_PER_DAY x num_consumers``.
        active: ``active[hour][generator_idx]`` is True iff that generator
                is on in that hour. Shape: ``HOURS_PER_DAY x num_generators``.
        hourly_cost: Total cost for each hour. Length ``HOURS_PER_DAY``.
        hourly_unserved: Indices of unserved consumers per hour.
    """

    served: tuple[tuple[bool, ...], ...]
    active: tuple[tuple[bool, ...], ...]
    hourly_cost: tuple[float, ...]
    hourly_unserved: tuple[tuple[int, ...], ...]

    @property
    def served_count(self) -> int:
        """Total number of (consumer, hour) pairs that were served."""
        return sum(sum(row) for row in self.served)

    @property
    def total_cost(self) -> float:
        """Sum of ``hourly_cost`` across all hours."""
        return sum(self.hourly_cost)
