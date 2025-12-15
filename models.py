"""Domain models for F1 race strategy optimization."""
from dataclasses import dataclass
from typing import Optional


# Mode abbreviation constant - used for output formatting
MODE_ABBREV = {'normal': 'N', 'push': 'P', 'conserve': 'C'}

# Available pace modes
PACE_MODES = ['normal', 'push', 'conserve']


@dataclass
class PaceMode:
    """Pace mode configuration affecting lap time and tire degradation."""
    name: str
    delta_per_lap: float  # Seconds gained/lost per lap
    degradation_factor: float  # Multiplier for tire wear rate

    @property
    def is_normal(self) -> bool:
        return self.name == 'normal'


@dataclass
class Compound:
    """Tire compound performance characteristics."""
    name: str
    avg_lap_time_s: float
    degradation_s_per_lap: float
    max_competitive_laps: int


@dataclass
class Inventory:
    """Available tire inventory for the race."""
    soft_new: int
    soft_scrubbed: int
    medium_new: int
    medium_scrubbed: int
    hard_new: int
    hard_scrubbed: int

    def get_total(self, compound: str) -> int:
        """Get total tires available for a compound (new + scrubbed)."""
        compound_lower = compound.lower()
        new_key = f"{compound_lower}_new"
        scrubbed_key = f"{compound_lower}_scrubbed"
        return getattr(self, new_key, 0) + getattr(self, scrubbed_key, 0)

    def get_new_count(self, compound: str) -> int:
        """Get count of new tires for a compound."""
        return getattr(self, f"{compound.lower()}_new", 0)

    def get_scrubbed_count(self, compound: str) -> int:
        """Get count of scrubbed tires for a compound."""
        return getattr(self, f"{compound.lower()}_scrubbed", 0)


@dataclass
class RaceConfig:
    """Complete race configuration."""
    race_laps: int
    pit_loss_seconds: float
    compounds: dict[str, Compound]
    inventory: Inventory
    pace_modes: dict[str, PaceMode]
    top_strategies: int
    max_pit_stops: int
    min_stint_laps: int | str  # Can be 'auto' or int
    stint_lap_step: int
    scrubbed_life_penalty: int
    require_medium_or_hard: bool

    def get_pace_mode(self, mode_name: str) -> PaceMode:
        """Get pace mode by name, returns normal mode for 'normal'."""
        if mode_name == 'normal':
            return PaceMode(name='normal', delta_per_lap=0.0, degradation_factor=1.0)
        return self.pace_modes[mode_name]


@dataclass
class Strategy:
    """A race strategy with stint breakdown and optimal pace modes."""
    stints: list[tuple[str, int]]  # List of (compound_name, laps)
    optimal_modes: tuple[str, ...]
    ert: float

    @property
    def pit_stops(self) -> int:
        """Number of pit stops in this strategy."""
        return len(self.stints) - 1

    @property
    def compound_sequence(self) -> list[str]:
        """List of compound names used in order."""
        return [stint[0] for stint in self.stints]

    @property
    def split(self) -> tuple[int, ...]:
        """Tuple of stint lengths."""
        return tuple(stint[1] for stint in self.stints)

    def get_pit_laps(self) -> list[int]:
        """Get list of lap numbers where pit stops occur."""
        pit_laps = []
        cumulative = 0
        for i in range(len(self.stints) - 1):
            cumulative += self.stints[i][1]
            pit_laps.append(cumulative)
        return pit_laps

    def format_pit_laps(self) -> str:
        """Format pit laps as comma-separated string."""
        return ", ".join(f"Lap {lap}" for lap in self.get_pit_laps())

    def format_strategy_string(self) -> str:
        """Format strategy as 'Compound(Mode) -> Compound(Mode) -> ...'"""
        stint_strs = [
            f"{compound}({MODE_ABBREV[mode]})"
            for compound, mode in zip(self.compound_sequence, self.optimal_modes)
        ]
        return " -> ".join(stint_strs)

    def format_split_string(self) -> str:
        """Format stint split as 'N - N - N'"""
        return " - ".join(map(str, self.split))

