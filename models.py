"""Domain models for F1 race strategy optimization."""
from __future__ import annotations
from dataclasses import dataclass


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
    require_two_compounds: bool  # F1 regulation: must use at least 2 different compounds
    # Safety car settings
    sc_pit_loss_seconds: float = 5.0
    sc_conserve_laps: int = 3
    sc_conserve_factor: float = 0.5
    position_loss_value: float = 2.5  # Seconds penalty per position lost
    # Tactical analysis settings
    drs_threshold_seconds: float = 1.0
    dirty_air_loss_per_lap: float = 0.5
    inlap_push_gain: float = 0.3
    outlap_penalty: float = 1.5

    def get_pace_mode(self, mode_name: str) -> PaceMode:
        """Get pace mode by name."""
        return self.pace_modes[mode_name]
    
    def has_attack_mode(self) -> bool:
        """Check if attack mode is configured."""
        return 'attack' in self.pace_modes


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


@dataclass
class SCAnalysis:
    """Result of safety car pit decision analysis."""
    # Current tire state
    current_compound: str
    stint_laps: int  # Laps completed on current tires
    remaining_laps: int  # Laps remaining in race
    current_wear_laps: int  # Effective wear (accounting for SC conservation)
    remaining_competitive_laps: int  # How many more laps current tires can do
    tire_wear_percent: float  # Percentage of tire life used
    can_finish_no_pit: bool  # Can finish race without pitting
    
    # Stay out option
    stay_out_strategy: Strategy | None
    stay_out_ert: float
    
    # Pit now option
    pit_now_strategy: Strategy | None
    pit_now_ert: float
    
    # Recommendation
    recommendation: str  # "PIT" or "STAY_OUT"
    time_delta: float  # ERT difference (positive = pit is faster)
    
    # Value breakdown
    sc_pit_value: float  # Pure timing savings: green flag pit - SC pit (e.g., 16s)
    
    # Position loss when pitting
    positions_lost: int  # Estimated positions lost by pitting under SC
    position_penalty: float  # positions_lost × position_loss_value
    
    # War gaming
    pace_deficit_per_lap: float  # How much slower per lap if staying out vs fresh tires
    total_time_loss: float  # Total time lost over remaining laps
    risk_assessment: str  # "LOW", "MODERATE", "HIGH"


@dataclass
class TireDomain:
    """A range of laps where one compound is fastest."""
    compound: str
    start_lap: int
    end_lap: int
    start_laptime_s: float
    end_laptime_s: float


@dataclass
class CrossoverPoint:
    """Point where one compound becomes faster than another."""
    lap: int
    from_compound: str
    from_laptime_s: float
    to_compound: str
    to_laptime_s: float


@dataclass
class TireDomainAnalysis:
    """Result of tire domain analysis showing which compound is fastest at each lap."""
    domains: list[TireDomain]
    crossover_points: list[CrossoverPoint]
    compound_details: dict[str, dict]  # Compound name -> {base_pace, deg, cliff, etc.}
    max_analysis_lap: int  # How far we analyzed


@dataclass
class UndercutAnalysis:
    """Result of undercut/overcut analysis."""
    # Current state
    gap_to_rival: float  # Positive = ahead, negative = behind
    current_lap: int
    rival_pit_lap: int
    laps_until_rival_pits: int
    
    # Your tire state
    your_compound: str
    your_tire_laps: int
    your_wear_percent: float
    
    # Rival tire state
    rival_compound: str
    rival_tire_laps: int
    rival_wear_percent: float
    
    # Pit-to compound
    pit_to_compound: str
    
    # Undercut analysis
    undercut_viable: bool
    fresh_tire_advantage: float  # Pace advantage per lap on fresh tires
    undercut_window_laps: int  # Laps you'd be on fresh tires before rival pits
    time_gained_undercut: float  # Total time gained during undercut window
    projected_gap_after_undercut: float  # Gap after rival pits (positive = ahead)
    
    # Overcut/stay out analysis
    overcut_viable: bool
    time_lost_staying_out: float  # Time lost while rival is on fresh tires
    projected_gap_after_overcut: float
    
    # Recommendation
    recommendation: str  # "UNDERCUT", "OVERCUT", or "STAY_OUT"
    recommendation_reason: str


@dataclass
class ModeScenario:
    """A scenario for DRS defense or attack analysis."""
    name: str  # e.g., "PUSH", "CONSERVE", "BURST_PUSH"
    mode_sequence: list[tuple[str, int]]  # [(mode, laps), ...]
    final_gap: float
    tire_wear_at_end: float  # Effective laps of wear
    tire_percent_at_end: float
    exceeds_tire_life: bool
    cliff_lap: int | None  # Lap when tires cliff (if applicable)
    sustainable: bool
    description: str


@dataclass
class DRSAnalysis:
    """Result of DRS defense analysis."""
    # Current state
    gap_to_attacker: float
    stint_laps_remaining: int
    in_drs_range: bool
    
    # Your tire state
    your_compound: str
    your_tire_laps: int
    your_wear_percent: float
    your_max_competitive_laps: int
    
    # Attacker tire state
    attacker_compound: str
    attacker_tire_laps: int
    attacker_wear_percent: float
    
    # Pace comparison
    base_pace_delta: float  # Your pace vs attacker (positive = you're slower)
    
    # Scenarios analyzed
    scenarios: list[ModeScenario]
    
    # Optimal burst push
    optimal_push_laps: int  # Optimal laps to push before conserving
    
    # Recommendation
    recommendation: str  # "PUSH", "CONSERVE", "BURST_PUSH"
    recommended_scenario: ModeScenario


@dataclass
class LiveStrategy:
    """A strategy option for mid-race recalculation."""
    strategy: Strategy
    next_pit_lap: int | None  # Lap number of next pit (None = no more pits)
    next_compound: str | None  # Compound after next pit
    remaining_on_current: int  # Laps before current tire hits cliff
    ert_to_finish: float


@dataclass
class LiveAnalysis:
    """Result of mid-race strategy recalculation."""
    # Current state
    current_lap: int  # Current lap number
    current_compound: str
    tire_laps: int  # Laps on current tires
    remaining_laps: int  # Laps remaining in race
    tire_wear_percent: float
    remaining_competitive_laps: int  # Laps before cliff
    
    # Can finish without pitting?
    can_finish_no_pit: bool
    no_pit_ert: float | None  # ERT if staying out to end
    
    # Best strategies from here
    strategies: list[LiveStrategy]
    
    # Recommendation
    recommended: LiveStrategy


@dataclass
class AttackAnalysis:
    """Result of attack/catch analysis."""
    # Current state
    gap_to_target: float
    stint_laps_remaining: int
    drs_threshold: float
    
    # Your tire state
    your_compound: str
    your_tire_laps: int
    your_wear_percent: float
    your_max_competitive_laps: int
    
    # Target tire state
    target_compound: str
    target_tire_laps: int
    target_wear_percent: float
    
    # Natural convergence
    natural_closing_rate: float  # Positive = closing
    laps_to_drs_natural: int | None  # None if not reachable
    can_reach_drs_naturally: bool
    
    # Scenarios analyzed
    scenarios: list[ModeScenario]
    
    # Recommendation
    recommendation: str  # "STAY_ON_PLAN", "PUSH", "ATTACK"
    recommended_scenario: ModeScenario
    tire_warning: str | None  # Warning if tires would be burned by stint end

