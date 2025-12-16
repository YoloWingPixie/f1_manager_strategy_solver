"""Core calculations: ERT, degradation model, and tire utilities."""
from __future__ import annotations
from math import floor
from itertools import product
from typing import TYPE_CHECKING

from models import RaceConfig, Compound, Inventory, PACE_MODES

if TYPE_CHECKING:
    from models import Strategy


# --- Shared Helpers ---

def calculate_wear_percent(tire_laps: int, max_competitive_laps: int) -> float:
    """Calculate tire wear as a percentage of max competitive life."""
    return (tire_laps / max_competitive_laps) * 100


def best_by_ert(strategies: list[Strategy]) -> Strategy | None:
    """Return the strategy with lowest ERT, or None if list is empty."""
    return min(strategies, key=lambda s: s.ert) if strategies else None


def count_compound_usage(compounds: list[str]) -> dict[str, int]:
    """Count how many times each compound appears in a list."""
    usage: dict[str, int] = {}
    for compound in compounds:
        usage[compound] = usage.get(compound, 0) + 1
    return usage


def find_best_two_stint_modes(
    comp1: Compound,
    laps1: int,
    wear1: float,
    comp2: Compound,
    laps2: int,
    wear2: float,
    pit_loss: float,
    config: RaceConfig
) -> tuple[tuple[str, str], float]:
    """
    Find the optimal pace mode combination for two stints.
    
    Returns (best_modes, best_ert) where best_modes is (mode1, mode2).
    """
    best_ert = float('inf')
    best_modes = ('normal', 'normal')
    
    for mode1 in PACE_MODES:
        for mode2 in PACE_MODES:
            ert1 = calculate_stint_ert_with_wear(comp1, laps1, wear1, mode1, config)
            ert2 = calculate_stint_ert_with_wear(comp2, laps2, wear2, mode2, config)
            total = ert1 + ert2 + pit_loss
            
            if total < best_ert:
                best_ert = total
                best_modes = (mode1, mode2)
    
    return best_modes, best_ert


# --- Degradation Model ---

def get_degradation_factor(lap: float, max_competitive_laps: int) -> float:
    """
    Get progressive degradation multiplier for a given lap of tire wear.
    
    Uses a quadratic curve that accelerates toward the "cliff":
    - Early laps: ~10% of average degradation (tires in sweet spot)
    - Approaching max life: curves up rapidly to 150% (the cliff)
    - Beyond max life: 2x the cliff rate (300%) - tires are done
    
    This models real tire behavior where deg is low early, then falls off a cliff.
    """
    if lap <= max_competitive_laps:
        # Quadratic curve: starts slow, accelerates toward cliff
        ratio = lap / max_competitive_laps
        # 0.1 at lap 0, curves up to 1.5 at max_competitive_laps
        return 0.1 + 1.4 * (ratio ** 2)
    else:
        # Beyond the cliff - 2x the cliff rate
        # At max: factor was 1.5, now it's 3.0 and continues climbing
        over_ratio = (lap - max_competitive_laps) / max_competitive_laps
        return 3.0 + 2.8 * over_ratio  # 2x the curve rate beyond cliff


def calculate_progressive_degradation(
    base_deg: float,
    start_lap: float,
    num_laps: int,
    max_competitive_laps: int
) -> float:
    """
    Calculate total degradation time over a stint using progressive model.
    
    Each lap's degradation = base_deg × factor(current_lap)
    where factor grows from 0.1 (fresh) to 1.5 (at max life).
    """
    total_deg = 0.0
    for i in range(num_laps):
        current_lap = start_lap + i
        factor = get_degradation_factor(current_lap, max_competitive_laps)
        total_deg += base_deg * factor
    return total_deg


# --- Tire Utilities ---

def get_tire_max_laps(
    stints: list[tuple[str, int]],
    config: RaceConfig
) -> list[int]:
    """
    Determine max competitive laps for each stint based on tire type (new vs scrubbed).
    New tires are used first, then scrubbed. Scrubbed have reduced life.
    """
    running_usage: dict[str, int] = {}
    max_laps_list = []

    for compound_name, _ in stints:
        running_usage[compound_name] = running_usage.get(compound_name, 0) + 1
        times_used = running_usage[compound_name]

        base_max = config.compounds[compound_name].max_competitive_laps
        new_count = config.inventory.get_new_count(compound_name)

        if times_used <= new_count:
            max_laps_list.append(base_max)
        else:
            max_laps_list.append(base_max - config.scrubbed_life_penalty)

    return max_laps_list


def calculate_remaining_competitive_laps(compound: Compound, effective_wear: float) -> int:
    """Calculate how many more competitive laps the tire can do."""
    return max(0, floor(compound.max_competitive_laps - effective_wear))


def calculate_effective_wear(
    stint_laps: int,
    sc_laps: int,
    config: RaceConfig
) -> float:
    """
    Calculate effective tire wear accounting for SC conservation.
    SC laps have reduced wear (sc_conserve_factor).
    Returns effective wear in "normal lap equivalents".
    """
    # SC laps count as reduced wear
    sc_wear = sc_laps * config.sc_conserve_factor
    # Normal laps before SC
    normal_laps = max(0, stint_laps - sc_laps)
    return normal_laps + sc_wear


def calculate_lap_time_at_wear(
    compound: Compound,
    lap_number: int,
    mode: str,
    config: RaceConfig
) -> float:
    """Calculate lap time for a compound at a specific lap of wear."""
    pace_mode = config.get_pace_mode(mode)
    base_deg = compound.degradation_s_per_lap * pace_mode.degradation_factor
    
    # Get degradation factor for this lap
    deg_factor = get_degradation_factor(lap_number, compound.max_competitive_laps)
    lap_degradation = base_deg * deg_factor
    
    return compound.avg_lap_time_s + lap_degradation + pace_mode.delta_per_lap


def estimate_tire_delta_per_lap(
    worn_compound: Compound,
    worn_laps: float,
    fresh_compound: Compound,
    config: RaceConfig
) -> float:
    """
    Estimate pace difference per lap between worn and fresh tires.
    Uses progressive degradation model.
    Returns positive value if worn tires are slower.
    """
    # Base pace difference between compounds
    compound_delta = worn_compound.avg_lap_time_s - fresh_compound.avg_lap_time_s
    
    # Current degradation level on worn tires (using progressive model)
    deg_factor = get_degradation_factor(worn_laps, worn_compound.max_competitive_laps)
    current_deg = worn_compound.degradation_s_per_lap * deg_factor
    
    # Fresh tires start at 10% of average degradation
    fresh_deg = fresh_compound.degradation_s_per_lap * 0.1
    
    return compound_delta + (current_deg - fresh_deg)


# --- ERT Calculation ---

def calculate_ert(
    stints: list[tuple[str, int]],
    stint_modes: tuple[str, ...],
    config: RaceConfig
) -> float:
    """
    Calculate Estimated Race Time (ERT) for a strategy with per-stint pace modes.
    
    Uses progressive degradation model where tire deg starts at 10% of average
    and increases to 150% at max competitive tire life.
    
    Returns float('inf') if strategy is invalid (stint exceeds tire life).
    """
    total_time = 0.0
    num_stops = len(stints) - 1
    max_laps_per_stint = get_tire_max_laps(stints, config)

    for i, ((compound_name, N), mode) in enumerate(zip(stints, stint_modes)):
        if N <= 0:
            return float('inf')

        compound = config.compounds[compound_name]
        pace_mode = config.get_pace_mode(mode)

        max_laps_base = max_laps_per_stint[i]
        adjusted_max_laps = floor(max_laps_base / pace_mode.degradation_factor)
        
        if N > adjusted_max_laps:
            return float('inf')  # Stint exceeds tire life for this mode

        # Base degradation adjusted for pace mode
        base_deg = compound.degradation_s_per_lap * pace_mode.degradation_factor

        # Progressive degradation: each stint starts fresh (lap 0)
        degradation_time = calculate_progressive_degradation(
            base_deg, 0, N, compound.max_competitive_laps
        )
        
        # Calculate time: base + degradation + pace adjustment
        stint_time = (N * compound.avg_lap_time_s) + degradation_time + (N * pace_mode.delta_per_lap)
        total_time += stint_time

    total_time += num_stops * config.pit_loss_seconds
    return total_time


def calculate_stint_ert_with_wear(
    compound: Compound,
    laps: int,
    starting_wear: float,
    mode: str,
    config: RaceConfig
) -> float:
    """
    Calculate ERT for a stint starting with pre-worn tires.
    Uses progressive degradation model where deg starts low and increases with wear.
    """
    pace_mode = config.get_pace_mode(mode)
    base_deg = compound.degradation_s_per_lap * pace_mode.degradation_factor
    
    # Base time
    base_time = laps * compound.avg_lap_time_s
    
    # Progressive degradation: starts low, increases toward tire cliff
    degradation_time = calculate_progressive_degradation(
        base_deg, starting_wear, laps, compound.max_competitive_laps
    )
    
    # Pace adjustment
    pace_time = laps * pace_mode.delta_per_lap
    
    return base_time + degradation_time + pace_time


def find_optimal_modes(
    stints: list[tuple[str, int]],
    config: RaceConfig
) -> tuple[tuple[str, ...], float]:
    """
    Find the optimal pace mode for each stint by trying all permutations.
    Returns (best_modes, best_ert) tuple.
    """
    num_stints = len(stints)
    best_modes = tuple(['normal'] * num_stints)
    best_ert = calculate_ert(stints, best_modes, config)

    for modes in product(PACE_MODES, repeat=num_stints):
        ert = calculate_ert(stints, modes, config)
        if ert < best_ert:
            best_ert = ert
            best_modes = modes

    return best_modes, best_ert


# --- Strategy Validation ---

def check_inventory(
    strategy_compounds: list[str],
    inventory: Inventory,
    require_medium_or_hard: bool,
    require_two_compounds: bool = True
) -> bool:
    """Check if a strategy's tire usage is covered by inventory."""
    usage = count_compound_usage(strategy_compounds)

    # Check total usage per compound
    if usage.get('Soft', 0) > inventory.get_total('Soft'):
        return False
    if usage.get('Medium', 0) > inventory.get_total('Medium'):
        return False
    if usage.get('Hard', 0) > inventory.hard_new:
        return False

    # Check scrubbed availability for double usage
    if usage.get('Soft', 0) == 2 and inventory.soft_scrubbed < 1:
        return False
    if usage.get('Medium', 0) == 2 and inventory.medium_scrubbed < 1:
        return False

    # Hard is assumed to only be used once
    if usage.get('Hard', 0) > 1:
        return False

    # Optional F1 regulation: Must use at least one Medium OR Hard tire
    if require_medium_or_hard:
        if usage.get('Medium', 0) == 0 and usage.get('Hard', 0) == 0:
            return False

    # F1 regulation: Must use at least 2 different compounds
    if require_two_compounds:
        unique_compounds = len(usage)
        if unique_compounds < 2:
            return False

    return True


def calculate_min_stint(config: RaceConfig) -> int:
    """
    Calculate minimum viable stint length based on pit loss.
    Uses pit_loss / 2 as baseline - ensures undercuts remain possible
    while preventing absurdly short stints.
    """
    return max(floor(config.pit_loss_seconds / 2), 8)

