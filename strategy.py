"""Strategy generation, ERT calculation, and optimization logic."""
import sys
from math import floor
from itertools import product

from tqdm import tqdm

from models import RaceConfig, Strategy, Compound, Inventory, PACE_MODES


def get_tire_max_laps(
    stints: list[tuple[str, int]],
    config: RaceConfig
) -> list[int]:
    """
    Determine max competitive laps for each stint based on tire type (new vs scrubbed).
    New tires are used first, then scrubbed. Scrubbed have reduced life.
    """
    usage: dict[str, int] = {}
    max_laps_list = []

    for compound_name, _ in stints:
        usage[compound_name] = usage.get(compound_name, 0) + 1
        times_used = usage[compound_name]

        base_max = config.compounds[compound_name].max_competitive_laps
        new_count = config.inventory.get_new_count(compound_name)

        if times_used <= new_count:
            max_laps_list.append(base_max)
        else:
            max_laps_list.append(base_max - config.scrubbed_life_penalty)

    return max_laps_list


def calculate_ert(
    stints: list[tuple[str, int]],
    stint_modes: tuple[str, ...],
    config: RaceConfig
) -> float:
    """
    Calculate Estimated Race Time (ERT) for a strategy with per-stint pace modes.
    
    Formula: Stint Time = (N * Base Lap Time) + (D_adjusted * N * (N-1) / 2) + (N * pace_delta)
    
    Push increases degradation, Conserve decreases it.
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

        # Adjust degradation based on mode
        D = compound.degradation_s_per_lap * pace_mode.degradation_factor

        # Calculate time: base + degradation + pace adjustment
        degradation_time = D * N * (N - 1) / 2
        stint_time = (N * compound.avg_lap_time_s) + degradation_time + (N * pace_mode.delta_per_lap)
        total_time += stint_time

    total_time += num_stops * config.pit_loss_seconds
    return total_time


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


def check_inventory(
    strategy_compounds: list[str],
    inventory: Inventory,
    require_medium_or_hard: bool
) -> bool:
    """Check if a strategy's tire usage is covered by inventory."""
    usage: dict[str, int] = {}
    for compound in strategy_compounds:
        usage[compound] = usage.get(compound, 0) + 1

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

    return True


def calculate_min_stint(config: RaceConfig) -> int:
    """
    Calculate minimum viable stint length based on pit loss.
    Uses pit_loss / 2 as baseline - ensures undercuts remain possible
    while preventing absurdly short stints.
    """
    return max(floor(config.pit_loss_seconds / 2), 8)


def generate_strategies(config: RaceConfig) -> list[Strategy]:
    """Generate all valid strategies up to max_pit_stops."""
    # Calculate or use configured min stint
    if config.min_stint_laps == 'auto':
        min_stint = calculate_min_stint(config)
    else:
        min_stint = int(config.min_stint_laps)

    # First stint is FREE (no pit stop to pay off) - allow shorter for tactical undercuts
    first_stint_min = 8
    compound_names = list(config.compounds.keys())
    all_strategies: list[list[tuple[str, int]]] = []

    def recurse_stints(
        current_stints: list[tuple[str, int]],
        laps_remaining: int,
        max_stops: int
    ) -> None:
        current_stops = len(current_stints) - 1

        # Base Case 1: Laps complete (End of race)
        if laps_remaining == 0:
            strategy_compounds = [s[0] for s in current_stints]
            if check_inventory(strategy_compounds, config.inventory, config.require_medium_or_hard):
                if current_stops <= max_stops:
                    all_strategies.append(current_stints)
            return

        # Base Case 2: Max stops reached but race incomplete
        if current_stops >= max_stops and laps_remaining > 0:
            return

        # Recursive Step: Try all compounds and valid stint lengths
        if current_stops < max_stops:
            for comp_name in compound_names:
                L_max = config.compounds[comp_name].max_competitive_laps

                is_first_stint = len(current_stints) == 0
                min_this_stint = first_stint_min if is_first_stint else min_stint

                max_this_stint = min(laps_remaining, L_max)
                for N in range(min_this_stint, max_this_stint + 1, config.stint_lap_step):
                    remaining_after = laps_remaining - N
                    if 0 < remaining_after < min_stint:
                        continue

                    new_stints = current_stints + [(comp_name, N)]
                    strategy_compounds = [s[0] for s in new_stints]
                    if check_inventory(strategy_compounds, config.inventory, config.require_medium_or_hard):
                        recurse_stints(new_stints, remaining_after, max_stops)

    print("Generating strategy combinations...", end=" ", flush=True)
    recurse_stints([], config.race_laps, config.max_pit_stops)
    print(f"found {len(all_strategies)} candidates.")

    # Deduplicate by stint split
    unique_stints: dict[tuple[int, ...], list[tuple[str, int]]] = {}
    for stints in all_strategies:
        stint_split = tuple(s[1] for s in stints)
        if stint_split not in unique_stints:
            unique_stints[stint_split] = stints

    # Calculate ERT with optimal modes for all unique strategies
    final_strategies: list[Strategy] = []

    for stint_split, stints in tqdm(
        unique_stints.items(),
        desc="Optimizing pace modes",
        unit="strategy",
        file=sys.stdout
    ):
        optimal_modes, optimal_ert = find_optimal_modes(stints, config)
        final_strategies.append(Strategy(
            stints=stints,
            optimal_modes=optimal_modes,
            ert=optimal_ert
        ))

    return final_strategies


def find_clean_air_strategy(
    config: RaceConfig,
    top_strategies: list[Strategy],
    all_strategies: list[Strategy]
) -> Strategy | None:
    """
    Identify the Clean Air strategy with rules:
    1. First pit MUST undercut the earliest pit of ALL top strategies
    2. Second pit (if exists) must NOT land on ANY top strategy pit lap
    3. Prefer 1-stop if within 102% of best 2-stop time
    4. Max 2 stops for clean air
    """
    if not top_strategies:
        return None

    # Collect ALL pit stop laps from top strategies
    top_pit_laps: set[int] = set()
    for strat in top_strategies:
        top_pit_laps.update(strat.get_pit_laps())

    # Find the earliest first pit stop among ALL top strategies
    earliest_first_pit = min(strat.stints[0][1] for strat in top_strategies)

    def is_valid_clean_air(strat: Strategy) -> bool:
        """Check if strategy meets clean air requirements."""
        pits = strat.get_pit_laps()
        if not pits:
            return False

        # Rule 1: First pit must undercut earliest top strategy pit
        if pits[0] >= earliest_first_pit:
            return False

        # Rule 2: Second pit (if exists) must not land on any top strategy pit lap
        if len(pits) >= 2 and pits[1] in top_pit_laps:
            return False

        return True

    # Find best 1-stop and 2-stop candidates
    one_stop_candidates = [s for s in all_strategies if s.pit_stops == 1 and is_valid_clean_air(s)]
    two_stop_candidates = [s for s in all_strategies if s.pit_stops == 2 and is_valid_clean_air(s)]

    best_1stop = min(one_stop_candidates, key=lambda s: s.ert) if one_stop_candidates else None
    best_2stop = min(two_stop_candidates, key=lambda s: s.ert) if two_stop_candidates else None

    # Rule 3: Prefer 1-stop if within 102% of 2-stop time
    if best_1stop and best_2stop:
        if best_1stop.ert <= best_2stop.ert * 1.02:
            return best_1stop
        return best_2stop
    
    return best_1stop or best_2stop

