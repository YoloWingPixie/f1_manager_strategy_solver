"""Strategy generation and analysis functions."""
import sys
from math import floor, ceil

from tqdm import tqdm

from models import (
    RaceConfig, Strategy, PACE_MODES, SCAnalysis,
    TireDomainAnalysis, TireDomain, CrossoverPoint,
    UndercutAnalysis, DRSAnalysis, AttackAnalysis, ModeScenario,
    LiveAnalysis, LiveStrategy
)
from core import (
    get_tire_max_laps, calculate_ert, find_optimal_modes, check_inventory,
    calculate_min_stint, calculate_effective_wear, calculate_remaining_competitive_laps,
    calculate_stint_ert_with_wear, calculate_lap_time_at_wear, estimate_tire_delta_per_lap,
    get_degradation_factor
)


# --- Race Strategy Generation ---

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
            if check_inventory(strategy_compounds, config.inventory, config.require_medium_or_hard, config.require_two_compounds):
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
                    # Don't check F1 regulations during recursion - strategy isn't complete yet
                    # Only check basic inventory availability
                    if check_inventory(strategy_compounds, config.inventory, require_medium_or_hard=False, require_two_compounds=False):
                        recurse_stints(new_stints, remaining_after, max_stops)

    print("Generating strategy combinations...", end=" ", flush=True)
    recurse_stints([], config.race_laps, config.max_pit_stops)
    print(f"found {len(all_strategies)} candidates.")

    # Deduplicate by stint split AND compound sequence
    # Key: (compound_sequence, stint_split) to keep different compound orders
    unique_stints: dict[tuple[tuple[str, ...], tuple[int, ...]], list[tuple[str, int]]] = {}
    for stints in all_strategies:
        compound_seq = tuple(s[0] for s in stints)
        stint_split = tuple(s[1] for s in stints)
        key = (compound_seq, stint_split)
        if key not in unique_stints:
            unique_stints[key] = stints

    # Calculate ERT with optimal modes for all unique strategies
    final_strategies: list[Strategy] = []

    for (compound_seq, stint_split), stints in tqdm(
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


def generate_continuation_strategies(
    remaining_laps: int,
    current_compound: str,
    current_wear: float,
    config: RaceConfig,
    use_sc_pit_loss: bool = False,
    include_stay_out: bool = True
) -> list[Strategy]:
    """
    Generate strategies to finish race from current position.
    
    If include_stay_out=True, includes option to continue on current tires.
    Pit loss is either SC pit loss or green flag pit loss.
    """
    pit_loss = config.sc_pit_loss_seconds if use_sc_pit_loss else config.pit_loss_seconds
    min_stint = max(floor(pit_loss / 2), 5) if not use_sc_pit_loss else 1
    
    compound = config.compounds[current_compound]
    remaining_on_current = calculate_remaining_competitive_laps(compound, current_wear, config)
    
    strategies: list[Strategy] = []
    compound_names = list(config.compounds.keys())
    
    # Option 1: Stay out and finish on current tires (no pit)
    if include_stay_out and remaining_on_current >= remaining_laps:
        # Can finish without pitting
        stints = [(current_compound, remaining_laps)]
        for mode in PACE_MODES:
            # Check if mode allows this stint length given wear
            adjusted_max = floor(remaining_on_current / config.get_pace_mode(mode).degradation_factor)
            if remaining_laps <= adjusted_max:
                ert = calculate_stint_ert_with_wear(
                    compound, remaining_laps, current_wear, mode, config
                )
                strategies.append(Strategy(
                    stints=stints,
                    optimal_modes=(mode,),
                    ert=ert
                ))
    
    # Option 2: Stay out for some laps, then pit for fresh tires
    if include_stay_out:
        for stay_laps in range(min_stint, min(remaining_on_current, remaining_laps - min_stint) + 1):
            laps_after_pit = remaining_laps - stay_laps
            if laps_after_pit < min_stint:
                continue
                
            for new_compound in compound_names:
                new_comp = config.compounds[new_compound]
                if laps_after_pit > new_comp.max_competitive_laps:
                    continue
                    
                stints = [(current_compound, stay_laps), (new_compound, laps_after_pit)]
                
                # Find best mode combination
                best_ert = float('inf')
                best_modes = ('normal', 'normal')
                
                for mode1 in PACE_MODES:
                    for mode2 in PACE_MODES:
                        # First stint with wear
                        ert1 = calculate_stint_ert_with_wear(
                            compound, stay_laps, current_wear, mode1, config
                        )
                        # Second stint fresh
                        ert2 = calculate_stint_ert_with_wear(
                            new_comp, laps_after_pit, 0, mode2, config
                        )
                        total = ert1 + ert2 + config.pit_loss_seconds  # Green flag pit
                        
                        if total < best_ert:
                            best_ert = total
                            best_modes = (mode1, mode2)
                
                if best_ert < float('inf'):
                    strategies.append(Strategy(
                        stints=stints,
                        optimal_modes=best_modes,
                        ert=best_ert
                    ))
    
    # Option 3: Pit immediately for fresh tires (only when not staying out)
    if not include_stay_out:
        for new_compound in compound_names:
            new_comp = config.compounds[new_compound]
            
            # Single stint to finish
            if remaining_laps <= new_comp.max_competitive_laps:
                stints = [(new_compound, remaining_laps)]
                best_ert = float('inf')
                best_mode = 'normal'
                
                for mode in PACE_MODES:
                    adjusted_max = floor(new_comp.max_competitive_laps / config.get_pace_mode(mode).degradation_factor)
                    if remaining_laps <= adjusted_max:
                        ert = calculate_stint_ert_with_wear(
                            new_comp, remaining_laps, 0, mode, config
                        ) + pit_loss
                        if ert < best_ert:
                            best_ert = ert
                            best_mode = mode
                
                if best_ert < float('inf'):
                    strategies.append(Strategy(
                        stints=stints,
                        optimal_modes=(best_mode,),
                        ert=best_ert
                    ))
            
            # Two stints after pit (pit now, then pit again later)
            for first_stint_laps in range(min_stint, min(remaining_laps - min_stint, new_comp.max_competitive_laps) + 1):
                second_stint_laps = remaining_laps - first_stint_laps
                if second_stint_laps < min_stint:
                    continue
                    
                for second_compound in compound_names:
                    second_comp = config.compounds[second_compound]
                    if second_stint_laps > second_comp.max_competitive_laps:
                        continue
                    
                    stints = [(new_compound, first_stint_laps), (second_compound, second_stint_laps)]
                    
                    best_ert = float('inf')
                    best_modes = ('normal', 'normal')
                    
                    for mode1 in PACE_MODES:
                        for mode2 in PACE_MODES:
                            ert1 = calculate_stint_ert_with_wear(new_comp, first_stint_laps, 0, mode1, config)
                            ert2 = calculate_stint_ert_with_wear(second_comp, second_stint_laps, 0, mode2, config)
                            # SC pit + green flag pit
                            total = ert1 + ert2 + pit_loss + config.pit_loss_seconds
                            
                            if total < best_ert:
                                best_ert = total
                                best_modes = (mode1, mode2)
                    
                    if best_ert < float('inf'):
                        strategies.append(Strategy(
                            stints=stints,
                            optimal_modes=best_modes,
                            ert=best_ert
                        ))
    
    return strategies


# --- Safety Car Analysis ---

def analyze_safety_car(
    stint_laps: int,
    remaining_laps: int,
    current_compound: str,
    config: RaceConfig,
    positions_at_risk: int = 0
) -> SCAnalysis:
    """
    Main safety car analysis - compare pit vs stay out options.
    
    Focuses on realistic comparison:
    1. Can you finish on current tires without pitting?
    2. If you must pit, how much does SC pit save vs green flag pit?
    3. What's the optimal strategy for each choice?
    4. Factor in track position loss from pitting
    """
    compound = config.compounds[current_compound]
    
    # Calculate tire wear (SC laps have reduced wear)
    effective_wear = calculate_effective_wear(stint_laps, config.sc_conserve_laps, config)
    remaining_competitive = calculate_remaining_competitive_laps(compound, effective_wear, config)
    wear_percent = (effective_wear / compound.max_competitive_laps) * 100
    
    # Key question: Can we finish without pitting?
    can_finish_no_pit = remaining_competitive >= remaining_laps
    
    # SC pit timing value (max savings from pit timing alone)
    sc_pit_value = config.pit_loss_seconds - config.sc_pit_loss_seconds
    
    # Position loss penalty when pitting
    position_penalty = positions_at_risk * config.position_loss_value
    
    # Generate STAY OUT strategies (continue on current tires)
    stay_out_strategies = generate_continuation_strategies(
        remaining_laps=remaining_laps,
        current_compound=current_compound,
        current_wear=effective_wear,
        config=config,
        use_sc_pit_loss=False,  # Any future pits are green flag
        include_stay_out=True
    )
    
    # Generate PIT NOW strategies - for SAME compound (fair comparison)
    pit_same_compound_strategies = []
    pit_same_stints = [(current_compound, remaining_laps)]
    if remaining_laps <= compound.max_competitive_laps:
        for mode in PACE_MODES:
            ert = calculate_stint_ert_with_wear(
                compound, remaining_laps, 0, mode, config
            ) + config.sc_pit_loss_seconds
            pit_same_compound_strategies.append(Strategy(
                stints=pit_same_stints,
                optimal_modes=(mode,),
                ert=ert
            ))
    
    # Generate PIT NOW strategies - optimal (any compound)
    pit_now_strategies = generate_continuation_strategies(
        remaining_laps=remaining_laps,
        current_compound=current_compound,
        current_wear=0,  # Fresh tires
        config=config,
        use_sc_pit_loss=True,  # SC pit loss
        include_stay_out=False  # Must pit
    )
    
    # Find best of each category
    best_stay_out = min(stay_out_strategies, key=lambda s: s.ert) if stay_out_strategies else None
    best_pit_same = min(pit_same_compound_strategies, key=lambda s: s.ert) if pit_same_compound_strategies else None
    best_pit_optimal = min(pit_now_strategies, key=lambda s: s.ert) if pit_now_strategies else None
    
    # Use best pit option (may be same compound or different)
    best_pit_now = best_pit_optimal
    
    stay_out_ert = best_stay_out.ert if best_stay_out else float('inf')
    pit_now_ert_raw = best_pit_now.ert if best_pit_now else float('inf')
    
    # Add position penalty to pit decision
    # This represents the cost of losing track position by pitting
    pit_now_ert_adjusted = pit_now_ert_raw + position_penalty
    
    # Time delta includes position penalty
    time_delta = stay_out_ert - pit_now_ert_adjusted
    
    # Recommendation (factoring in position loss)
    if pit_now_ert_adjusted < stay_out_ert:
        recommendation = "PIT"
    else:
        recommendation = "STAY_OUT"
    
    # War gaming: estimate pace deficit if staying out on worn tires
    # vs rivals who pit for fresh tires
    best_fresh_compound = min(
        config.compounds.values(),
        key=lambda c: c.avg_lap_time_s
    )
    pace_deficit = estimate_tire_delta_per_lap(
        compound, effective_wear, best_fresh_compound, config
    )
    total_time_loss = pace_deficit * remaining_laps
    
    # Risk assessment
    if pace_deficit < 0.2:
        risk = "LOW"
    elif pace_deficit < 0.5:
        risk = "MODERATE"
    else:
        risk = "HIGH"
    
    return SCAnalysis(
        current_compound=current_compound,
        stint_laps=stint_laps,
        remaining_laps=remaining_laps,
        current_wear_laps=floor(effective_wear),
        remaining_competitive_laps=remaining_competitive,
        tire_wear_percent=wear_percent,
        can_finish_no_pit=can_finish_no_pit,
        stay_out_strategy=best_stay_out,
        stay_out_ert=stay_out_ert,
        pit_now_strategy=best_pit_now,
        pit_now_ert=pit_now_ert_raw,  # Raw ERT without position penalty
        recommendation=recommendation,
        time_delta=time_delta,
        sc_pit_value=sc_pit_value,
        positions_lost=positions_at_risk,
        position_penalty=position_penalty,
        pace_deficit_per_lap=pace_deficit,
        total_time_loss=total_time_loss,
        risk_assessment=risk
    )


# --- Tire Domain Analysis ---

def analyze_tire_domains(
    config: RaceConfig,
    mode: str = 'normal',
    starting_wear: int = 0
) -> TireDomainAnalysis:
    """
    Analyze which tire compound is fastest at each lap of a stint.
    Identifies crossover points where one compound becomes faster than another.
    """
    compounds = list(config.compounds.values())
    
    # Find the maximum analysis lap (longest tire life)
    max_lap = max(c.max_competitive_laps for c in compounds)
    
    # Calculate lap time for each compound at each lap
    lap_times: dict[str, list[float]] = {}
    for compound in compounds:
        lap_times[compound.name] = []
        for lap in range(starting_wear, max_lap + 1):
            lap_time = calculate_lap_time_at_wear(compound, lap, mode, config)
            lap_times[compound.name].append(lap_time)
    
    # Find fastest compound at each lap
    fastest_at_lap: list[str] = []
    for lap_idx in range(len(lap_times[compounds[0].name])):
        fastest = min(compounds, key=lambda c: lap_times[c.name][lap_idx])
        fastest_at_lap.append(fastest.name)
    
    # Build domains (ranges where a compound is fastest)
    domains: list[TireDomain] = []
    crossover_points: list[CrossoverPoint] = []
    
    current_compound = fastest_at_lap[0]
    domain_start = starting_wear
    
    for lap_idx, fastest in enumerate(fastest_at_lap):
        actual_lap = starting_wear + lap_idx
        
        if fastest != current_compound:
            # End current domain
            prev_lap = actual_lap - 1
            domains.append(TireDomain(
                compound=current_compound,
                start_lap=domain_start,
                end_lap=prev_lap,
                start_laptime_s=lap_times[current_compound][domain_start - starting_wear],
                end_laptime_s=lap_times[current_compound][prev_lap - starting_wear]
            ))
            
            # Record crossover
            crossover_points.append(CrossoverPoint(
                lap=actual_lap,
                from_compound=current_compound,
                from_laptime_s=lap_times[current_compound][lap_idx],
                to_compound=fastest,
                to_laptime_s=lap_times[fastest][lap_idx]
            ))
            
            current_compound = fastest
            domain_start = actual_lap
    
    # Add final domain
    final_lap = starting_wear + len(fastest_at_lap) - 1
    domains.append(TireDomain(
        compound=current_compound,
        start_lap=domain_start,
        end_lap=final_lap,
        start_laptime_s=lap_times[current_compound][domain_start - starting_wear],
        end_laptime_s=lap_times[current_compound][final_lap - starting_wear]
    ))
    
    # Build compound details
    compound_details = {}
    for compound in compounds:
        compound_details[compound.name] = {
            'base_pace_s': compound.avg_lap_time_s,
            'degradation_s_per_lap': compound.degradation_s_per_lap,
            'max_competitive_laps': compound.max_competitive_laps
        }
    
    return TireDomainAnalysis(
        domains=domains,
        crossover_points=crossover_points,
        compound_details=compound_details,
        max_analysis_lap=max_lap
    )


# --- Undercut/Overcut Analysis ---

def analyze_undercut(
    gap_to_rival: float,
    current_lap: int,
    rival_pit_lap: int,
    your_compound: str,
    your_tire_laps: int,
    rival_compound: str,
    rival_tire_laps: int,
    pit_to_compound: str | None,
    config: RaceConfig
) -> UndercutAnalysis:
    """
    Analyze whether to undercut or overcut a rival.
    
    gap_to_rival: Positive = you're ahead, negative = you're behind
    """
    laps_until_rival_pits = rival_pit_lap - current_lap
    
    your_comp = config.compounds[your_compound]
    rival_comp = config.compounds[rival_compound]
    
    # Default to fastest compound if not specified
    if pit_to_compound is None:
        pit_to_compound = min(config.compounds.values(), key=lambda c: c.avg_lap_time_s).name
    pit_to_comp = config.compounds[pit_to_compound]
    
    # Calculate current wear percentages
    your_wear_pct = (your_tire_laps / your_comp.max_competitive_laps) * 100
    rival_wear_pct = (rival_tire_laps / rival_comp.max_competitive_laps) * 100
    
    # Calculate pace on worn vs fresh tires
    your_current_pace = calculate_lap_time_at_wear(your_comp, your_tire_laps, 'normal', config)
    fresh_pace = calculate_lap_time_at_wear(pit_to_comp, 0, 'normal', config)
    rival_worn_pace = calculate_lap_time_at_wear(rival_comp, rival_tire_laps, 'normal', config)
    
    fresh_tire_advantage = your_current_pace - fresh_pace  # How much faster you'd be on fresh
    
    # UNDERCUT SCENARIO: You pit now, rival pits at rival_pit_lap
    # You're on fresh tires for laps_until_rival_pits laps
    undercut_window_laps = laps_until_rival_pits
    
    # Time gained during undercut window
    # Each lap: you're on progressively aging fresh tires, rival is on progressively aging worn tires
    time_gained_undercut = 0.0
    for i in range(undercut_window_laps):
        your_pace_this_lap = calculate_lap_time_at_wear(pit_to_comp, i, 'normal', config)
        your_pace_this_lap -= config.inlap_push_gain if i == 0 else 0  # In-lap push (you just pitted)
        
        rival_pace_this_lap = calculate_lap_time_at_wear(
            rival_comp, rival_tire_laps + i, 'normal', config
        )
        time_gained_undercut += rival_pace_this_lap - your_pace_this_lap
    
    # Account for out-lap penalty (your first lap after pit)
    time_gained_undercut -= config.outlap_penalty
    
    # Projected gap after undercut
    # Current gap + time gained - pit delta (both pit once, so net pit delta = 0 over both sequences)
    projected_gap_undercut = gap_to_rival + time_gained_undercut
    undercut_viable = projected_gap_undercut > 0  # You'd be ahead
    
    # OVERCUT SCENARIO: Rival pits first, you stay out
    # Rival is on fresh tires, you're on worn tires for laps_until_rival_pits laps
    time_lost_staying_out = 0.0
    for i in range(undercut_window_laps):
        your_pace_this_lap = calculate_lap_time_at_wear(
            your_comp, your_tire_laps + i, 'normal', config
        )
        # Rival: fresh tires progressively wearing, with out-lap penalty on first lap
        rival_pace_this_lap = calculate_lap_time_at_wear(pit_to_comp, i, 'normal', config)
        if i == 0:
            rival_pace_this_lap += config.outlap_penalty
        
        time_lost_staying_out += your_pace_this_lap - rival_pace_this_lap
    
    projected_gap_overcut = gap_to_rival - time_lost_staying_out
    overcut_viable = projected_gap_overcut > 0  # You'd still be ahead
    
    # Determine recommendation
    if gap_to_rival < 0:  # You're behind
        if undercut_viable:
            recommendation = "UNDERCUT"
            reason = f"Undercut gains {time_gained_undercut:.1f}s, puts you {projected_gap_undercut:.1f}s ahead"
        else:
            recommendation = "STAY_OUT"
            reason = f"Undercut only gains {time_gained_undercut:.1f}s, not enough to jump rival"
    else:  # You're ahead
        if not overcut_viable and undercut_viable:
            recommendation = "UNDERCUT"
            reason = f"Cover the undercut - staying out loses {time_lost_staying_out:.1f}s"
        elif overcut_viable:
            recommendation = "STAY_OUT"
            reason = f"Track position safe - stay out and pit after rival"
        else:
            recommendation = "UNDERCUT"
            reason = "Pit to defend - rival's undercut would work"
    
    return UndercutAnalysis(
        gap_to_rival=gap_to_rival,
        current_lap=current_lap,
        rival_pit_lap=rival_pit_lap,
        laps_until_rival_pits=laps_until_rival_pits,
        your_compound=your_compound,
        your_tire_laps=your_tire_laps,
        your_wear_percent=your_wear_pct,
        rival_compound=rival_compound,
        rival_tire_laps=rival_tire_laps,
        rival_wear_percent=rival_wear_pct,
        pit_to_compound=pit_to_compound,
        undercut_viable=undercut_viable,
        fresh_tire_advantage=fresh_tire_advantage,
        undercut_window_laps=undercut_window_laps,
        time_gained_undercut=time_gained_undercut,
        projected_gap_after_undercut=projected_gap_undercut,
        overcut_viable=overcut_viable,
        time_lost_staying_out=time_lost_staying_out,
        projected_gap_after_overcut=projected_gap_overcut,
        recommendation=recommendation,
        recommendation_reason=reason
    )


# --- DRS Defense Analysis ---

def simulate_gap_scenario(
    initial_gap: float,
    stint_laps: int,
    your_compound,  # Compound
    your_start_wear: int,
    attacker_compound,  # Compound
    attacker_start_wear: int,
    your_modes: list[tuple[str, int]],  # [(mode, laps), ...]
    config: RaceConfig
) -> ModeScenario:
    """Simulate a gap defense scenario with specified pace modes."""
    gap = initial_gap
    your_wear = your_start_wear
    attacker_wear = attacker_start_wear
    
    mode_idx = 0
    mode_laps_remaining = your_modes[0][1] if your_modes else 0
    current_mode = your_modes[0][0] if your_modes else 'normal'
    
    cliff_lap = None
    
    for lap in range(stint_laps):
        # Advance to next mode if needed
        while mode_laps_remaining == 0 and mode_idx < len(your_modes) - 1:
            mode_idx += 1
            current_mode = your_modes[mode_idx][0]
            mode_laps_remaining = your_modes[mode_idx][1]
        
        # Calculate pace for this lap
        your_pace = calculate_lap_time_at_wear(your_compound, your_wear, current_mode, config)
        attacker_pace = calculate_lap_time_at_wear(attacker_compound, attacker_wear, 'normal', config)
        
        # Update gap (positive = you're ahead)
        gap -= (your_pace - attacker_pace)
        
        # Update wear
        pace_mode = config.get_pace_mode(current_mode)
        your_wear += pace_mode.degradation_factor
        attacker_wear += 1  # Attacker on normal mode
        
        mode_laps_remaining -= 1
        
        # Check for cliff
        if cliff_lap is None and your_wear >= your_compound.max_competitive_laps:
            cliff_lap = lap + your_start_wear
    
    # Calculate final state
    tire_wear_at_end = your_wear
    tire_percent = (your_wear / your_compound.max_competitive_laps) * 100
    exceeds_life = your_wear > your_compound.max_competitive_laps
    sustainable = gap > 0 and not exceeds_life
    
    # Build description
    mode_str = " -> ".join(f"{m}({l})" for m, l in your_modes)
    
    return ModeScenario(
        name=your_modes[0][0].upper() if len(your_modes) == 1 else "BURST_PUSH",
        mode_sequence=your_modes,
        final_gap=gap,
        tire_wear_at_end=tire_wear_at_end,
        tire_percent_at_end=tire_percent,
        exceeds_tire_life=exceeds_life,
        cliff_lap=cliff_lap,
        sustainable=sustainable,
        description=mode_str
    )


def analyze_drs_defense(
    gap_to_attacker: float,
    stint_laps_remaining: int,
    your_compound: str,
    your_tire_laps: int,
    attacker_compound: str,
    attacker_tire_laps: int,
    config: RaceConfig
) -> DRSAnalysis:
    """
    Analyze DRS defense options: push to escape, conserve, or burst then conserve.
    """
    your_comp = config.compounds[your_compound]
    attacker_comp = config.compounds[attacker_compound]
    
    in_drs = gap_to_attacker < config.drs_threshold_seconds
    
    your_wear_pct = (your_tire_laps / your_comp.max_competitive_laps) * 100
    attacker_wear_pct = (attacker_tire_laps / attacker_comp.max_competitive_laps) * 100
    
    # Base pace comparison (both on normal mode)
    your_pace = calculate_lap_time_at_wear(your_comp, your_tire_laps, 'normal', config)
    attacker_pace = calculate_lap_time_at_wear(attacker_comp, attacker_tire_laps, 'normal', config)
    base_pace_delta = your_pace - attacker_pace  # Positive = you're slower
    
    scenarios: list[ModeScenario] = []
    
    # Scenario 1: CONSERVE (accept DRS, defend on track)
    conserve_scenario = simulate_gap_scenario(
        initial_gap=gap_to_attacker,
        stint_laps=stint_laps_remaining,
        your_compound=your_comp,
        your_start_wear=your_tire_laps,
        attacker_compound=attacker_comp,
        attacker_start_wear=attacker_tire_laps,
        your_modes=[('conserve', stint_laps_remaining)],
        config=config
    )
    conserve_scenario.name = "CONSERVE"
    scenarios.append(conserve_scenario)
    
    # Scenario 2: PUSH all stint
    push_scenario = simulate_gap_scenario(
        initial_gap=gap_to_attacker,
        stint_laps=stint_laps_remaining,
        your_compound=your_comp,
        your_start_wear=your_tire_laps,
        attacker_compound=attacker_comp,
        attacker_start_wear=attacker_tire_laps,
        your_modes=[('push', stint_laps_remaining)],
        config=config
    )
    push_scenario.name = "PUSH"
    scenarios.append(push_scenario)
    
    # Scenario 3: Find optimal burst push duration
    best_burst = None
    best_burst_gap = -float('inf')
    
    for push_laps in range(1, stint_laps_remaining):
        conserve_laps = stint_laps_remaining - push_laps
        burst_scenario = simulate_gap_scenario(
            initial_gap=gap_to_attacker,
            stint_laps=stint_laps_remaining,
            your_compound=your_comp,
            your_start_wear=your_tire_laps,
            attacker_compound=attacker_comp,
            attacker_start_wear=attacker_tire_laps,
            your_modes=[('push', push_laps), ('conserve', conserve_laps)],
            config=config
        )
        
        # Prefer sustainable scenarios with best final gap
        if burst_scenario.sustainable and burst_scenario.final_gap > best_burst_gap:
            best_burst = burst_scenario
            best_burst_gap = burst_scenario.final_gap
            best_burst.name = f"BURST_PUSH ({push_laps} laps)"
    
    optimal_push_laps = 0
    if best_burst:
        optimal_push_laps = best_burst.mode_sequence[0][1]
        scenarios.append(best_burst)
    
    # Determine recommendation
    sustainable_scenarios = [s for s in scenarios if s.sustainable]
    if sustainable_scenarios:
        # Pick the one with best final gap
        recommended = max(sustainable_scenarios, key=lambda s: s.final_gap)
    else:
        # Nothing is sustainable, pick least bad option
        recommended = max(scenarios, key=lambda s: s.final_gap)
    
    recommendation = recommended.name.split()[0]  # "BURST_PUSH (3 laps)" -> "BURST_PUSH"
    
    return DRSAnalysis(
        gap_to_attacker=gap_to_attacker,
        stint_laps_remaining=stint_laps_remaining,
        in_drs_range=in_drs,
        your_compound=your_compound,
        your_tire_laps=your_tire_laps,
        your_wear_percent=your_wear_pct,
        your_max_competitive_laps=your_comp.max_competitive_laps,
        attacker_compound=attacker_compound,
        attacker_tire_laps=attacker_tire_laps,
        attacker_wear_percent=attacker_wear_pct,
        base_pace_delta=base_pace_delta,
        scenarios=scenarios,
        optimal_push_laps=optimal_push_laps,
        recommendation=recommendation,
        recommended_scenario=recommended
    )


# --- Attack/Catch Analysis ---

def simulate_attack_scenario(
    initial_gap: float,
    stint_laps: int,
    your_compound,  # Compound
    your_start_wear: int,
    target_compound,  # Compound
    target_start_wear: int,
    your_modes: list[tuple[str, int]],
    drs_threshold: float,
    config: RaceConfig
) -> ModeScenario:
    """Simulate an attack scenario, tracking gap to target."""
    gap = initial_gap
    your_wear = float(your_start_wear)
    target_wear = float(target_start_wear)
    
    mode_idx = 0
    mode_laps_remaining = your_modes[0][1] if your_modes else 0
    current_mode = your_modes[0][0] if your_modes else 'normal'
    
    cliff_lap = None
    laps_to_drs = None
    
    for lap in range(stint_laps):
        # Advance to next mode if needed
        while mode_laps_remaining == 0 and mode_idx < len(your_modes) - 1:
            mode_idx += 1
            current_mode = your_modes[mode_idx][0]
            mode_laps_remaining = your_modes[mode_idx][1]
        
        # Calculate pace
        your_pace = calculate_lap_time_at_wear(your_compound, int(your_wear), current_mode, config)
        target_pace = calculate_lap_time_at_wear(target_compound, int(target_wear), 'normal', config)
        
        # Update gap (we want to close it, so subtract our gain)
        gap -= (target_pace - your_pace)
        
        # Check if we've reached DRS
        if laps_to_drs is None and gap <= drs_threshold:
            laps_to_drs = lap + 1
        
        # Update wear
        pace_mode = config.get_pace_mode(current_mode)
        your_wear += pace_mode.degradation_factor
        target_wear += 1
        
        mode_laps_remaining -= 1
        
        # Check for cliff
        if cliff_lap is None and your_wear >= your_compound.max_competitive_laps:
            cliff_lap = lap + your_start_wear
    
    tire_percent = (your_wear / your_compound.max_competitive_laps) * 100
    exceeds_life = your_wear > your_compound.max_competitive_laps
    sustainable = gap <= drs_threshold and not exceeds_life
    
    mode_str = " -> ".join(f"{m}({l})" for m, l in your_modes)
    
    return ModeScenario(
        name=your_modes[0][0].upper(),
        mode_sequence=your_modes,
        final_gap=gap,
        tire_wear_at_end=your_wear,
        tire_percent_at_end=tire_percent,
        exceeds_tire_life=exceeds_life,
        cliff_lap=cliff_lap,
        sustainable=sustainable,
        description=mode_str
    )


def analyze_attack(
    gap_to_target: float,
    stint_laps_remaining: int,
    your_compound: str,
    your_tire_laps: int,
    target_compound: str,
    target_tire_laps: int,
    config: RaceConfig
) -> AttackAnalysis:
    """
    Analyze whether to attack to catch car ahead or stay on plan.
    """
    your_comp = config.compounds[your_compound]
    target_comp = config.compounds[target_compound]
    drs_threshold = config.drs_threshold_seconds
    
    your_wear_pct = (your_tire_laps / your_comp.max_competitive_laps) * 100
    target_wear_pct = (target_tire_laps / target_comp.max_competitive_laps) * 100
    
    # Calculate natural convergence rate
    your_pace = calculate_lap_time_at_wear(your_comp, your_tire_laps, 'normal', config)
    target_pace = calculate_lap_time_at_wear(target_comp, target_tire_laps, 'normal', config)
    natural_closing_rate = target_pace - your_pace  # Positive = closing
    
    # Can we reach DRS naturally?
    if natural_closing_rate > 0:
        laps_to_drs_natural = ceil((gap_to_target - drs_threshold) / natural_closing_rate)
        can_reach_naturally = laps_to_drs_natural <= stint_laps_remaining
    else:
        laps_to_drs_natural = None
        can_reach_naturally = False
    
    scenarios: list[ModeScenario] = []
    
    # Scenario 1: STAY ON PLAN (normal mode)
    stay_scenario = simulate_attack_scenario(
        initial_gap=gap_to_target,
        stint_laps=stint_laps_remaining,
        your_compound=your_comp,
        your_start_wear=your_tire_laps,
        target_compound=target_comp,
        target_start_wear=target_tire_laps,
        your_modes=[('normal', stint_laps_remaining)],
        drs_threshold=drs_threshold,
        config=config
    )
    stay_scenario.name = "STAY_ON_PLAN"
    scenarios.append(stay_scenario)
    
    # Scenario 2: PUSH mode
    push_scenario = simulate_attack_scenario(
        initial_gap=gap_to_target,
        stint_laps=stint_laps_remaining,
        your_compound=your_comp,
        your_start_wear=your_tire_laps,
        target_compound=target_comp,
        target_start_wear=target_tire_laps,
        your_modes=[('push', stint_laps_remaining)],
        drs_threshold=drs_threshold,
        config=config
    )
    push_scenario.name = "PUSH"
    scenarios.append(push_scenario)
    
    # Scenario 3: ATTACK mode (if available)
    if config.has_attack_mode():
        attack_scenario = simulate_attack_scenario(
            initial_gap=gap_to_target,
            stint_laps=stint_laps_remaining,
            your_compound=your_comp,
            your_start_wear=your_tire_laps,
            target_compound=target_comp,
            target_start_wear=target_tire_laps,
            your_modes=[('attack', stint_laps_remaining)],
            drs_threshold=drs_threshold,
            config=config
        )
        attack_scenario.name = "ATTACK"
        scenarios.append(attack_scenario)
    
    # Determine recommendation and tire warning
    tire_warning = None
    
    # Check if attack would burn tires by stint end
    for scenario in scenarios:
        if scenario.exceeds_tire_life:
            tire_warning = f"{scenario.name} would exceed tire life by stint end"
            break
    
    # Recommendation logic:
    # 1. If natural convergence reaches DRS, stay on plan
    # 2. If push reaches DRS sustainably, push
    # 3. If only attack reaches DRS, attack (with warning if needed)
    # 4. Otherwise, stay on plan
    
    recommended = scenarios[0]  # Default to stay on plan
    
    if can_reach_naturally:
        recommended = scenarios[0]
        recommendation = "STAY_ON_PLAN"
    else:
        # Find scenarios that reach DRS
        reaching_scenarios = [s for s in scenarios if s.final_gap <= drs_threshold]
        sustainable_reaching = [s for s in reaching_scenarios if s.sustainable]
        
        if sustainable_reaching:
            # Pick least aggressive that works
            for name in ["STAY_ON_PLAN", "PUSH", "ATTACK"]:
                match = [s for s in sustainable_reaching if s.name == name]
                if match:
                    recommended = match[0]
                    break
        elif reaching_scenarios:
            # Nothing sustainable, but something reaches - warn about tires
            recommended = reaching_scenarios[0]
            tire_warning = f"{recommended.name} reaches DRS but burns tires"
        
        recommendation = recommended.name
    
    return AttackAnalysis(
        gap_to_target=gap_to_target,
        stint_laps_remaining=stint_laps_remaining,
        drs_threshold=drs_threshold,
        your_compound=your_compound,
        your_tire_laps=your_tire_laps,
        your_wear_percent=your_wear_pct,
        your_max_competitive_laps=your_comp.max_competitive_laps,
        target_compound=target_compound,
        target_tire_laps=target_tire_laps,
        target_wear_percent=target_wear_pct,
        natural_closing_rate=natural_closing_rate,
        laps_to_drs_natural=laps_to_drs_natural,
        can_reach_drs_naturally=can_reach_naturally,
        scenarios=scenarios,
        recommendation=recommendation,
        recommended_scenario=recommended,
        tire_warning=tire_warning
    )


# --- Live Mid-Race Strategy Analysis ---

def analyze_live_strategy(
    current_lap: int,
    current_compound: str,
    tire_laps: int,
    remaining_laps: int,
    config: RaceConfig
) -> LiveAnalysis:
    """
    Analyze optimal strategy from current mid-race position.
    
    This is for "what should I do from HERE?" decisions - works for:
    - Red flag restarts (with tire wear already accumulated)
    - Mid-race strategy changes
    - General "where should I pit?" questions
    """
    compound = config.compounds[current_compound]
    
    # Calculate current tire state
    wear_percent = (tire_laps / compound.max_competitive_laps) * 100
    remaining_competitive = calculate_remaining_competitive_laps(compound, tire_laps, config)
    can_finish_no_pit = remaining_competitive >= remaining_laps
    
    # Generate all continuation strategies
    strategies = generate_continuation_strategies(
        remaining_laps=remaining_laps,
        current_compound=current_compound,
        current_wear=float(tire_laps),
        config=config,
        use_sc_pit_loss=False,
        include_stay_out=True
    )
    
    if not strategies:
        # Fallback - shouldn't happen with valid inputs
        raise ValueError("No valid strategies found for given inputs")
    
    # Convert to LiveStrategy format with additional info
    live_strategies: list[LiveStrategy] = []
    
    for strat in strategies:
        # Determine next pit info
        pit_laps = strat.get_pit_laps()
        if pit_laps:
            next_pit_lap = pit_laps[0]  # This is relative to "now", so it's "in X laps"
            next_compound = strat.compound_sequence[1] if len(strat.compound_sequence) > 1 else None
        else:
            next_pit_lap = None
            next_compound = None
        
        live_strategies.append(LiveStrategy(
            strategy=strat,
            next_pit_lap=next_pit_lap,
            next_compound=next_compound,
            remaining_on_current=remaining_competitive,
            ert_to_finish=strat.ert
        ))
    
    # Sort by ERT
    live_strategies.sort(key=lambda s: s.ert_to_finish)
    
    # Deduplicate by strategy string (keep best for each compound sequence + mode)
    seen_keys: set[str] = set()
    unique_strategies: list[LiveStrategy] = []
    for ls in live_strategies:
        key = ls.strategy.format_strategy_string()
        if key not in seen_keys:
            seen_keys.add(key)
            unique_strategies.append(ls)
    
    # Limit to top strategies
    top_live = unique_strategies[:config.top_strategies]
    
    # Find no-pit option ERT if exists
    no_pit_ert = None
    for ls in top_live:
        if ls.next_pit_lap is None:
            no_pit_ert = ls.ert_to_finish
            break
    
    return LiveAnalysis(
        current_lap=current_lap,
        current_compound=current_compound,
        tire_laps=tire_laps,
        remaining_laps=remaining_laps,
        tire_wear_percent=wear_percent,
        remaining_competitive_laps=remaining_competitive,
        can_finish_no_pit=can_finish_no_pit,
        no_pit_ert=no_pit_ert,
        strategies=top_live,
        recommended=top_live[0]
    )

