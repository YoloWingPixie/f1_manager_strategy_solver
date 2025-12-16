"""F1 Race Strategy Optimizer - CLI Entry Point."""
import yaml
import argparse
import re
from math import floor

from tabulate import tabulate

from models import (
    RaceConfig, Compound, Inventory, PaceMode, Strategy, SCAnalysis,
    TireDomainAnalysis, TireDomain, CrossoverPoint,
    UndercutAnalysis, DRSAnalysis, AttackAnalysis, ModeScenario,
    LiveAnalysis, LiveStrategy
)
from strategy import (
    generate_strategies, find_clean_air_strategy, calculate_min_stint, analyze_safety_car,
    analyze_tire_domains, analyze_undercut, analyze_drs_defense, analyze_attack,
    analyze_live_strategy
)


def parse_laptime(value) -> float:
    """
    Parse a lap time value that can be either:
    - A string like "1:31.209" or "01:31.209" (M:SS.mmm or MM:SS.mmm)
    - A number (already in seconds)
    Returns time in seconds as a float.
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        match = re.match(r'^(\d+):(\d+(?:\.\d+)?)$', value.strip())
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            return minutes * 60 + seconds
        
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot parse lap time: {value}")
    
    raise ValueError(f"Invalid lap time type: {type(value)}")


def load_config(path: str) -> RaceConfig:
    """Load and parse YAML configuration into RaceConfig."""
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    # Parse compounds
    compounds: dict[str, Compound] = {}
    for name, data in raw['compounds'].items():
        avg_time = parse_laptime(data.get('avg_lap_time', data.get('avg_lap_time_s')))
        compounds[name] = Compound(
            name=name,
            avg_lap_time_s=avg_time,
            degradation_s_per_lap=data['degradation_s_per_lap'],
            max_competitive_laps=data['max_competitive_laps']
        )

    # Parse inventory
    inv_raw = raw['inventory']
    inventory = Inventory(
        soft_new=inv_raw.get('soft_new', 0),
        soft_scrubbed=inv_raw.get('soft_scrubbed', 0),
        medium_new=inv_raw.get('medium_new', 0),
        medium_scrubbed=inv_raw.get('medium_scrubbed', 0),
        hard_new=inv_raw.get('hard_new', 0),
        hard_scrubbed=inv_raw.get('hard_scrubbed', 0)
    )

    # Parse pace modes
    pace_modes: dict[str, PaceMode] = {}
    for mode_name, mode_data in raw['pace_modes'].items():
        pace_modes[mode_name] = PaceMode(
            name=mode_name,
            delta_per_lap=mode_data['delta_per_lap'],
            degradation_factor=mode_data['degradation_factor']
        )

    return RaceConfig(
        race_laps=raw['race_laps'],
        pit_loss_seconds=raw['pit_loss_seconds'],
        compounds=compounds,
        inventory=inventory,
        pace_modes=pace_modes,
        top_strategies=raw.get('top_strategies', 5),
        max_pit_stops=raw.get('max_pit_stops', 3),
        min_stint_laps=raw.get('min_stint_laps', 'auto'),
        stint_lap_step=raw.get('stint_lap_step', 1),
        scrubbed_life_penalty=raw.get('scrubbed_life_penalty', 3),
        require_medium_or_hard=raw.get('require_medium_or_hard', True),
        sc_pit_loss_seconds=raw.get('sc_pit_loss_seconds', 5.0),
        sc_conserve_laps=raw.get('sc_conserve_laps', 3),
        sc_conserve_factor=raw.get('sc_conserve_factor', 0.5),
        position_loss_value=raw.get('position_loss_value', 2.5),
        drs_threshold_seconds=raw.get('drs_threshold_seconds', 1.0),
        dirty_air_loss_per_lap=raw.get('dirty_air_loss_per_lap', 0.5),
        inlap_push_gain=raw.get('inlap_push_gain', 0.3),
        outlap_penalty=raw.get('outlap_penalty', 1.5)
    )


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.ms format."""
    hours = floor(seconds / 3600)
    seconds %= 3600
    minutes = floor(seconds / 60)
    seconds %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def print_top_strategies(strategies: list[Strategy], config: RaceConfig) -> None:
    """Print the top N strategies table."""
    push_cfg = config.pace_modes['push']
    conserve_cfg = config.pace_modes['conserve']

    print("\n\n" + "=" * 80)
    print(f"TOP {config.top_strategies} OPTIMAL RACE STRATEGIES (Ranked by Optimized ERT)")
    print(f"Push: {push_cfg.delta_per_lap:+.2f}s/lap, +{(push_cfg.degradation_factor-1)*100:.0f}% deg | "
          f"Conserve: {conserve_cfg.delta_per_lap:+.2f}s/lap, {(conserve_cfg.degradation_factor-1)*100:.0f}% deg")
    print("=" * 80)

    output_table = []
    for i, strat in enumerate(strategies[:config.top_strategies]):
        output_table.append([
            i + 1,
            strat.format_strategy_string(),
            strat.pit_stops,
            strat.format_split_string(),
            strat.format_pit_laps(),
            format_time(strat.ert)
        ])

    headers = ["Rank", "Strategy (Mode)", "Stops", "Stint Split", "Pit Laps", "ERT"]
    print(tabulate(output_table, headers=headers, tablefmt="github"))


def print_clean_air_strategy(
    clean_air_strat: Strategy,
    top_strategies: list[Strategy],
    best_ert: float,
    config: RaceConfig
) -> None:
    """Print the clean air strategy details."""
    # Get earliest first pit from top strategies
    earliest_first_pit = min(s.stints[0][1] for s in top_strategies[:config.top_strategies])

    # Collect pit laps from top strategies
    top_pit_laps: set[int] = set()
    for strat in top_strategies[:config.top_strategies]:
        top_pit_laps.update(strat.get_pit_laps())

    print("\n\n" + "=" * 80)
    print("BIAS: CLEAN AIR POSITION STRATEGY")
    print("=" * 80)

    first_stint_laps = clean_air_strat.stints[0][1]
    time_penalty = clean_air_strat.ert - best_ert

    clean_air_output = [
        ["Strategy", clean_air_strat.format_strategy_string()],
        ["Stint Split", clean_air_strat.format_split_string()],
        ["Pit Stops", clean_air_strat.pit_stops],
        ["Pit Laps", clean_air_strat.format_pit_laps()],
        ["Optimized ERT", format_time(clean_air_strat.ert)],
        ["vs Rank 1", f"+{time_penalty:.3f}s"]
    ]

    print(tabulate(clean_air_output, tablefmt="plain"))

    # Per-stint mode breakdown
    print("\n**Stint Modes:**")
    for i, (stint, mode) in enumerate(zip(clean_air_strat.stints, clean_air_strat.optimal_modes)):
        compound, laps = stint
        print(f"  Stint {i+1}: {compound} ({laps} laps) - {mode.upper()}")

    # Justification
    print(f"\n**Undercut Target:** Lap {earliest_first_pit} (earliest pit among top strategies)")
    print(f"**First Pit:** Lap {first_stint_laps} (undercut by {earliest_first_pit - first_stint_laps} lap(s))")

    if clean_air_strat.pit_stops == 1:
        print("\n**Recommendation:** 1-STOP avoids traffic from 2-stoppers pitting mid-race")
    else:
        second_pit = first_stint_laps + clean_air_strat.stints[1][1]
        nearby_pits = [lap for lap in top_pit_laps if abs(lap - second_pit) <= 3]
        print(f"\n**Second Pit:** Lap {second_pit}", end="")
        if not nearby_pits:
            print(" (CLEAR WINDOW - no top strategy pits within +/-3 laps)")
        else:
            print(f" (nearby pits: {sorted(nearby_pits)})")


def print_sc_analysis(analysis: SCAnalysis, config: RaceConfig) -> None:
    """Print safety car analysis results."""
    print("\n" + "=" * 80)
    print("SAFETY CAR ANALYSIS")
    print("=" * 80)
    
    # Current state
    print(f"\nCurrent: {analysis.current_compound} tires, {analysis.stint_laps} laps completed, "
          f"{analysis.remaining_laps} laps remaining")
    print(f"Tire Status: ~{analysis.tire_wear_percent:.0f}% worn, "
          f"{analysis.remaining_competitive_laps} competitive laps remaining")
    
    # Key insight: can we finish without pitting?
    if analysis.can_finish_no_pit:
        print(f">>> Can finish race WITHOUT pitting ({analysis.remaining_competitive_laps} laps available)")
    else:
        print(f">>> MUST PIT - only {analysis.remaining_competitive_laps} laps left on tires, need {analysis.remaining_laps}")
    
    print(f"\nSC Pit Value: {analysis.sc_pit_value:.0f}s saved vs green flag pit "
          f"({config.pit_loss_seconds:.0f}s - {config.sc_pit_loss_seconds:.0f}s)")
    
    # Position loss penalty
    if analysis.positions_lost > 0:
        print(f"Position Penalty: {analysis.positions_lost} positions × {config.position_loss_value:.1f}s = "
              f"{analysis.position_penalty:.1f}s added to PIT cost")
    
    # Option A: Stay Out
    print("\n" + "-" * 40)
    print("OPTION A: STAY OUT")
    print("-" * 40)
    if analysis.stay_out_strategy:
        strat = analysis.stay_out_strategy
        print(f"Strategy: {strat.format_strategy_string()}")
        print(f"Stint Split: {strat.format_split_string()}")
        if strat.pit_stops > 0:
            print(f"Pit Laps: {strat.format_pit_laps()} (green flag, {config.pit_loss_seconds:.0f}s each)")
        else:
            print("Pit Laps: None (finish on current tires)")
        print(f"ERT to finish: {format_time(analysis.stay_out_ert)}")
    else:
        print("No valid strategy - cannot finish on current tires")
    
    # Option B: Pit Now
    print("\n" + "-" * 40)
    print("OPTION B: PIT NOW (SC)")
    print("-" * 40)
    if analysis.pit_now_strategy:
        strat = analysis.pit_now_strategy
        print(f"Strategy: {strat.format_strategy_string()}")
        print(f"Stint Split: {strat.format_split_string()}")
        print(f"SC Pit: Now ({config.sc_pit_loss_seconds:.0f}s)")
        if strat.pit_stops > 0:
            print(f"Additional Pits: {strat.format_pit_laps()} (green flag)")
        print(f"ERT to finish: {format_time(analysis.pit_now_ert)}")
    else:
        print("No valid strategy available")
    
    # Recommendation with breakdown
    print("\n" + "=" * 40)
    delta = abs(analysis.time_delta)
    
    if analysis.recommendation == "PIT":
        print(f">>> RECOMMENDATION: PIT NOW (saves {delta:.1f}s)")
        
        # Explain where the time difference comes from
        if analysis.stay_out_strategy and analysis.pit_now_strategy:
            stay_comp = analysis.stay_out_strategy.compound_sequence
            pit_comp = analysis.pit_now_strategy.compound_sequence
            stay_pits = analysis.stay_out_strategy.pit_stops
            
            # Check if strategies use different compound TYPES (not just sequence length)
            stay_compounds = set(stay_comp)
            pit_compounds = set(pit_comp)
            compounds_differ = stay_compounds != pit_compounds
            
            if analysis.can_finish_no_pit and stay_pits == 0:
                print(f"    Note: STAY OUT avoids pitting entirely")
                print(f"    PIT NOW costs {config.sc_pit_loss_seconds:.0f}s but provides fresh tires")
            elif compounds_differ:
                # Different compounds - breakdown the sources
                print(f"    - SC pit timing value: ~{analysis.sc_pit_value:.0f}s (vs green flag pit)")
                if analysis.positions_lost > 0:
                    print(f"    - Position penalty: -{analysis.position_penalty:.1f}s ({analysis.positions_lost} positions lost)")
                compound_benefit = delta - analysis.sc_pit_value + analysis.position_penalty
                if compound_benefit > 0:
                    print(f"    - Compound switch benefit: ~{compound_benefit:.0f}s")
            else:
                # Same compounds - pure pit timing difference
                if analysis.positions_lost > 0:
                    print(f"    Position penalty factored in: -{analysis.position_penalty:.1f}s")
                print(f"    Reason: SC pit timing advantage + fresh tires")
    else:
        # STAY_OUT recommendation
        print(f">>> RECOMMENDATION: STAY OUT (saves {delta:.1f}s)")
        if analysis.positions_lost > 0:
            print(f"    Position penalty factored in: {analysis.position_penalty:.1f}s ({analysis.positions_lost} positions)")
        if analysis.can_finish_no_pit:
            print(f"    Reason: Can finish without pitting")
        else:
            print(f"    Reason: Better to pit on green flag later")
    print("=" * 40)
    
    # War Gaming
    print("\n" + "-" * 40)
    print("WAR GAMING: If you STAY OUT while rivals PIT")
    print("-" * 40)
    print(f"Pace deficit vs fresh tires: ~{analysis.pace_deficit_per_lap:.2f}s/lap")
    print(f"Total time loss over {analysis.remaining_laps} laps: ~{analysis.total_time_loss:.1f}s")
    print(f"Position Risk: {analysis.risk_assessment}")
    
    if analysis.risk_assessment == "LOW":
        print("\nAnalysis: Tire delta is minimal. Track position likely safe.")
    elif analysis.risk_assessment == "MODERATE":
        print("\nAnalysis: Noticeable pace deficit. May lose 1-2 positions in closing laps.")
    else:
        print("\nAnalysis: Significant pace deficit. High risk of being overtaken.")


def run_race_command(config: RaceConfig) -> None:
    """Run the race strategy generation command."""
    print("--- Race Strategy Optimizer Initializing ---")
    
    # Display config info
    if config.min_stint_laps == 'auto':
        min_stint = calculate_min_stint(config)
        print(f"Race Laps: {config.race_laps}, Pit Loss: {config.pit_loss_seconds}s, Min Stint: {min_stint} (auto)")
    else:
        print(f"Race Laps: {config.race_laps}, Pit Loss: {config.pit_loss_seconds}s, Min Stint: {config.min_stint_laps}")
    
    # Generate strategies
    strategies = generate_strategies(config)

    # Filter out strategies with infinite ERT
    strategies = [s for s in strategies if s.ert != float('inf')]
    print(f"Successfully generated {len(strategies)} unique, inventory-valid strategies.")

    if not strategies:
        print("Error: No valid strategies could be generated with the given constraints.")
        return

    # Sort by ERT
    ranked_strategies = sorted(strategies, key=lambda x: x.ert)

    # Deduplicate by Strategy (Mode) string - keep best stint split for each
    seen_keys: set[str] = set()
    unique_strategies: list[Strategy] = []
    for strat in ranked_strategies:
        key = strat.format_strategy_string()  # e.g. "Medium(P) -> Hard(N)"
        if key not in seen_keys:
            seen_keys.add(key)
            unique_strategies.append(strat)
    
    # Find clean air strategy
    clean_air_strat = find_clean_air_strategy(
        config,
        unique_strategies[:config.top_strategies],
        ranked_strategies
    )

    # Output top N unique strategies by ERT
    print_top_strategies(unique_strategies, config)

    if clean_air_strat:
        print_clean_air_strategy(
            clean_air_strat,
            unique_strategies,
            ranked_strategies[0].ert,
            config
        )
    else:
        print("\n[INFO] Could not find a suitable Clean Air Strategy.")


def run_sc_interactive(config: RaceConfig) -> None:
    """Run safety car analysis in interactive mode."""
    print("\n" + "=" * 50)
    print("SAFETY CAR DECISION - Interactive Mode")
    print("=" * 50)
    print(f"Race: {config.race_laps} laps")
    
    compounds = list(config.compounds.keys())
    
    try:
        # Current lap
        current_lap = int(input("\nCurrent lap number: "))
        remaining = config.race_laps - current_lap
        if remaining <= 0:
            print(f"Error: current lap must be < {config.race_laps}")
            return
        print(f"  -> {remaining} laps remaining")
        
        # Stint laps
        stint_laps = int(input("Laps on current tires: "))
        
        # Compound selection
        print(f"\nCompounds: {', '.join(f'{i+1}={c}' for i, c in enumerate(compounds))}")
        comp_idx = int(input("Current compound (enter number): ")) - 1
        if comp_idx < 0 or comp_idx >= len(compounds):
            print("Invalid compound selection")
            return
        compound = compounds[comp_idx]
        
        # Position loss
        pos_loss = int(input("Positions lost if you pit (0 if unknown): "))
        
    except ValueError:
        print("Invalid input - please enter numbers only")
        return
    except KeyboardInterrupt:
        print("\nCancelled")
        return
    
    # Run analysis
    run_sc_command(config, stint_laps, remaining, compound, pos_loss)


def run_sc_command(config: RaceConfig, stint_laps: int, remaining: int, compound: str, positions_at_risk: int) -> None:
    """Run the safety car analysis command."""
    print("\n--- Safety Car Decision Analysis ---")
    
    # Validate compound
    if compound not in config.compounds:
        print(f"Error: Unknown compound '{compound}'. Available: {list(config.compounds.keys())}")
        return
    
    # Run analysis
    analysis = analyze_safety_car(
        stint_laps=stint_laps,
        remaining_laps=remaining,
        current_compound=compound,
        config=config,
        positions_at_risk=positions_at_risk
    )
    
    # Print results
    print_sc_analysis(analysis, config)


# --- Tire Domain Analysis ---

def format_laptime(seconds: float) -> str:
    """Format lap time as M:SS.ms"""
    minutes = floor(seconds / 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


def print_tire_domains(analysis: TireDomainAnalysis, config: RaceConfig, show_chart: bool = False) -> None:
    """Print tire domain analysis results."""
    print("\n" + "=" * 60)
    print("TIRE DOMAIN ANALYSIS (Fresh Stint)")
    print("=" * 60)
    print("Based on progressive degradation model\n")
    
    # Domain table
    print("COMPOUND DOMAINS:")
    print("-" * 60)
    headers = ["Lap Range", "Fastest", "Lap Time Range"]
    rows = []
    for domain in analysis.domains:
        lap_range = f"Laps {domain.start_lap:3d}-{domain.end_lap:3d}"
        time_range = f"{format_laptime(domain.start_laptime_s)} -> {format_laptime(domain.end_laptime_s)}"
        rows.append([lap_range, domain.compound, time_range])
    print(tabulate(rows, headers=headers, tablefmt="plain"))
    
    # Crossover points
    if analysis.crossover_points:
        print("\nCROSSOVER POINTS:")
        print("-" * 60)
        for cp in analysis.crossover_points:
            print(f"Lap {cp.lap}: {cp.from_compound} ({format_laptime(cp.from_laptime_s)}) "
                  f"slower than {cp.to_compound} ({format_laptime(cp.to_laptime_s)})")
            print(f"        -> Switch domain: {cp.from_compound} -> {cp.to_compound}")
    
    # Compound details
    print("\nCOMPOUND DETAILS:")
    print("-" * 60)
    for name, details in analysis.compound_details.items():
        print(f"{name}:")
        print(f"  Base pace: {format_laptime(details['base_pace_s'])} | "
              f"Deg: {details['degradation_s_per_lap']:.2f}s/lap | "
              f"Cliff: {details['max_competitive_laps']} laps")
    
    # ASCII chart
    if show_chart:
        print("\nDOMAIN CHART:")
        print("-" * 60)
        print_domain_chart(analysis)


def print_domain_chart(analysis: TireDomainAnalysis) -> None:
    """Print ASCII domain chart."""
    max_lap = analysis.max_analysis_lap
    chart_width = 50
    scale = chart_width / max_lap
    
    # Header with lap markers
    header = "Lap: "
    for lap in range(0, max_lap + 1, 10):
        pos = int(lap * scale)
        header = header[:5 + pos] + str(lap).ljust(10)[:10 - pos % 10 if pos % 10 else 10]
    print(header[:5 + chart_width])
    
    # Domain bars
    for domain in analysis.domains:
        start_pos = int(domain.start_lap * scale)
        end_pos = int(domain.end_lap * scale)
        width = max(1, end_pos - start_pos)
        
        line = "      " + " " * start_pos + "[" + "=" * (width - 2) + "]"
        compound_label = f" {domain.compound}"
        print(line + compound_label)


def run_tires_command(config: RaceConfig, mode: str, starting_wear: int, show_chart: bool) -> None:
    """Run tire domain analysis."""
    print("\n--- Tire Domain Analysis ---")
    analysis = analyze_tire_domains(config, mode, starting_wear)
    print_tire_domains(analysis, config, show_chart)


# --- Undercut Analysis ---

def print_undercut_analysis(analysis: UndercutAnalysis, config: RaceConfig) -> None:
    """Print undercut/overcut analysis results."""
    position = "AHEAD" if analysis.gap_to_rival > 0 else "BEHIND"
    
    print("\n" + "=" * 60)
    print(f"UNDERCUT ANALYSIS: You are {abs(analysis.gap_to_rival):.1f}s {position} rival")
    print("=" * 60)
    print(f"Rival expected pit: Lap {analysis.rival_pit_lap} (in {analysis.laps_until_rival_pits} laps)")
    
    # Tire state
    print(f"\nYOUR TIRES: {analysis.your_compound}, {analysis.your_tire_laps} laps (~{analysis.your_wear_percent:.0f}% worn)")
    print(f"RIVAL TIRES: {analysis.rival_compound}, {analysis.rival_tire_laps} laps (~{analysis.rival_wear_percent:.0f}% worn)")
    print(f"PIT TO: {analysis.pit_to_compound}")
    
    # Undercut option
    print("\n" + "-" * 40)
    print(f"OPTION: UNDERCUT (pit now, Lap {analysis.current_lap})")
    print("-" * 40)
    print(f"  Fresh tire pace advantage: ~{analysis.fresh_tire_advantage:.2f}s/lap")
    print(f"  Laps in clean air before rival pits: {analysis.undercut_window_laps}")
    print(f"  Time gained in window: {analysis.time_gained_undercut:.1f}s")
    print(f"  Out-lap penalty: -{config.outlap_penalty:.1f}s")
    gap_str = f"+{analysis.projected_gap_after_undercut:.1f}s AHEAD" if analysis.projected_gap_after_undercut > 0 else f"{analysis.projected_gap_after_undercut:.1f}s BEHIND"
    print(f"  Projected gap after rival pits: {gap_str}")
    print(f"  >>> {'UNDERCUT VIABLE' if analysis.undercut_viable else 'UNDERCUT NOT VIABLE'}")
    
    # Overcut/stay out option
    print("\n" + "-" * 40)
    print("OPTION: STAY OUT (pit after rival)")
    print("-" * 40)
    print(f"  Rival exits on fresh tires")
    print(f"  Time lost over {analysis.laps_until_rival_pits} laps: {analysis.time_lost_staying_out:.1f}s")
    gap_str = f"+{analysis.projected_gap_after_overcut:.1f}s AHEAD" if analysis.projected_gap_after_overcut > 0 else f"{analysis.projected_gap_after_overcut:.1f}s BEHIND"
    print(f"  Projected gap: {gap_str}")
    print(f"  >>> {'OVERCUT VIABLE' if analysis.overcut_viable else 'OVERCUT NOT VIABLE'}")
    
    # Recommendation
    print("\n" + "=" * 40)
    print(f">>> RECOMMENDATION: {analysis.recommendation}")
    print(f"    {analysis.recommendation_reason}")
    print("=" * 40)


def run_undercut_interactive(config: RaceConfig) -> None:
    """Run undercut analysis in interactive mode."""
    print("\n" + "=" * 50)
    print("UNDERCUT ANALYSIS - Interactive Mode")
    print("=" * 50)
    
    compounds = list(config.compounds.keys())
    
    try:
        gap = float(input("\nGap to rival (positive=ahead, negative=behind): "))
        current_lap = int(input("Current race lap: "))
        rival_pit = int(input("Expected lap rival will pit: "))
        your_tire_laps = int(input("Laps on your current tires: "))
        rival_tire_laps = int(input("Laps on rival's tires: "))
        
        print(f"\nCompounds: {', '.join(f'{i+1}={c}' for i, c in enumerate(compounds))}")
        your_idx = int(input("Your compound (enter number): ")) - 1
        rival_idx = int(input("Rival compound (enter number): ")) - 1
        
        pit_to_input = input("Pit to compound (enter number, or press Enter for fastest): ")
        pit_to = compounds[int(pit_to_input) - 1] if pit_to_input.strip() else None
        
    except (ValueError, IndexError):
        print("Invalid input")
        return
    except KeyboardInterrupt:
        print("\nCancelled")
        return

    run_undercut_command(config, gap, current_lap, rival_pit, your_tire_laps,
                        rival_tire_laps, compounds[your_idx], compounds[rival_idx], pit_to)


def run_undercut_command(config: RaceConfig, gap: float, current_lap: int, rival_pit: int,
                        your_tire_laps: int, rival_tire_laps: int,
                        your_compound: str, rival_compound: str, pit_to: str | None) -> None:
    """Run undercut analysis command."""
    analysis = analyze_undercut(
        gap_to_rival=gap,
        current_lap=current_lap,
        rival_pit_lap=rival_pit,
        your_compound=your_compound,
        your_tire_laps=your_tire_laps,
        rival_compound=rival_compound,
        rival_tire_laps=rival_tire_laps,
        pit_to_compound=pit_to,
        config=config
    )
    print_undercut_analysis(analysis, config)


# --- DRS Defense Analysis ---

def print_drs_analysis(analysis: DRSAnalysis, config: RaceConfig) -> None:
    """Print DRS defense analysis results."""
    drs_status = "IN DRS RANGE" if analysis.in_drs_range else "OUTSIDE DRS"
    
    print("\n" + "=" * 60)
    print("DRS DEFENSE ANALYSIS")
    print("=" * 60)
    print(f"Current gap: {analysis.gap_to_attacker:.1f}s ({drs_status})")
    print(f"Stint remaining: {analysis.stint_laps_remaining} laps")
    
    # Tire states
    print(f"\nYOUR TIRES: {analysis.your_compound}, {analysis.your_tire_laps} laps "
          f"(~{analysis.your_wear_percent:.0f}% worn)")
    print(f"  - Max competitive laps: {analysis.your_max_competitive_laps}")
    print(f"  - Base pace deficit vs attacker: ~{analysis.base_pace_delta:.2f}s/lap")
    
    print(f"\nATTACKER: {analysis.attacker_compound}, {analysis.attacker_tire_laps} laps "
          f"(~{analysis.attacker_wear_percent:.0f}% worn)")
    
    # Scenarios
    for scenario in analysis.scenarios:
        print("\n" + "-" * 40)
        print(f"SCENARIO: {scenario.name}")
        print("-" * 40)
        print(f"  Mode: {scenario.description}")
        
        gap_str = f"+{scenario.final_gap:.1f}s" if scenario.final_gap > 0 else f"{scenario.final_gap:.1f}s"
        print(f"  Final gap: {gap_str}")
        print(f"  Tire wear at end: {scenario.tire_wear_at_end:.0f} effective laps ({scenario.tire_percent_at_end:.0f}%)")
        
        if scenario.exceeds_tire_life:
            print(f"  WARNING: Exceeds tire life (cliff at lap ~{scenario.cliff_lap})")
        
        if scenario.sustainable:
            print(f"  >>> SUSTAINABLE - position defended")
        else:
            if scenario.final_gap <= 0:
                print(f"  >>> OVERTAKEN")
            else:
                print(f"  >>> NOT SUSTAINABLE - tires will fail")
    
    # Recommendation
    print("\n" + "=" * 40)
    rec = analysis.recommended_scenario
    if "BURST" in analysis.recommendation:
        push_laps = analysis.optimal_push_laps
        print(f">>> RECOMMENDATION: BURST PUSH ({push_laps} laps) then CONSERVE")
    else:
        print(f">>> RECOMMENDATION: {analysis.recommendation}")
    print(f"    Final gap: {rec.final_gap:.1f}s | Tire: {rec.tire_percent_at_end:.0f}% worn")
    print("=" * 40)


def run_drs_interactive(config: RaceConfig) -> None:
    """Run DRS defense analysis in interactive mode."""
    print("\n" + "=" * 50)
    print("DRS DEFENSE ANALYSIS - Interactive Mode")
    print("=" * 50)
    
    compounds = list(config.compounds.keys())
    
    try:
        gap = float(input("\nGap to car behind (seconds): "))
        stint_laps = int(input("Laps remaining in stint: "))
        your_tire_laps = int(input("Laps on your tires: "))
        attacker_tire_laps = int(input("Laps on attacker's tires: "))
        
        print(f"\nCompounds: {', '.join(f'{i+1}={c}' for i, c in enumerate(compounds))}")
        your_idx = int(input("Your compound (enter number): ")) - 1
        attacker_idx = int(input("Attacker compound (enter number): ")) - 1
        
    except (ValueError, IndexError):
        print("Invalid input")
        return
    except KeyboardInterrupt:
        print("\nCancelled")
        return
    
    run_drs_command(config, gap, stint_laps, your_tire_laps, attacker_tire_laps,
                   compounds[your_idx], compounds[attacker_idx])


def run_drs_command(config: RaceConfig, gap: float, stint_laps_remaining: int,
                   your_tire_laps: int, attacker_tire_laps: int,
                   your_compound: str, attacker_compound: str) -> None:
    """Run DRS defense analysis command."""
    analysis = analyze_drs_defense(
        gap_to_attacker=gap,
        stint_laps_remaining=stint_laps_remaining,
        your_compound=your_compound,
        your_tire_laps=your_tire_laps,
        attacker_compound=attacker_compound,
        attacker_tire_laps=attacker_tire_laps,
        config=config
    )
    print_drs_analysis(analysis, config)


# --- Attack Analysis ---

def print_attack_analysis(analysis: AttackAnalysis, config: RaceConfig) -> None:
    """Print attack/catch analysis results."""
    print("\n" + "=" * 60)
    print("ATTACK ANALYSIS: Catching car ahead")
    print("=" * 60)
    print(f"Current gap: {analysis.gap_to_target:.1f}s | Target: DRS range (<{analysis.drs_threshold}s)")
    print(f"Stint remaining: {analysis.stint_laps_remaining} laps")
    
    # Tire states
    print(f"\nYOUR TIRES: {analysis.your_compound}, {analysis.your_tire_laps} laps "
          f"(~{analysis.your_wear_percent:.0f}% worn)")
    print(f"TARGET TIRES: {analysis.target_compound}, {analysis.target_tire_laps} laps "
          f"(~{analysis.target_wear_percent:.0f}% worn)")
    
    # Natural convergence
    print("\n" + "-" * 40)
    print("NATURAL CONVERGENCE:")
    print("-" * 40)
    closing_str = f"~{analysis.natural_closing_rate:.2f}s/lap" if analysis.natural_closing_rate > 0 else "NOT CLOSING"
    print(f"  Natural closing rate: {closing_str}")
    if analysis.can_reach_drs_naturally:
        print(f"  Laps to DRS range: {analysis.laps_to_drs_natural} laps")
        print(f"  >>> THEY COME BACK TO YOU - no attack needed")
    else:
        if analysis.laps_to_drs_natural:
            print(f"  Laps to DRS: {analysis.laps_to_drs_natural} (exceeds stint)")
        print(f"  >>> THEY WON'T COME BACK in this stint")
    
    # Mode scenarios
    for scenario in analysis.scenarios:
        if scenario.name == "STAY_ON_PLAN":
            continue  # Already covered above
        
        print("\n" + "-" * 40)
        print(f"{scenario.name} MODE:")
        print("-" * 40)
        
        if scenario.final_gap <= analysis.drs_threshold:
            print(f"  Reaches DRS: YES")
        else:
            print(f"  Reaches DRS: NO (final gap {scenario.final_gap:.1f}s)")
        
        print(f"  Tire wear at end: {scenario.tire_wear_at_end:.0f} effective laps ({scenario.tire_percent_at_end:.0f}%)")
        
        if scenario.exceeds_tire_life:
            print(f"  WARNING: Exceeds tire competitive life!")
        
        if scenario.sustainable:
            print(f"  >>> REACHES DRS with sustainable tires")
        elif scenario.final_gap <= analysis.drs_threshold:
            print(f"  >>> REACHES DRS but burns tires")
        else:
            print(f"  >>> DOESN'T REACH DRS")
    
    # Recommendation
    print("\n" + "=" * 40)
    print(f">>> RECOMMENDATION: {analysis.recommendation}")
    rec = analysis.recommended_scenario
    if analysis.recommendation == "STAY_ON_PLAN":
        if analysis.can_reach_drs_naturally:
            print(f"    Natural convergence gets you to DRS in {analysis.laps_to_drs_natural} laps")
        else:
            print(f"    No mode reaches DRS sustainably")
    else:
        print(f"    Final gap: {rec.final_gap:.1f}s | Tire: {rec.tire_percent_at_end:.0f}% worn")
    
    if analysis.tire_warning:
        print(f"\n    WARNING: {analysis.tire_warning}")
    print("=" * 40)


def run_attack_interactive(config: RaceConfig) -> None:
    """Run attack analysis in interactive mode."""
    print("\n" + "=" * 50)
    print("ATTACK ANALYSIS - Interactive Mode")
    print("=" * 50)
    
    compounds = list(config.compounds.keys())
    
    try:
        gap = float(input("\nGap to car ahead (seconds): "))
        stint_laps = int(input("Laps remaining in stint: "))
        your_tire_laps = int(input("Laps on your tires: "))
        target_tire_laps = int(input("Laps on target's tires: "))
        
        print(f"\nCompounds: {', '.join(f'{i+1}={c}' for i, c in enumerate(compounds))}")
        your_idx = int(input("Your compound (enter number): ")) - 1
        target_idx = int(input("Target compound (enter number): ")) - 1
        
    except (ValueError, IndexError):
        print("Invalid input")
        return
    except KeyboardInterrupt:
        print("\nCancelled")
        return
    
    run_attack_command(config, gap, stint_laps, your_tire_laps, target_tire_laps,
                      compounds[your_idx], compounds[target_idx])


def run_attack_command(config: RaceConfig, gap: float, stint_laps_remaining: int,
                      your_tire_laps: int, target_tire_laps: int,
                      your_compound: str, target_compound: str) -> None:
    """Run attack analysis command."""
    analysis = analyze_attack(
        gap_to_target=gap,
        stint_laps_remaining=stint_laps_remaining,
        your_compound=your_compound,
        your_tire_laps=your_tire_laps,
        target_compound=target_compound,
        target_tire_laps=target_tire_laps,
        config=config
    )
    print_attack_analysis(analysis, config)


# --- Live Mid-Race Strategy Functions ---

def print_live_analysis(analysis: LiveAnalysis, config: RaceConfig) -> None:
    """Print live mid-race strategy analysis."""
    print("\n" + "=" * 60)
    print("LIVE STRATEGY ANALYSIS")
    print("=" * 60)
    
    # Current state
    print(f"\nLap {analysis.current_lap} of {config.race_laps} | "
          f"{analysis.remaining_laps} laps remaining")
    print(f"Tires: {analysis.current_compound}, {analysis.tire_laps} laps worn "
          f"(~{analysis.tire_wear_percent:.0f}%)")
    print(f"Tire cliff in: {analysis.remaining_competitive_laps} laps "
          f"(lap {analysis.current_lap + analysis.remaining_competitive_laps})")
    
    if analysis.can_finish_no_pit:
        print("\n>>> Can finish on current tires")
    
    # Strategy table
    print("\n" + "-" * 60)
    print("OPTIMAL STRATEGIES FROM HERE")
    print("-" * 60)
    
    table_data = []
    for i, ls in enumerate(analysis.strategies[:config.top_strategies]):
        strat = ls.strategy
        
        # Format pit info - convert to actual lap number
        if ls.next_pit_lap is None:
            pit_info = "No pit"
            next_tire = "-"
        else:
            actual_pit_lap = analysis.current_lap + ls.next_pit_lap
            pit_info = f"Lap {actual_pit_lap}"
            next_tire = ls.next_compound or "-"
        
        table_data.append([
            i + 1,
            strat.format_strategy_string(),
            pit_info,
            next_tire,
            format_time(ls.ert_to_finish)
        ])
    
    headers = ["#", "Strategy", "Pit Lap", "Next Tire", "ERT"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    
    # Recommendation
    rec = analysis.recommended
    print("\n" + "=" * 60)
    print(f">>> RECOMMENDED: {rec.strategy.format_strategy_string()}")
    if rec.next_pit_lap:
        actual_pit_lap = analysis.current_lap + rec.next_pit_lap
        print(f"    Pit on lap {actual_pit_lap} for {rec.next_compound}")
    else:
        print("    Stay out to finish")
    print(f"    ERT: {format_time(rec.ert_to_finish)}")
    print("=" * 60)


def run_live_interactive(config: RaceConfig) -> None:
    """Run live strategy analysis in interactive mode."""
    print("\n" + "=" * 50)
    print("LIVE STRATEGY - Interactive Mode")
    print("=" * 50)
    print(f"Race: {config.race_laps} laps")
    
    compounds = list(config.compounds.keys())
    
    try:
        # Current lap
        current_lap = int(input("\nCurrent lap number: "))
        remaining = config.race_laps - current_lap
        if remaining <= 0:
            print(f"Error: current lap must be < {config.race_laps}")
            return
        print(f"  -> {remaining} laps remaining")
        
        # Compound
        print(f"\nCompounds: {', '.join(f'{i+1}={c}' for i, c in enumerate(compounds))}")
        compound_idx = int(input("Current tire compound (enter number): ")) - 1
        compound = compounds[compound_idx]
        
        # Tire wear
        tire_laps = int(input("Laps on current tires: "))
        
    except (ValueError, IndexError):
        print("Invalid input")
        return
    except KeyboardInterrupt:
        print("\nCancelled")
        return
    
    run_live_command(config, current_lap, compound, tire_laps, remaining)


def run_live_command(config: RaceConfig, current_lap: int, compound: str, tire_laps: int, remaining: int) -> None:
    """Run live mid-race strategy analysis."""
    try:
        analysis = analyze_live_strategy(
            current_lap=current_lap,
            current_compound=compound,
            tire_laps=tire_laps,
            remaining_laps=remaining,
            config=config
        )
        print_live_analysis(analysis, config)
    except ValueError as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="F1 Race Strategy Optimization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add --config at top level for backward compatibility (defaults to config.yaml)
    parser.add_argument('--config', type=str, default='config.yaml',
                       help="Path to the YAML configuration file (default: config.yaml)")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Race command (full strategy generation)
    race_parser = subparsers.add_parser('race', help='Generate optimal race strategies')
    race_parser.add_argument('--config', type=str, default='config.yaml',
                            help="Path to the YAML configuration file (default: config.yaml)")
    
    # SC command (safety car analysis)
    sc_parser = subparsers.add_parser('sc', help='Safety car pit decision analysis (interactive if no args)')
    sc_parser.add_argument('--config', type=str, default='config.yaml',
                          help="Path to the YAML configuration file (default: config.yaml)")
    sc_parser.add_argument('--current-lap', type=int,
                          help="Current lap number (remaining laps calculated from race_laps in config)")
    sc_parser.add_argument('--stint-laps', type=int,
                          help="Laps on current tires")
    sc_parser.add_argument('--compound', type=str,
                          choices=['Soft', 'Medium', 'Hard'],
                          help="Current tire compound")
    sc_parser.add_argument('--pos-loss', type=int, default=0,
                          help="Estimated positions lost if you pit (cars that will stay out)")
    
    # Tires command (tire domain analysis)
    tires_parser = subparsers.add_parser('tires', help='Analyze tire compound domains and crossover points')
    tires_parser.add_argument('--config', type=str, default='config.yaml',
                             help="Path to the YAML configuration file (default: config.yaml)")
    tires_parser.add_argument('--mode', type=str, choices=['normal', 'push', 'conserve'],
                             default='normal', help="Pace mode for analysis (default: normal)")
    tires_parser.add_argument('--starting-wear', type=int, default=0,
                             help="Starting tire wear in laps (default: 0 = fresh)")
    tires_parser.add_argument('--chart', action='store_true',
                             help="Display ASCII domain chart")
    
    # Undercut command
    undercut_parser = subparsers.add_parser('undercut', help='Undercut/overcut analysis vs rival')
    undercut_parser.add_argument('--config', type=str, default='config.yaml',
                                help="Path to the YAML configuration file (default: config.yaml)")
    undercut_parser.add_argument('--gap', type=float,
                                help="Gap to rival in seconds (positive = ahead, negative = behind)")
    undercut_parser.add_argument('--current-lap', type=int,
                                help="Current race lap")
    undercut_parser.add_argument('--rival-pit', type=int,
                                help="Expected lap rival will pit")
    undercut_parser.add_argument('--your-tire-laps', type=int,
                                help="Laps on your current tires")
    undercut_parser.add_argument('--rival-tire-laps', type=int,
                                help="Laps on rival's current tires")
    undercut_parser.add_argument('--your-compound', type=str, choices=['Soft', 'Medium', 'Hard'],
                                help="Your current tire compound")
    undercut_parser.add_argument('--rival-compound', type=str, choices=['Soft', 'Medium', 'Hard'],
                                help="Rival's current tire compound")
    undercut_parser.add_argument('--pit-to', type=str, choices=['Soft', 'Medium', 'Hard'],
                                help="Compound to pit onto (default: fastest)")
    
    # DRS command
    drs_parser = subparsers.add_parser('drs', help='DRS defense analysis - should you push or conserve?')
    drs_parser.add_argument('--config', type=str, default='config.yaml',
                           help="Path to the YAML configuration file (default: config.yaml)")
    drs_parser.add_argument('--gap', type=float,
                           help="Gap to car behind in seconds")
    drs_parser.add_argument('--stint-laps-remaining', type=int,
                           help="Laps remaining until stint end")
    drs_parser.add_argument('--your-tire-laps', type=int,
                           help="Laps on your current tires")
    drs_parser.add_argument('--attacker-tire-laps', type=int,
                           help="Laps on attacker's tires")
    drs_parser.add_argument('--your-compound', type=str, choices=['Soft', 'Medium', 'Hard'],
                           help="Your tire compound")
    drs_parser.add_argument('--attacker-compound', type=str, choices=['Soft', 'Medium', 'Hard'],
                           help="Attacker's tire compound")
    
    # Attack command
    attack_parser = subparsers.add_parser('attack', help='Attack analysis - should you chase or let them come back?')
    attack_parser.add_argument('--config', type=str, default='config.yaml',
                              help="Path to the YAML configuration file (default: config.yaml)")
    attack_parser.add_argument('--gap', type=float,
                              help="Gap to car ahead in seconds")
    attack_parser.add_argument('--stint-laps-remaining', type=int,
                              help="Laps remaining until stint end")
    attack_parser.add_argument('--your-tire-laps', type=int,
                              help="Laps on your current tires")
    attack_parser.add_argument('--target-tire-laps', type=int,
                              help="Laps on target's tires")
    attack_parser.add_argument('--your-compound', type=str, choices=['Soft', 'Medium', 'Hard'],
                              help="Your tire compound")
    attack_parser.add_argument('--target-compound', type=str, choices=['Soft', 'Medium', 'Hard'],
                              help="Target's tire compound")
    
    # Live mid-race strategy command
    live_parser = subparsers.add_parser('live', help='Mid-race strategy recalculation from current position')
    live_parser.add_argument('--config', type=str, default='config.yaml',
                            help="Path to the YAML configuration file (default: config.yaml)")
    live_parser.add_argument('--current-lap', type=int,
                            help="Current lap number (remaining laps calculated from race_laps in config)")
    live_parser.add_argument('--compound', type=str, choices=['Soft', 'Medium', 'Hard'],
                            help="Current tire compound")
    live_parser.add_argument('--tire-laps', type=int,
                            help="Laps on current tires")
    
    args = parser.parse_args()

    # Default to race command if no subcommand but --config is provided
    if args.command is None:
        if args.config:
            # Backward compatibility: treat as race command
            args.command = 'race'
        else:
            parser.print_help()
            return
    
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Route to appropriate command
    if args.command == 'race':
        run_race_command(config)
    elif args.command == 'sc':
        # Check if we have enough args for non-interactive mode
        has_required = all([
            args.current_lap is not None,
            args.stint_laps is not None,
            args.compound is not None
        ])
        
        if not has_required:
            # Interactive mode
            run_sc_interactive(config)
        else:
            # Calculate remaining laps from config
            remaining = config.race_laps - args.current_lap
            if remaining <= 0:
                print(f"Error: current-lap ({args.current_lap}) must be < race_laps ({config.race_laps})")
                return
            run_sc_command(config, args.stint_laps, remaining, args.compound, args.pos_loss)
    elif args.command == 'tires':
        run_tires_command(config, args.mode, args.starting_wear, args.chart)
    elif args.command == 'undercut':
        has_required = all([
            args.gap is not None,
            args.current_lap is not None,
            args.rival_pit is not None,
            args.your_tire_laps is not None,
            args.rival_tire_laps is not None,
            args.your_compound is not None,
            args.rival_compound is not None
        ])
        if not has_required:
            run_undercut_interactive(config)
        else:
            run_undercut_command(config, args.gap, args.current_lap, args.rival_pit,
                               args.your_tire_laps, args.rival_tire_laps,
                               args.your_compound, args.rival_compound, args.pit_to)
    elif args.command == 'drs':
        has_required = all([
            args.gap is not None,
            args.stint_laps_remaining is not None,
            args.your_tire_laps is not None,
            args.attacker_tire_laps is not None,
            args.your_compound is not None,
            args.attacker_compound is not None
        ])
        if not has_required:
            run_drs_interactive(config)
        else:
            run_drs_command(config, args.gap, args.stint_laps_remaining,
                          args.your_tire_laps, args.attacker_tire_laps,
                          args.your_compound, args.attacker_compound)
    elif args.command == 'attack':
        has_required = all([
            args.gap is not None,
            args.stint_laps_remaining is not None,
            args.your_tire_laps is not None,
            args.target_tire_laps is not None,
            args.your_compound is not None,
            args.target_compound is not None
        ])
        if not has_required:
            run_attack_interactive(config)
        else:
            run_attack_command(config, args.gap, args.stint_laps_remaining,
                             args.your_tire_laps, args.target_tire_laps,
                             args.your_compound, args.target_compound)
    elif args.command == 'live':
        has_required = all([
            args.current_lap is not None,
            args.compound is not None,
            args.tire_laps is not None
        ])
        if not has_required:
            run_live_interactive(config)
        else:
            # Calculate remaining laps from config
            remaining = config.race_laps - args.current_lap
            if remaining <= 0:
                print(f"Error: current-lap ({args.current_lap}) must be < race_laps ({config.race_laps})")
                return
            run_live_command(config, args.current_lap, args.compound, args.tire_laps, remaining)


if __name__ == "__main__":
    main()
