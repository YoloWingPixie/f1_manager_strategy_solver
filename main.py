"""F1 Race Strategy Optimizer - CLI Entry Point."""
import yaml
import argparse
import re
from math import floor

from tabulate import tabulate

from models import RaceConfig, Compound, Inventory, PaceMode, Strategy, SCAnalysis
from strategy import generate_strategies, find_clean_air_strategy, calculate_min_stint, analyze_safety_car


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
        position_loss_value=raw.get('position_loss_value', 2.5)
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
            print(" (CLEAR WINDOW - no top strategy pits within ±3 laps)")
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
        print(f">>> RECOMMENDATION: PIT NOW")
        
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
                print(f"    Time difference: {delta:.1f}s")
                print(f"    Note: STAY OUT avoids pitting entirely")
                print(f"    PIT NOW costs {config.sc_pit_loss_seconds:.0f}s but provides fresh tires")
            elif compounds_differ:
                # Different compounds - breakdown the sources
                print(f"\n    Time Breakdown:")
                print(f"    - SC pit timing value: ~{analysis.sc_pit_value:.0f}s (vs green flag pit)")
                if analysis.positions_lost > 0:
                    print(f"    - Position penalty: -{analysis.position_penalty:.1f}s ({analysis.positions_lost} positions lost)")
                compound_benefit = delta - analysis.sc_pit_value + analysis.position_penalty
                if compound_benefit > 0:
                    print(f"    - Compound switch benefit: ~{compound_benefit:.0f}s")
                    print(f"      (PIT NOW uses {pit_comp[0]}, STAY OUT uses {stay_comp[0]})")
                print(f"    - Net advantage: {delta:.1f}s")
                print(f"\n    NOTE: Most savings come from switching to faster compound,")
                print(f"    not from pit timing. Check if compound lap times are realistic.")
            else:
                # Same compounds - pure pit timing difference
                print(f"    Time saved: {delta:.1f}s")
                if analysis.positions_lost > 0:
                    print(f"    Position penalty factored in: -{analysis.position_penalty:.1f}s")
                if delta <= analysis.sc_pit_value + 5:
                    print(f"    Reason: SC pit timing advantage ({analysis.sc_pit_value:.0f}s)")
                else:
                    print(f"    Reason: Fresh tires + pit timing")
    else:
        print(f">>> RECOMMENDATION: STAY OUT")
        print(f"    Time saved vs pitting: {delta:.1f}s")
        if analysis.positions_lost > 0:
            print(f"    Position penalty factored in: {analysis.position_penalty:.1f}s ({analysis.positions_lost} positions)")
        if analysis.can_finish_no_pit:
            print(f"    Reason: Can finish without pitting, avoiding {config.sc_pit_loss_seconds:.0f}s pit loss")
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

    # Deduplicate: keep only best lap split per (compound_sequence, pit_stops)
    seen_keys: set[tuple] = set()
    unique_strategies: list[Strategy] = []
    for strat in ranked_strategies:
        key = (tuple(strat.compound_sequence), strat.pit_stops)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_strategies.append(strat)

    # Find clean air strategy
    clean_air_strat = find_clean_air_strategy(
        config,
        unique_strategies[:config.top_strategies],
        ranked_strategies
    )

    # Output
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
    
    compounds = list(config.compounds.keys())
    
    try:
        # Current lap
        current_lap = int(input("\nCurrent lap number: "))
        
        # Last pit lap
        last_pit = int(input("Last pit stop lap (0 if none): "))
        stint_laps = current_lap - last_pit
        print(f"  → Stint laps: {stint_laps}")
        
        # Remaining laps
        remaining = int(input("Laps remaining in race: "))
        
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
    # Option 1: Direct stint laps
    sc_parser.add_argument('--stint-laps', type=int,
                          help="Laps completed on current tires")
    # Option 2: Calculate from last pit and current lap
    sc_parser.add_argument('--last-pit', type=int,
                          help="Lap number of last pit stop (0 if no pit yet)")
    sc_parser.add_argument('--current-lap', type=int,
                          help="Current lap number")
    sc_parser.add_argument('--remaining', type=int,
                          help="Laps remaining in race")
    sc_parser.add_argument('--compound', type=str,
                          choices=['Soft', 'Medium', 'Hard'],
                          help="Current tire compound")
    sc_parser.add_argument('--pos-loss', type=int, default=0,
                          help="Estimated positions lost if you pit (cars that will stay out)")
    
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
        has_stint_info = (args.stint_laps is not None or 
                         (args.last_pit is not None and args.current_lap is not None))
        has_required = has_stint_info and args.remaining is not None and args.compound is not None
        
        if not has_required:
            # Interactive mode
            run_sc_interactive(config)
        else:
            # Calculate stint_laps from either direct input or last-pit + current-lap
            if args.stint_laps is not None:
                stint_laps = args.stint_laps
            else:
                stint_laps = args.current_lap - args.last_pit
                if stint_laps < 0:
                    print(f"Error: current-lap ({args.current_lap}) must be >= last-pit ({args.last_pit})")
                    return
            run_sc_command(config, stint_laps, args.remaining, args.compound, args.pos_loss)


if __name__ == "__main__":
    main()
