"""F1 Race Strategy Optimizer - CLI Entry Point."""
import argparse

from models import RaceConfig, Strategy
from config import load_config
from core import calculate_min_stint
from analysis import (
    generate_strategies, find_clean_air_strategy, analyze_safety_car,
    analyze_tire_domains, analyze_undercut, analyze_drs_defense, analyze_attack,
    analyze_live_strategy
)
from output import (
    format_time, print_top_strategies, print_clean_air_strategy,
    print_sc_analysis, print_tire_domains, print_undercut_analysis,
    print_drs_analysis, print_attack_analysis, print_live_analysis
)


def prompt_compound(compounds: list[str], label: str) -> str:
    """Prompt user to select a compound from a list. Raises IndexError if invalid."""
    print(f"\nCompounds: {', '.join(f'{i+1}={c}' for i, c in enumerate(compounds))}")
    idx = int(input(f"{label} (enter number): ")) - 1
    if idx < 0 or idx >= len(compounds):
        raise IndexError("Invalid compound selection")
    return compounds[idx]


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

    # Sort by ERT, with compound order preference as tie-breaker
    # Preference: Soft openers (1-stop) and Soft bookends (2-stop) rank higher
    def sort_key(s: Strategy) -> tuple[float, int]:
        seq = s.compound_sequence
        preference = 0  # Lower = better
        
        # 1-stop: Soft -> M/H preferred over M/H -> Soft
        if len(seq) == 2 and 'Soft' in seq:
            if seq[0] == 'Soft':
                preference = 0  # Preferred
            else:
                preference = 1
        
        # 2-stop with 2 Softs: Prefer bookend (Soft -> X -> Soft)
        if len(seq) == 3 and seq.count('Soft') == 2:
            if seq[0] == 'Soft' and seq[2] == 'Soft':
                preference = 0  # Bookend preferred
            elif seq[0] == 'Soft':
                preference = 1  # At least starts with Soft
            else:
                preference = 2  # Others
        
        return (s.ert, preference)
    
    ranked_strategies = sorted(strategies, key=sort_key)

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
        current_lap = int(input("\nCurrent lap number: "))
        remaining = config.race_laps - current_lap
        if remaining <= 0:
            print(f"Error: current lap must be < {config.race_laps}")
            return
        print(f"  -> {remaining} laps remaining")
        
        stint_laps = int(input("Laps on current tires: "))
        compound = prompt_compound(compounds, "Current compound")
        pos_loss = int(input("Positions lost if you pit (0 if unknown): "))
        
    except (ValueError, IndexError) as e:
        print(f"Invalid input: {e}")
        return
    except KeyboardInterrupt:
        print("\nCancelled")
        return
    
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


def run_tires_command(config: RaceConfig, mode: str, starting_wear: int, show_chart: bool) -> None:
    """Run tire domain analysis."""
    print("\n--- Tire Domain Analysis ---")
    analysis = analyze_tire_domains(config, mode, starting_wear)
    print_tire_domains(analysis, config, show_chart)


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
        
        your_compound = prompt_compound(compounds, "Your compound")
        rival_compound = prompt_compound(compounds, "Rival compound")
        
        pit_to_input = input("Pit to compound (enter number, or press Enter for fastest): ")
        pit_to = compounds[int(pit_to_input) - 1] if pit_to_input.strip() else None
        
    except (ValueError, IndexError) as e:
        print(f"Invalid input: {e}")
        return
    except KeyboardInterrupt:
        print("\nCancelled")
        return

    run_undercut_command(config, gap, current_lap, rival_pit, your_tire_laps,
                        rival_tire_laps, your_compound, rival_compound, pit_to)


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
        
        your_compound = prompt_compound(compounds, "Your compound")
        attacker_compound = prompt_compound(compounds, "Attacker compound")
        
    except (ValueError, IndexError) as e:
        print(f"Invalid input: {e}")
        return
    except KeyboardInterrupt:
        print("\nCancelled")
        return
    
    run_drs_command(config, gap, stint_laps, your_tire_laps, attacker_tire_laps,
                   your_compound, attacker_compound)


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
        
        your_compound = prompt_compound(compounds, "Your compound")
        target_compound = prompt_compound(compounds, "Target compound")
        
    except (ValueError, IndexError) as e:
        print(f"Invalid input: {e}")
        return
    except KeyboardInterrupt:
        print("\nCancelled")
        return
    
    run_attack_command(config, gap, stint_laps, your_tire_laps, target_tire_laps,
                      your_compound, target_compound)


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


def run_live_interactive(config: RaceConfig) -> None:
    """Run live strategy analysis in interactive mode."""
    print("\n" + "=" * 50)
    print("LIVE STRATEGY - Interactive Mode")
    print("=" * 50)
    print(f"Race: {config.race_laps} laps")
    
    compounds = list(config.compounds.keys())
    
    try:
        current_lap = int(input("\nCurrent lap number: "))
        remaining = config.race_laps - current_lap
        if remaining <= 0:
            print(f"Error: current lap must be < {config.race_laps}")
            return
        print(f"  -> {remaining} laps remaining")
        
        compound = prompt_compound(compounds, "Current tire compound")
        tire_laps = int(input("Laps on current tires: "))
        
    except (ValueError, IndexError) as e:
        print(f"Invalid input: {e}")
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

