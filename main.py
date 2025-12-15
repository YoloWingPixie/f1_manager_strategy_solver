"""F1 Race Strategy Optimizer - CLI Entry Point."""
import yaml
import argparse
import re
from math import floor

from tabulate import tabulate

from models import RaceConfig, Compound, Inventory, PaceMode, Strategy
from strategy import generate_strategies, find_clean_air_strategy, calculate_min_stint


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
        require_medium_or_hard=raw.get('require_medium_or_hard', True)
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


def main():
    parser = argparse.ArgumentParser(description="Race Strategy Optimization CLI.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

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


if __name__ == "__main__":
    main()
