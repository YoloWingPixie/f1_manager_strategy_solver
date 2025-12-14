import yaml
import argparse
import re
import sys
from math import floor
from itertools import product
from tabulate import tabulate
from tqdm import tqdm

# Pace mode constants
MODES = ['normal', 'push', 'conserve']

def parse_laptime(value):
    """
    Parse a lap time value that can be either:
    - A string like "1:31.209" or "01:31.209" (M:SS.mmm or MM:SS.mmm)
    - A number (already in seconds)
    Returns time in seconds as a float.
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Match patterns like "1:31.209" or "01:31.209"
        match = re.match(r'^(\d+):(\d+(?:\.\d+)?)$', value.strip())
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            return minutes * 60 + seconds
        
        # Try parsing as plain number string
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot parse lap time: {value}")
    
    raise ValueError(f"Invalid lap time type: {type(value)}")

def get_pace_delta(mode, config):
    """Get pace delta for a given mode from config."""
    if mode == 'normal':
        return 0.0
    return config['pace_modes'][mode]['delta_per_lap']

def get_degradation_factor(mode, config):
    """Get degradation factor for a given mode from config."""
    if mode == 'normal':
        return 1.0
    return config['pace_modes'][mode]['degradation_factor']

def get_tire_max_laps(stints, compound_data, inventory, config):
    """
    Determine max competitive laps for each stint based on tire type (new vs scrubbed).
    New tires are used first, then scrubbed. Scrubbed have reduced life.
    Returns list of max_laps for each stint.
    """
    scrubbed_penalty = config.get('scrubbed_life_penalty', 3)
    
    # Track how many of each compound we've used
    usage = {}
    max_laps_list = []
    
    for compound_name, _ in stints:
        usage[compound_name] = usage.get(compound_name, 0) + 1
        times_used = usage[compound_name]
        
        base_max = compound_data[compound_name]['max_competitive_laps']
        
        # Determine if this is a new or scrubbed tire
        # New tires are used first
        new_key = compound_name.lower() + '_new'
        scrubbed_key = compound_name.lower() + '_scrubbed'
        
        new_count = inventory.get(new_key, 0)
        
        if times_used <= new_count:
            # Using a new tire
            max_laps_list.append(base_max)
        else:
            # Using a scrubbed tire
            max_laps_list.append(base_max - scrubbed_penalty)
    
    return max_laps_list

# --- Core Calculation Function (MANDATORY Formula) ---
def calculate_ert_with_modes(stints, stint_modes, pit_loss_seconds, compound_data, config, inventory=None):
    """
    Calculates the Estimated Race Time (ERT) for a given strategy with per-stint pace modes.
    Formula: Stint Time = (N * Base Lap Time) + (D_adjusted * N * (N-1) / 2) + (N * pace_delta)
    
    stints: list of (compound_name, N) tuples
    stint_modes: list of mode strings ('normal', 'push', 'conserve') per stint
    inventory: if provided, adjusts max laps for scrubbed tires
    
    Push increases degradation, Conserve decreases it.
    """
    total_time = 0.0
    num_stops = len(stints) - 1
    
    # Get max laps per stint (accounting for new vs scrubbed)
    if inventory:
        max_laps_per_stint = get_tire_max_laps(stints, compound_data, inventory, config)
    else:
        max_laps_per_stint = [compound_data[s[0]]['max_competitive_laps'] for s in stints]
    
    for i, ((compound_name, N), mode) in enumerate(zip(stints, stint_modes)):
        if N <= 0:
            return float('inf')
        
        data = compound_data[compound_name]
        base_time = data['avg_lap_time_s']
        D_base = data['degradation_s_per_lap']
        max_laps_base = max_laps_per_stint[i]
        pace_delta = get_pace_delta(mode, config)
        deg_factor = get_degradation_factor(mode, config)
        
        # Degradation factor affects both time AND max stint length
        # Higher deg = faster wear = fewer laps possible
        adjusted_max_laps = floor(max_laps_base / deg_factor)
        if N > adjusted_max_laps:
            return float('inf')  # Stint exceeds tire life for this mode
        
        # Adjust degradation based on mode
        D = D_base * deg_factor
        
        # Calculate time: base + degradation + pace adjustment
        degradation_time = D * N * (N - 1) / 2
        stint_time = (N * base_time) + degradation_time + (N * pace_delta)
        total_time += stint_time

    total_time += num_stops * pit_loss_seconds
    return total_time

def find_optimal_modes(stints, pit_loss_seconds, compound_data, config, inventory=None):
    """
    Find the optimal pace mode for each stint by trying all permutations.
    Returns (best_modes, best_ert) tuple.
    """
    num_stints = len(stints)
    best_modes = tuple(['normal'] * num_stints)
    best_ert = calculate_ert_with_modes(stints, best_modes, pit_loss_seconds, compound_data, config, inventory)
    
    # Try all mode permutations
    for modes in product(MODES, repeat=num_stints):
        ert = calculate_ert_with_modes(stints, modes, pit_loss_seconds, compound_data, config, inventory)
        if ert < best_ert:
            best_ert = ert
            best_modes = modes
    
    return best_modes, best_ert

# --- Strategy Generation ---
def is_valid_stint(compound_name, N, compound_data):
    """Checks if a stint length N is within the maximum competitive limits."""
    max_laps = compound_data[compound_name]['max_competitive_laps']
    return 1 <= N <= max_laps

def check_inventory(strategy, inventory):
    """Checks if a strategy's tire usage is covered by inventory (New and Scrubbed)."""
    usage = {}
    for compound in strategy:
        usage[compound] = usage.get(compound, 0) + 1
    
    # Check total Soft usage
    if usage.get('Soft', 0) > (inventory.get('soft_new', 0) + inventory.get('soft_scrubbed', 0)):
        return False
    
    # Check total Medium usage
    if usage.get('Medium', 0) > (inventory.get('medium_new', 0) + inventory.get('medium_scrubbed', 0)):
        return False
        
    # Check total Hard usage
    if usage.get('Hard', 0) > inventory.get('hard_new', 0):
        return False
        
    # Crucially, check for implicit 'scrubbed' usage: A tire used twice requires two units.
    # The second tire is implicitly the scrubbed one if available.
    if usage.get('Soft', 0) == 2 and inventory.get('soft_scrubbed', 0) < 1:
        return False
    if usage.get('Medium', 0) == 2 and inventory.get('medium_scrubbed', 0) < 1:
        return False
        
    # Hard is assumed to only be used once (inventory hard_new: 1)
    if usage.get('Hard', 0) > 1:
        return False # Should be covered by the total H check, but explicit is better
    
    # MANDATORY: Must use at least one Medium OR Hard tire (F1 regulation)
    if usage.get('Medium', 0) == 0 and usage.get('Hard', 0) == 0:
        return False
        
    return True

def calculate_min_stint(config):
    """
    Calculate minimum viable stint length based on pit loss.
    A stint should be long enough that fresh tires provide meaningful benefit,
    but not so long that it prevents tactical flexibility.
    """
    pit_loss = config['pit_loss_seconds']
    
    # Use pit_loss / 2 as baseline - ensures undercuts remain possible
    # while preventing absurdly short stints
    min_stint = max(floor(pit_loss / 2), 8)
    return min_stint

def generate_strategies(config):
    """Generates all valid 1, 2, and 3-stop strategies."""
    RACE_LAPS = config['race_laps']
    COMPOUND_DATA = config['compounds']
    INVENTORY = config['inventory']
    
    # Calculate or use configured min stint
    min_stint_cfg = config.get('min_stint_laps', 'auto')
    if min_stint_cfg == 'auto':
        MIN_STINT = calculate_min_stint(config)
    else:
        MIN_STINT = int(min_stint_cfg)
    
    # First stint is FREE (no pit stop to pay off) - allow shorter for tactical undercuts
    FIRST_STINT_MIN = 8
    
    LAP_STEP = config.get('stint_lap_step', 1)
    compound_names = list(COMPOUND_DATA.keys())
    
    all_strategies = []

    # Helper function for recursive generation
    def recurse_stints(current_stints, laps_remaining, max_stops):
        current_stops = len(current_stints) - 1
        
        # Base Case 1: Laps complete (End of race)
        if laps_remaining == 0:
            # Check if this combination of tires is allowed by inventory
            strategy_compounds = [s[0] for s in current_stints]
            if check_inventory(strategy_compounds, INVENTORY):
                # Final check on max stops. 3 stops = 4 stints.
                if current_stops <= max_stops:
                    all_strategies.append(current_stints)
            return

        # Base Case 2: Max stops reached but race incomplete = invalid strategy (don't save)
        if current_stops >= max_stops and laps_remaining > 0:
            return

        # Recursive Step: Try all compounds and valid stint lengths (with step)
        if current_stops < max_stops:
            for comp_name in compound_names:
                L_max = COMPOUND_DATA[comp_name]['max_competitive_laps']
                
                # First stint is FREE (no pit stop) - lower minimum for tactical flexibility
                # Subsequent stints must justify their pit stop cost
                is_first_stint = len(current_stints) == 0
                min_this_stint = FIRST_STINT_MIN if is_first_stint else MIN_STINT
                
                # Iterate through stint lengths with step
                max_this_stint = min(laps_remaining, L_max)
                for N in range(min_this_stint, max_this_stint + 1, LAP_STEP):
                    remaining_after = laps_remaining - N
                    # Skip if remaining laps are > 0 but < MIN_STINT (invalid for next stint)
                    if 0 < remaining_after < MIN_STINT:
                        continue
                    
                    new_stints = current_stints + [(comp_name, N)]
                    
                    # Optimization: Check inventory usage *mid-recursion* to prune branches
                    strategy_compounds = [s[0] for s in new_stints]
                    if check_inventory(strategy_compounds, INVENTORY):
                        recurse_stints(new_stints, remaining_after, max_stops)


    # Run generation for strategies up to max_pit_stops
    MAX_STOPS = config.get('max_pit_stops', 3)
    print("Generating strategy combinations...", end=" ", flush=True)
    recurse_stints([], RACE_LAPS, MAX_STOPS)
    print(f"found {len(all_strategies)} candidates.")
    
    # Deduplicate by stint split first to get count for progress bar
    unique_stints = {}
    for stints in all_strategies:
        stint_split = tuple(s[1] for s in stints)
        if stint_split not in unique_stints:
            unique_stints[stint_split] = stints
    
    # Post-process: Calculate ERT with optimal modes for all unique strategies
    final_strategies = []
    
    for stint_split, stints in tqdm(unique_stints.items(), 
                                     desc="Optimizing pace modes",
                                     unit="strategy",
                                     file=sys.stdout):
        # Find optimal pace modes for this strategy
        optimal_modes, optimal_ert = find_optimal_modes(
            stints, config['pit_loss_seconds'], COMPOUND_DATA, config, INVENTORY
        )
        
        compound_sequence = [s[0] for s in stints]
        
        final_strategies.append({
            'stints': stints,
            'compound_sequence': compound_sequence,
            'split': stint_split,
            'pit_stops': len(stints) - 1,
            'optimal_modes': optimal_modes,
            'ert': optimal_ert
        })
            
    return final_strategies


# --- Specialized Strategy Function ---
def find_clean_air_strategy(config, top_strategies, all_strategies):
    """
    Identifies the Clean Air strategy with rules:
    1. First pit MUST undercut the earliest pit of ALL top strategies
    2. Second pit (if exists) must NOT land on ANY top strategy pit lap
    3. Prefer 1-stop if within 102% of best 2-stop time
    4. Max 2 stops for clean air
    """
    if not top_strategies:
        return None
    
    # Collect ALL pit stop laps from top strategies
    top_pit_laps = set()
    for strat in top_strategies:
        cumulative = 0
        for i in range(len(strat['stints']) - 1):
            cumulative += strat['stints'][i][1]
            top_pit_laps.add(cumulative)
    
    # Find the earliest first pit stop among ALL top strategies
    earliest_first_pit = min(strat['stints'][0][1] for strat in top_strategies)
    
    def get_pit_laps(strat):
        """Get list of pit lap numbers for a strategy."""
        pits = []
        cumulative = 0
        for i in range(len(strat['stints']) - 1):
            cumulative += strat['stints'][i][1]
            pits.append(cumulative)
        return pits
    
    def is_valid_clean_air(strat):
        """Check if strategy meets clean air requirements."""
        pits = get_pit_laps(strat)
        if not pits:
            return False
        
        # Rule 1: First pit must undercut earliest top strategy pit
        if pits[0] >= earliest_first_pit:
            return False
        
        # Rule 2: Second pit (if exists) must not land on any top strategy pit lap
        if len(pits) >= 2:
            if pits[1] in top_pit_laps:
                return False
        
        return True
    
    # Find best 1-stop candidates (must undercut)
    one_stop_candidates = []
    for strat in all_strategies:
        if strat['pit_stops'] != 1:
            continue
        if is_valid_clean_air(strat):
            one_stop_candidates.append(strat)
    
    # Find best 2-stop candidates (must undercut AND avoid second pit conflict)
    two_stop_candidates = []
    for strat in all_strategies:
        if strat['pit_stops'] != 2:
            continue
        if is_valid_clean_air(strat):
            two_stop_candidates.append(strat)
    
    # Get best of each category
    best_1stop = min(one_stop_candidates, key=lambda s: s['ert']) if one_stop_candidates else None
    best_2stop = min(two_stop_candidates, key=lambda s: s['ert']) if two_stop_candidates else None
    
    # Rule 3: Prefer 1-stop if within 102% of 2-stop time
    if best_1stop and best_2stop:
        if best_1stop['ert'] <= best_2stop['ert'] * 1.02:
            return best_1stop
        else:
            return best_2stop
    elif best_1stop:
        return best_1stop
    elif best_2stop:
        return best_2stop
    
    return None

# --- Utility Functions ---
def format_time(seconds):
    """Converts seconds to HH:MM:SS.ms format."""
    hours = floor(seconds / 3600)
    seconds %= 3600
    minutes = floor(seconds / 60)
    seconds %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

def get_pit_laps(stints):
    """Calculates the cumulative lap numbers for the pit stops."""
    cumulative_laps = 0
    pit_laps = []
    
    # Pit 1 occurs at the end of Stint 1, Pit 2 at the end of Stint 2, etc.
    for i in range(len(stints) - 1):
        cumulative_laps += stints[i][1]
        pit_laps.append(f"Lap {cumulative_laps}")
        
    return ", ".join(pit_laps)


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Race Strategy Optimization CLI.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Parse lap times in compound data (convert "M:SS.mmm" to seconds)
    for compound_name, data in config['compounds'].items():
        if 'avg_lap_time' in data:
            data['avg_lap_time_s'] = parse_laptime(data['avg_lap_time'])
        elif 'avg_lap_time_s' not in data:
            raise ValueError(f"Compound {compound_name} missing avg_lap_time or avg_lap_time_s")

    print("--- Race Strategy Optimizer Initializing ---")
    
    # Calculate min stint for display
    min_stint_cfg = config.get('min_stint_laps', 'auto')
    if min_stint_cfg == 'auto':
        min_stint = calculate_min_stint(config)
        print(f"Race Laps: {config['race_laps']}, Pit Loss: {config['pit_loss_seconds']}s, Min Stint: {min_stint} (auto)")
    else:
        print(f"Race Laps: {config['race_laps']}, Pit Loss: {config['pit_loss_seconds']}s, Min Stint: {min_stint_cfg}")
    
    # 1. Generate and Filter All Strategies
    strategies = generate_strategies(config)
    print(f"Successfully generated {len(strategies)} unique, inventory-valid strategies.")

    if not strategies:
        print("Error: No valid strategies could be generated with the given constraints.")
        return

    # 2. Sort by ERT and Rank
    ranked_strategies = sorted(strategies, key=lambda x: x['ert'])
    
    # 3. Deduplicate: keep only the best lap split per (compound_sequence, pit_stops)
    seen_keys = set()
    unique_strategies = []
    for strat in ranked_strategies:
        key = (tuple(strat['compound_sequence']), strat['pit_stops'])
        if key not in seen_keys:
            seen_keys.add(key)
            unique_strategies.append(strat)
    
    # 4. Identify Clean Air Strategy (uses top strategies to find undercut opportunity)
    top_n = config.get('top_strategies', 5)
    clean_air_strat = find_clean_air_strategy(config, unique_strategies[:top_n], ranked_strategies)

    # --- Output Top Strategies ---
    push_cfg = config['pace_modes']['push']
    conserve_cfg = config['pace_modes']['conserve']
    COMPOUND_DATA = config['compounds']
    
    print("\n\n" + "="*80)
    print(f"TOP {top_n} OPTIMAL RACE STRATEGIES (Ranked by Optimized ERT)")
    print(f"Push: {push_cfg['delta_per_lap']:+.2f}s/lap, +{(push_cfg['degradation_factor']-1)*100:.0f}% deg | "
          f"Conserve: {conserve_cfg['delta_per_lap']:+.2f}s/lap, {(conserve_cfg['degradation_factor']-1)*100:.0f}% deg")
    print("="*80)
    
    output_table = []
    
    for i, strat in enumerate(unique_strategies[:top_n]):
        # Format compound + mode per stint: "Soft(P) -> Medium(C) -> Hard(N)"
        mode_abbrev = {'normal': 'N', 'push': 'P', 'conserve': 'C'}
        stint_strs = [f"{comp}({mode_abbrev[mode]})" 
                      for comp, mode in zip(strat['compound_sequence'], strat['optimal_modes'])]
        sequence_str = " -> ".join(stint_strs)
        
        split_str = " - ".join(map(str, strat['split']))
        pit_laps_str = get_pit_laps(strat['stints'])
        
        output_table.append([
            i + 1,
            sequence_str,
            strat['pit_stops'],
            split_str,
            pit_laps_str,
            format_time(strat['ert'])
        ])
        
    headers = ["Rank", "Strategy (Mode)", "Stops", "Stint Split", "Pit Laps", "ERT"]
    print(tabulate(output_table, headers=headers, tablefmt="github"))
    
    # --- Output Clean Air Strategy ---
    COMPOUND_DATA = config['compounds']
    
    # Get earliest first pit from top strategies for context
    earliest_first_pit = min(s['stints'][0][1] for s in unique_strategies[:top_n])
    
    # Collect pit laps from top strategies for clear window analysis
    top_pit_laps = set()
    for strat in unique_strategies[:top_n]:
        cumulative = 0
        for i in range(len(strat['stints']) - 1):
            cumulative += strat['stints'][i][1]
            top_pit_laps.add(cumulative)
    
    if clean_air_strat:
        print("\n\n" + "="*80)
        print("BIAS: CLEAN AIR POSITION STRATEGY")
        print("="*80)
        
        first_stint_laps = clean_air_strat['stints'][0][1]
        pit_stops = clean_air_strat['pit_stops']
        
        # Format with optimal modes
        mode_abbrev = {'normal': 'N', 'push': 'P', 'conserve': 'C'}
        stint_strs = [f"{comp}({mode_abbrev[mode]})" 
                      for comp, mode in zip(clean_air_strat['compound_sequence'], clean_air_strat['optimal_modes'])]
        sequence_str = " -> ".join(stint_strs)
        
        split_str = " - ".join(map(str, clean_air_strat['split']))
        pit_laps_str = get_pit_laps(clean_air_strat['stints'])
        
        # Calculate time penalty relative to the fastest strategy
        time_penalty = clean_air_strat['ert'] - ranked_strategies[0]['ert']
        
        clean_air_output = [
            ["Strategy", sequence_str],
            ["Stint Split", split_str],
            ["Pit Stops", pit_stops],
            ["Pit Laps", pit_laps_str],
            ["Optimized ERT", format_time(clean_air_strat['ert'])],
            ["vs Rank 1", f"+{time_penalty:.3f}s"]
        ]
        
        print(tabulate(clean_air_output, tablefmt="plain"))
        
        # Per-stint mode breakdown
        print("\n**Stint Modes:**")
        for i, (stint, mode) in enumerate(zip(clean_air_strat['stints'], clean_air_strat['optimal_modes'])):
            compound, laps = stint
            mode_label = mode.upper()
            print(f"  Stint {i+1}: {compound} ({laps} laps) - {mode_label}")
        
        # Justification based on strategy type
        print(f"\n**Undercut Target:** Lap {earliest_first_pit} (earliest pit among top strategies)")
        print(f"**First Pit:** Lap {first_stint_laps} (undercut by {earliest_first_pit - first_stint_laps} lap(s))")
        
        if pit_stops == 1:
            print("\n**Recommendation:** 1-STOP avoids traffic from 2-stoppers pitting mid-race")
        else:
            # Calculate second pit lap
            second_pit = first_stint_laps + clean_air_strat['stints'][1][1]
            nearby_pits = [lap for lap in top_pit_laps if abs(lap - second_pit) <= 3]
            print(f"\n**Second Pit:** Lap {second_pit}", end="")
            if not nearby_pits:
                print(" (CLEAR WINDOW - no top strategy pits within ±3 laps)")
            else:
                print(f" (nearby pits: {sorted(nearby_pits)})")
    else:
        print("\n[INFO] Could not find a suitable Clean Air Strategy.")


if __name__ == "__main__":
    main()