"""Output formatting and printing functions."""
from math import floor

from tabulate import tabulate

from models import (
    RaceConfig, Strategy, SCAnalysis, TireDomainAnalysis,
    UndercutAnalysis, DRSAnalysis, AttackAnalysis, LiveAnalysis
)


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.ms format."""
    hours = floor(seconds / 3600)
    seconds %= 3600
    minutes = floor(seconds / 60)
    seconds %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def format_laptime(seconds: float) -> str:
    """Format lap time as M:SS.ms"""
    minutes = floor(seconds / 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


def is_soft_opener(strat: Strategy) -> bool:
    """Check if strategy opens on Soft tires."""
    return strat.compound_sequence[0] == 'Soft'


def is_soft_bookend(strat: Strategy) -> bool:
    """Check if strategy starts and ends on Soft with different compound in middle."""
    seq = strat.compound_sequence
    if len(seq) < 3:
        return False
    # Starts with Soft, ends with Soft, middle is different
    return seq[0] == 'Soft' and seq[-1] == 'Soft' and any(c != 'Soft' for c in seq[1:-1])


def find_preferred_strategies(strategies: list[Strategy], best_ert: float) -> dict[str, Strategy | None]:
    """
    Find best strategies matching racing preferences:
    - soft_opener_1stop: Best 1-stop starting on Soft (Soft -> M/H)
    - soft_bookend_2stop: Best 2-stop with Soft bookends (Soft -> M/H -> Soft)
    """
    preferred = {
        'soft_opener_1stop': None,
        'soft_bookend_2stop': None,
    }
    
    for strat in strategies:
        # 1-stop soft opener: Soft -> Medium or Soft -> Hard
        if strat.pit_stops == 1 and is_soft_opener(strat):
            if strat.compound_sequence[1] in ('Medium', 'Hard'):
                if preferred['soft_opener_1stop'] is None or strat.ert < preferred['soft_opener_1stop'].ert:
                    preferred['soft_opener_1stop'] = strat
        
        # 2-stop soft bookend: Soft -> X -> Soft where X is Medium or Hard
        if strat.pit_stops == 2 and is_soft_bookend(strat):
            if preferred['soft_bookend_2stop'] is None or strat.ert < preferred['soft_bookend_2stop'].ert:
                preferred['soft_bookend_2stop'] = strat
    
    return preferred


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


def print_preferred_strategies(
    preferred: dict[str, Strategy | None],
    best_ert: float,
    config: RaceConfig
) -> None:
    """Print preferred strategy patterns based on racing preferences."""
    has_any = any(s is not None for s in preferred.values())
    
    print("\n\n" + "=" * 80)
    print("PREFERRED STRATEGY PATTERNS")
    print("=" * 80)
    
    if not has_any:
        print("\nNo strategies found matching preferred patterns.")
        print("(Soft opener 1-stop or Soft bookend 2-stop)")
        return
    
    # Soft opener 1-stop
    strat = preferred.get('soft_opener_1stop')
    if strat:
        delta = strat.ert - best_ert
        print(f"\n1-STOP SOFT OPENER (Soft -> M/H for track position at start)")
        print(f"  Strategy: {strat.format_strategy_string()}")
        print(f"  Stint Split: {strat.format_split_string()}")
        print(f"  Pit Lap: {strat.format_pit_laps()}")
        print(f"  ERT: {format_time(strat.ert)} ({'+' if delta >= 0 else ''}{delta:.2f}s vs optimal)")
    
    # Soft bookend 2-stop
    strat = preferred.get('soft_bookend_2stop')
    if strat:
        delta = strat.ert - best_ert
        print(f"\n2-STOP SOFT BOOKEND (Soft -> M/H -> Soft for start & finish pace)")
        print(f"  Strategy: {strat.format_strategy_string()}")
        print(f"  Stint Split: {strat.format_split_string()}")
        print(f"  Pit Laps: {strat.format_pit_laps()}")
        print(f"  ERT: {format_time(strat.ert)} ({'+' if delta >= 0 else ''}{delta:.2f}s vs optimal)")


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
        print(f"Position Penalty: {analysis.positions_lost} positions x {config.position_loss_value:.1f}s = "
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

