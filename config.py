"""Configuration loading and parsing."""
import re
import yaml

from models import RaceConfig, Compound, Inventory, PaceMode


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

    # Parse pace modes (always include 'normal' as baseline)
    pace_modes: dict[str, PaceMode] = {
        'normal': PaceMode(name='normal', delta_per_lap=0.0, degradation_factor=1.0)
    }
    for mode_name, mode_data in raw['pace_modes'].items():
        pace_modes[mode_name] = PaceMode(
            name=mode_name,
            delta_per_lap=mode_data['delta_per_lap'],
            degradation_factor=mode_data['degradation_factor']
        )

    # Validate degradation model
    deg_model = raw.get('degradation_model', 'progressive')
    if deg_model not in ('progressive', 'linear'):
        raise ValueError(f"Invalid degradation_model: {deg_model}. Must be 'progressive' or 'linear'.")

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
        require_two_compounds=raw.get('require_two_compounds', True),
        degradation_model=deg_model,
        sc_pit_loss_seconds=raw.get('sc_pit_loss_seconds', 5.0),
        sc_conserve_laps=raw.get('sc_conserve_laps', 3),
        sc_conserve_factor=raw.get('sc_conserve_factor', 0.5),
        position_loss_value=raw.get('position_loss_value', 2.5),
        drs_threshold_seconds=raw.get('drs_threshold_seconds', 1.0),
        dirty_air_loss_per_lap=raw.get('dirty_air_loss_per_lap', 0.5),
        inlap_push_gain=raw.get('inlap_push_gain', 0.3),
        outlap_penalty=raw.get('outlap_penalty', 1.5)
    )

