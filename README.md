# F1 Manager Strategy Tool

A command-line tool for optimizing race strategy decisions in F1 Manager games. Features include pre-race strategy generation, safety car pit decisions, tire domain analysis, undercut/overcut calculations, and gap management tools.

## Installation

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

## Configuration

All race parameters are configured in `config.yaml`. Key sections:

- **General**: Race laps, pit loss time, strategy limits, F1 compound regulations
- **Safety Car**: SC pit loss, conservation settings
- **Tactical Analysis**: DRS threshold, attack mode settings
- **Pace Modes**: Push/conserve lap time deltas and degradation factors
- **Tire Inventory**: Available sets of each compound
- **Compounds**: Lap times, degradation rates, and competitive life per compound

## Commands

### `race` - Full Strategy Generation

Generate optimal race strategies ranked by Estimated Race Time (ERT).

```bash
uv run python cli.py race [--config CONFIG]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | string | `config.yaml` | Path to configuration file |

**Example:**
```bash
uv run python cli.py race
uv run python cli.py race --config tracks/monaco.yaml
```

---

### `sc` - Safety Car Pit Decision

Analyze whether to pit under safety car or stay out.

```bash
uv run python cli.py sc [OPTIONS]
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--config` | string | No | Path to config file (default: `config.yaml`) |
| `--current-lap` | int | Yes | Current lap number (remaining calculated from `race_laps` in config) |
| `--stint-laps` | int | Yes | Laps completed on current tires |
| `--compound` | string | Yes | Current tire compound: `Soft`, `Medium`, `Hard` |
| `--pos-loss` | int | No | Estimated positions lost if you pit (default: 0) |

**Examples:**
```bash
# Interactive mode
uv run python cli.py sc

# Direct input (remaining laps calculated from config.race_laps - current_lap)
uv run python cli.py sc --current-lap 20 --stint-laps 10 --compound Medium

# With position loss estimate
uv run python cli.py sc --current-lap 15 --stint-laps 15 --compound Soft --pos-loss 2
```

---

### `live` - Mid-Race Strategy Recalculation

Calculate optimal strategy from your current mid-race position.

```bash
uv run python cli.py live [OPTIONS]
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--config` | string | No | Path to config file (default: `config.yaml`) |
| `--current-lap` | int | Yes | Current lap number (remaining calculated from `race_laps` in config) |
| `--compound` | string | Yes | Current tire compound: `Soft`, `Medium`, `Hard` |
| `--tire-laps` | int | Yes | Laps on current tires |

**Examples:**
```bash
# Interactive mode
uv run python cli.py live

# Direct input - mid-race recalculation
uv run python cli.py live --current-lap 25 --compound Medium --tire-laps 15

# After a red flag restart
uv run python cli.py live --current-lap 30 --compound Soft --tire-laps 10
```

**Output includes:**
- Current race position context (Lap X of Y)
- Tire status (wear %, competitive laps remaining, cliff lap)
- Top strategies ranked by ERT
- Pit lap recommendations (actual lap numbers, not "in X laps")
- Next tire compound for each strategy

---

### `tires` - Tire Domain Analysis

Analyze which compound is fastest at each lap of a stint, identifying crossover points.

```bash
uv run python cli.py tires [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | string | `config.yaml` | Path to config file |
| `--mode` | string | `normal` | Pace mode: `normal`, `push`, `conserve` |
| `--starting-wear` | int | `0` | Starting tire wear in laps |
| `--chart` | flag | off | Display ASCII domain chart |

**Examples:**
```bash
# Basic analysis
uv run python cli.py tires

# With ASCII chart
uv run python cli.py tires --chart

# Analyze push mode domains
uv run python cli.py tires --mode push --chart
```

---

### `undercut` - Undercut/Overcut Calculator

Determine optimal pit timing relative to a rival.

```bash
uv run python cli.py undercut [OPTIONS]
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--config` | string | No | Path to config file (default: `config.yaml`) |
| `--gap` | float | Yes | Gap to rival in seconds (positive = ahead, negative = behind) |
| `--current-lap` | int | Yes | Current race lap |
| `--rival-pit` | int | Yes | Expected lap rival will pit |
| `--your-tire-laps` | int | Yes | Laps on your current tires |
| `--rival-tire-laps` | int | Yes | Laps on rival's current tires |
| `--your-compound` | string | Yes | Your compound: `Soft`, `Medium`, `Hard` |
| `--rival-compound` | string | Yes | Rival's compound: `Soft`, `Medium`, `Hard` |
| `--pit-to` | string | No | Compound to pit onto (default: fastest available) |

**Examples:**
```bash
# Interactive mode
uv run python cli.py undercut

# You're 2.3s behind, rival pits lap 22
uv run python cli.py undercut --gap -2.3 --current-lap 18 --rival-pit 22 \
    --your-tire-laps 18 --rival-tire-laps 22 \
    --your-compound Medium --rival-compound Medium

# Specify pit-to compound
uv run python cli.py undercut --gap -1.5 --current-lap 20 --rival-pit 25 \
    --your-tire-laps 20 --rival-tire-laps 25 \
    --your-compound Hard --rival-compound Hard --pit-to Soft
```

---

### `drs` - DRS Defense Calculator

Decide whether to push to deny DRS or conserve tires when under pressure.

```bash
uv run python cli.py drs [OPTIONS]
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--config` | string | No | Path to config file (default: `config.yaml`) |
| `--gap` | float | Yes | Gap to car behind in seconds |
| `--stint-laps-remaining` | int | Yes | Laps remaining until stint end |
| `--your-tire-laps` | int | Yes | Laps on your current tires |
| `--attacker-tire-laps` | int | Yes | Laps on attacker's tires |
| `--your-compound` | string | Yes | Your compound: `Soft`, `Medium`, `Hard` |
| `--attacker-compound` | string | Yes | Attacker's compound: `Soft`, `Medium`, `Hard` |

**Examples:**
```bash
# Interactive mode
uv run python cli.py drs

# Car behind at 0.8s, 12 laps to go
uv run python cli.py drs --gap 0.8 --stint-laps-remaining 12 \
    --your-tire-laps 18 --attacker-tire-laps 8 \
    --your-compound Medium --attacker-compound Soft
```

**Output includes:**
- CONSERVE scenario (accept DRS, defend on track)
- PUSH scenario (push entire stint)
- BURST_PUSH scenario (optimal push duration then conserve)
- Recommendation with tire sustainability analysis

---

### `attack` - Attack/Catch Calculator

Analyze whether to actively chase a car ahead or let them come back naturally.

```bash
uv run python cli.py attack [OPTIONS]
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--config` | string | No | Path to config file (default: `config.yaml`) |
| `--gap` | float | Yes | Gap to car ahead in seconds |
| `--stint-laps-remaining` | int | Yes | Laps remaining until stint end |
| `--your-tire-laps` | int | Yes | Laps on your current tires |
| `--target-tire-laps` | int | Yes | Laps on target's tires |
| `--your-compound` | string | Yes | Your compound: `Soft`, `Medium`, `Hard` |
| `--target-compound` | string | Yes | Target's compound: `Soft`, `Medium`, `Hard` |

**Examples:**
```bash
# Interactive mode
uv run python cli.py attack

# 2.5s behind, 15 laps remaining
uv run python cli.py attack --gap 2.5 --stint-laps-remaining 15 \
    --your-tire-laps 10 --target-tire-laps 20 \
    --your-compound Soft --target-compound Medium
```

**Output includes:**
- Natural convergence analysis (will they come back?)
- STAY_ON_PLAN scenario (normal pace)
- PUSH scenario (push mode)
- ATTACK scenario (2x degradation, fastest closing)
- Tire burn warning if attack would exceed tire life by stint end

---

## Interactive Mode

All tactical tools (`sc`, `live`, `undercut`, `drs`, `attack`) support interactive mode. Simply run without arguments:

```bash
uv run python cli.py sc
uv run python cli.py live
uv run python cli.py undercut
uv run python cli.py drs
uv run python cli.py attack
```

You'll be prompted for each required input.

---

## Project Structure

```
f1_manager_strategy/
  models.py    - Data models (dataclasses)
  core.py      - ERT calculation, degradation model, tire utilities
  analysis.py  - Strategy generation and all analyzer functions
  output.py    - Output formatting and print functions
  config.py    - Configuration loading
  cli.py       - CLI entry point (argparse, command routing)
```

---

## Config Reference

### F1 Regulations

```yaml
require_medium_or_hard: true  # Must use at least one Medium or Hard tire
require_two_compounds: true   # Must use at least 2 different compounds
```

### Tactical Analysis Settings

```yaml
# --- TACTICAL ANALYSIS ---
drs_threshold_seconds: 1.0  # Gap required to deny DRS
dirty_air_loss_per_lap: 0.5 # Pace loss when following closely
inlap_push_gain: 0.3        # Seconds faster on in-lap (pushing before pit)
outlap_penalty: 1.5         # Seconds slower on out-lap (cold tires)
```

### Pace Modes

```yaml
pace_modes:
  push:
    delta_per_lap: -0.150     # Seconds gained per lap
    degradation_factor: 1.285 # ~29% faster tire wear
  conserve:
    delta_per_lap: 0.50       # Seconds lost per lap
    degradation_factor: 0.84  # ~16% slower tire wear
  attack:
    delta_per_lap: -0.35      # Seconds gained (more aggressive than push)
    degradation_factor: 2.0   # 2x tire wear rate
```

### Tire Degradation Model

The tool uses a progressive degradation model:
- **Early laps**: ~10% of average degradation (tires in sweet spot)
- **Approaching max life**: Curves up to 150% (the cliff)
- **Beyond max life**: 2x+ degradation rate (tires are done)

This models real tire behavior where degradation is low early, then falls off a cliff.
