import yaml
from main import generate_strategies, parse_laptime

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Parse lap times
for name, data in config['compounds'].items():
    if 'avg_lap_time' in data:
        data['avg_lap_time_s'] = parse_laptime(data['avg_lap_time'])

strategies = generate_strategies(config)

# Count by pit stops
by_stops = {}
for s in strategies:
    stops = s['pit_stops']
    by_stops[stops] = by_stops.get(stops, 0) + 1

print('\nStrategies by pit stops:')
for stops in sorted(by_stops.keys()):
    print(f'  {stops}-stop: {by_stops[stops]}')

# Show best for each stop count
print('\nBest strategy per stop count:')
for stops in sorted(by_stops.keys()):
    strats = [s for s in strategies if s['pit_stops'] == stops]
    best = min(strats, key=lambda x: x['ert'])
    mins = best['ert'] / 60
    print(f"  {stops}-stop: {best['compound_sequence']} {best['split']} -> {best['ert']:.2f}s ({mins:.2f} min)")

