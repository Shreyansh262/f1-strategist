# Simulation validation — historical replay
Engine v1 (no safety-car model). N=2000 rollouts/race, seed fixed.
**Overall: 0.79 of drivers' actual finishing positions fall inside their central-80% simulated band** (target ≈ 0.80).

| Circuit | coverage |
|---|---|
| Bahrain Grand Prix | 0.53 |
| Hungarian Grand Prix | 0.78 |
| Japanese Grand Prix | 1.00 |
| Monaco Grand Prix | 1.00 |

Misses concentrate where reality included SC/red-flag phases or first-lap incidents — dynamics the v1 engine deliberately excludes (documented scope).
Driver pace is leave-one-race-out; the race's own laps never inform its sim except the field-median track-condition anchor.
