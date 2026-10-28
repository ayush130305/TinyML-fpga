# Engine features by version

The engine is built from one parameterised module. Each feature is a parameter that is resolved when the design is elaborated, so a feature you do not turn on costs no logic. The numbers below come from `tools/measure.py` (Yosys, iCE40 mapping, engine only) and `tools/bench_cycles.py` (clock counts per inference, exact). Resource figures are LUT4 counts before the vendor calibration in `tools/calibration.json`.

## v0.3: parallel lanes and a hidden requantiser

| Build | LUT4 | multipliers | RAM 4k blocks | m_fc | m_cnn | m_avg |
|---|---|---|---|---|---|---|
| v0.2 engine | n/a | 1 | 13 | 4,868 | 47,223 | 31,072 |
| `LANES=1 RQ_MODE=0` | 1,986 | 1 | 13 | 3,127 | 30,930 | 21,860 |
| `LANES=1 RQ_MODE=1` | 2,452 | 5 | 13 | 3,062 | 17,772 | 9,817 |
| `LANES=2 RQ_MODE=1` | 2580 | 6 | 13 | 1,544 | 10,511 | 7,447 |
| `LANES=4 RQ_MODE=1` | 2,936 | 8 | 15 | 810 | 6,899 | 6,639 |
| `LANES=8 RQ_MODE=1` | 3,727 | 12 | 19 | 446 | 5,099 | 5,887 |

Clocks are engine clocks per inference with all weights resident (`tools/bench_cycles.py`). Link and host time come on top.

- **`LANES`** (1, 2, 4, 8): output channels computed at once. They share one input stream; weights and parameters live in per-lane banks. Depthwise layers use one lane at a time (each channel has its own input slice).
- **`RQ_MODE`**: 0 = serial requantiser, no multiplier; 1 = pipelined requantiser, four 16×16 multipliers, one result per clock. With several lanes use 1: the serial unit would handle the lanes one after another.
- **Overlap**: requantisation of a finished group runs in the background while the next group is multiplied (all builds).
