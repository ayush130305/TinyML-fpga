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

## v0.4: job descriptors, command queue, performance counters

`HAS_QUEUE=1` adds a descriptor RAM (`D_AW`, 64 bytes per slot, default 8 slots at 0x4000). A descriptor holds the same 22 bytes as the config block (0x100..0x115) plus the two new fields **XBASE** (where the job's input vectors start in X) and **YBASE** (where its outputs go in Y), so several jobs can live in X and Y at the same time. `QCNT` (0x118) says how many descriptors to run and `CTRL = 0x02` starts them back to back; `QDONE` (0x119) tells how many finished, so after an error it is the index of the failing job. Loading a descriptor takes 23 clocks. XBASE and YBASE also work for plain `CTRL = 0x01` jobs on every v2 build.

```python
eng.run_queue([(K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode, xbase, ybase), ...])
y = eng.read_y(ybase, nvec * N)
```

Six small jobs take 5 link frames through the queue instead of 12 (a UART frame costs a round trip each), at the price of the descriptor bytes. The queue is what v0.5's layer fusion runs on.

`HAS_PERF=1` adds four read-only 32-bit counters at 0x160, cleared by every start: **MAC clocks** (clocks spent issuing multiply-accumulates), **stall clocks** (a finished group waiting because the output unit was still busy), **output bytes** and **finished jobs**. `eng.perf()` returns them; MAC clocks divided by the CYCLES counter is the utilisation of the multipliers. The register model reproduces all four exactly.

| Build (1 lane, serial requantiser) | LUT4 | FF | RAM 4k blocks |
|---|---|---|---|
| base | 1,986 | 1,042 | 13 |
| + `HAS_QUEUE` | 2,184 | 1,077 | 14 |
| + `HAS_PERF` | 2,292 | 1,186 | 13 |
