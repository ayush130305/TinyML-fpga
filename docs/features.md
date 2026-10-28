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

## v0.5: on-chip window generator and layer fusion

`HAS_WIN=1` adds a window generator. With MODE bit 2 (`MODE_WIN`) a convolution or depthwise job reads the **image itself** (NHWC, int8) from X and walks the window of every output pixel with address counters: kernel size, stride, dilation, top / left padding and the depth multiplier come from a 31-byte window block (0x140..0x15E, write-only; in a descriptor it follows the 22 config bytes). Taps outside the image are forced to zero in the datapath (the same as a padding byte equal to the input zero point), so the host neither builds im2col patches nor sends them: a 3×3 conv over C channels used to ship 9·C bytes per output pixel and now ships each input byte once. The clock count of a job is unchanged: the address generator runs alongside the multiply-accumulate loop and costs no clocks.

```python
win = window_block(H, W, C, kh, kw, dh, dw, sh, sw, ph, pw, dm, ow, r0, xbase)   # None if a field does not fit
eng.run(K, N, nrows * ow, 0, wbase, pbase, in_off, zp, amin, amax, MODE_WIN, image, xbase, ybase, win)
```

`r0` is the first output row: a layer whose outputs do not fit Y is cut into jobs of several rows each, all reading the same image. `Model` does all of this by itself when the engine has the feature and every descriptor field can hold the geometry; otherwise a layer falls back to host-built patches.

`HAS_FUSE=1` lets a job write its outputs into X instead of Y (MODE bit 3, `MODE_OUTX`; YBASE is then an X address). Together with the queue, consecutive layers run as one chain: each layer reads the previous layer's output where it was written, two regions of X are used alternately, and only the last layer writes to Y. Pure reshapes (flatten) between layers cost nothing. `Model` plans such chains at load time (weights resident, one job per layer, everything fitting X, Y and the descriptor slots) and runs each with one descriptor upload and one start. Pooling and other host ops end a chain, and debugging runs (`invoke(keep=True)`, used by the LiteRT checker) never fuse so that every intermediate tensor exists.

Link traffic of one inference (bus frames / bytes, `tools/bench_link.py`; a UART frame costs a round trip):

| Build | m_fc | m_cnn | m_avg |
|---|---|---|---|
| host im2col | 8 / 212 | 31 / 2,630 | 22 / 1,801 |
| `HAS_WIN` | 8 / 212 | 22 / 1,243 | 17 / 919 |
| `HAS_WIN` + `HAS_QUEUE` + `HAS_FUSE` | 5 / 215 | 14 / 987 | 10 / 540 |

Cost (engine only, 1 lane, serial requantiser, Yosys LUT4 / FF; `tools/measure.py`):

| Build | LUT4 | FF | RAM 4k blocks |
|---|---|---|---|
| base | 2,061 | 1,058 | 13 |
| + `HAS_FUSE` | 2,090 | 1,088 | 13 |
| + `HAS_WIN` | 2,882 | 1,477 | 13 |
| + `HAS_WIN`, `HAS_FUSE`, `HAS_QUEUE` | 3,101 | 1,526 | 14 |

The window generator is the largest single feature (the adders for the address counters and the 31-byte block); it pays off when the link, not the FPGA, limits the inference time, which is the usual case behind a UART or a slow SPI.

## v0.6: INT4 / INT8 mixed precision

`HAS_W4=1` lets a layer keep its weights as signed 4-bit numbers, two per W byte (low nibble = even index; a row of K weights takes ceil(K/2) bytes and rows start on a byte). A layer selects it with MODE bit 4 (`MODE_W4`), so int8 and int4 layers mix freely in one model. The multiply-accumulate datapath is unchanged: a nibble selector in front of every lane sign-extends the weight to 8 bits, and the weight address advances every second clock. A layer takes the same number of clocks as at 8 bits and its weights take half the weight RAM, so models up to twice the size stay resident (no per-inference weight upload) and a smaller `W_AW` is enough.

The accuracy cost is handled on the PC. `python -m tinyml_fpga convert M.tflite --int4 auto` re-quantises layers to 4 bits per output channel (`w8 ≈ r·w4`, `r = max|w| / 7`) and folds the row scale into the layer's bias and fixed-point multiplier, so the engine's output scale is unchanged. `auto` leaves the first and last weight layer at 8 bits (the most sensitive ones), `all` converts everything, `1,2` picks weight layers by number. The command prints each layer's weight error and how far the model's outputs move against the int8 model. The 4-bit model is **not** bit-exact with LiteRT, by nature; what stays exact is hardware against software: the engine, the register model and the CPU engine agree on every output of the 4-bit model.

| Model | .tmf bytes int8 | int4 `all` | int4 `auto` |
|---|---|---|---|
| m_fc | 3,713 | 2,549 | 3,713 (nothing in the middle) |
| m_cnn | 3,565 | 2,882 | 2,975 |
| m_avg | 2,069 | 1,877 | 2,055 |

Measured on the digits samples: `m_cnn` int4 `all` keeps the top-1 label on 100% of 40 test images (`transform.compare`). A runtime that meets a 4-bit layer on an engine without `HAS_W4` runs that layer in software, so the same `.tmf` works everywhere.

Cost (engine only, Yosys LUT4): +48 with 1 lane (2,050 → 2,098), +87 with 4 lanes (3,047 → 3,134) for the nibble selectors and the row-stride logic; no extra RAM or multipliers.

## v0.7: compressed weights with hardware decompression (codebook)

`HAS_CB=1` stores a layer's weights as 4-bit **indices** into a 16-entry int8 table (MODE bit 5, `MODE_CB`). The indices are packed like the 4-bit weights (two per byte, rows of ceil(K/2) bytes), and a 16:1 selector in every lane replaces the nibble's sign extension with the table value, so decompression happens in the datapath at full speed. The table is not a new memory: it is one 16-byte **parameter record** placed right after the layer's own records in parameter bank 0 (local index `PBASE + ceil(N / LANES)`). At job start the engine spends 17 clocks copying it into a register file; a table that does not fit the bank is a job error.

Compared with plain int4 the table is *fitted to the weights* (1-D k-means in `transform.to_codebook`), so the same 4 bits per weight give a smaller error, and no scale has to be folded into bias and multiplier: the table holds the original int8 values. The cost is 16 bytes per layer.

```bash
python -m tinyml_fpga convert M.tflite --codebook auto     # auto / all / layer numbers, like --int4
```

| m_cnn, all layers | .tmf bytes | weight error (RMS, % of weights) per layer |
|---|---|---|
| int8 | 3,565 | 0 |
| int4 (uniform) | 2,882 | 6.5, 6.5, 9.1, 7.3 |
| codebook | 2,964 | 5.0, 6.3, 8.5, 5.9 |

`driver`: `eng.upload_cb(PBASE + rows, table)`; `Model` places and uploads tables itself, streams them with each channel chunk when weights do not fit, and runs a codebook layer in software on an engine without the feature. int4 and codebook layers mix freely with int8 ones in one model.

Cost (engine only, Yosys LUT4): +261 with 1 lane (2,106 → 2,367, 129 more flip-flops for the table), +780 with 4 lanes (2,695 → 3,475) because every lane has its own selector. Use it where weight memory or accuracy at 4 bits matters more than LUTs; plain `HAS_W4` is the cheap option.
