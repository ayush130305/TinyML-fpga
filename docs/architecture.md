# Architecture

## Split of work

| Where | What |
|---|---|
| PC | training (Keras), int8 export, `.tflite → .tmf` conversion with every multiplier/shift precomputed, bit-exact check against LiteRT |
| Host: PC (CPython) or MCU (MicroPython) | loads the `.tmf`, keeps weights resident on the FPGA when they fit, builds conv/depthwise patches (im2col), batches jobs, runs pooling / reshape / softmax |
| FPGA | `tmf_engine`: batched int8 dot products with exact TFLite requantisation |

## The engine (`rtl/tmf_engine.v`)

A job is configured with K, N, NVEC, XSTRIDE, WBASE, PBASE, IN_OFF, OUT_ZP, ACT_MIN/MAX and MODE, then started:

```
for v in 0..NVEC-1:                 # input vectors (one per output pixel for conv)
  for o in 0..N-1:                  # output channels
    acc = bias[o] + sum_k (X[v*XSTRIDE + (DW ? o*K : 0) + k] + IN_OFF) * W[WBASE + o*K + k]
    Y[v*N + o] = clamp(((acc*mult[o] + 2^(ts-1)) >>> ts) + OUT_ZP, ACT_MIN, ACT_MAX),  ts = 31 - shift[o]
```

- **Memories:** four simple dual-port byte RAMs with registered reads: X (inputs), W (weights), P (16 bytes per channel: bias i32, multiplier i32, shift i8), and Y (outputs). Sizes are parameters (`*_AW`) and are reported in read-only registers, so the driver adapts to any build.
- **Datapath:** one 9×8-bit signed MAC per clock, 3-stage pipeline (RAM read → product register → accumulate).
- **Requantiser (`tmf_requant.v`):** 32-step shift-and-add multiply, then a rounding add, then an arithmetic shift in 8-bit and 1-bit steps, then the zero point and an int8 clamp. One shared 64-bit adder. This is TFLite's `MultiplyByQuantizedMultiplier` in its single-rounding form, the variant LiteRT executes (verified; the older double-rounding form differs by 1 LSB on about 0.3% of outputs).
- **Safety:** before every output the engine checks that all X/W/P/Y accesses are in range; a bad job stops with STATUS = error | done instead of corrupting memory. Bus writes are ignored while a job runs.
- **Cost:** `K + 54 + ts/8 + ts%8` clocks per output (raw mode `K + 20`). The Python register model computes the same number, and the testbench checks it.

## Register map (byte bus)

| Address | Name | Access | Meaning |
|---|---|---|---|
| 0x100 | K | u16 RW | dot length |
| 0x102 | N | u16 RW | output channels |
| 0x104 | NVEC | u16 RW | vectors per job |
| 0x106 | XSTRIDE | u16 RW | bytes between vectors in X |
| 0x108 | WBASE | u16 RW | byte offset of channel 0's weights |
| 0x10A | PBASE | u16 RW | channel index of channel 0's parameters |
| 0x10C | IN_OFF | s16 RW | −input zero point |
| 0x10E | OUT_ZP | s8 RW | output zero point |
| 0x10F / 0x110 | ACT_MIN / ACT_MAX | s8 RW | clamp |
| 0x111 | MODE | RW | bit0 depthwise, bit1 raw int32 output |
| 0x11F | CTRL | W | 0x01 start, 0x80 clear status (last byte of the config frame, so one frame configures and starts) |
| 0x120 | STATUS | R | bit0 busy, bit1 done, bit2 error |
| 0x121 | CYCLES | u32 R | clocks of the last job |
| 0x130–0x135 | MAGIC 0x4D, VERSION, W_AW, X_AW, Y_AW, P_AW | R | identification and sizes |
| 0x1000 / 0x2000 / 0x3000 / 0x8000 | X / Y / P / W windows | | Y is read-only from the bus |

Addresses below 0x100 are unused by the engine, so it can share a bus with other registers.

## Runtime planning (`tinyml_fpga/model.py`)

- **Residency:** if all engine-layer weights and parameters fit in W and P, they are uploaded once at `load()`. Otherwise each layer's weights are streamed in before it runs.
- **Channel chunks:** a layer is split into channel groups that fit W, P, Y and (for depthwise) X.
- **Batching:** as many vectors per job as X and Y allow. One write frame burst for X, one config frame that also starts the job, a status poll, then one read for Y.
- **Fallback:** a layer whose single vector cannot fit in X runs in software, with identical results.

## Verification chain

```
LiteRT  ==bit-exact==  kernels.py (CpuEngine)  ==same code==  SimBus register model  ==tb_engine.v==  tmf_engine.v
                                                                     |
                                    recorded real inferences ==tb_uart_top.v==  uart_bus + engine
```
