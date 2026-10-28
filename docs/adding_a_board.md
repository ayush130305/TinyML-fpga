# Adding a board

The engine is independent of the board and the FPGA vendor. It sits on a plain byte bus:

```verilog
input  [15:0] bus_addr,   // stable at least one clock before bus_rd
input         bus_wr,     // one-clock strobe, bus_addr / bus_wdata valid
input  [7:0]  bus_wdata,
input         bus_rd,     // one-clock strobe; bus_rdata is sampled in this clock
output [7:0]  bus_rdata
```

So a board only needs something that drives this bus:

| Host link | Bus master (RTL) | Host side (Python) |
|---|---|---|
| USB-UART from a PC (any board) | `rtl/uart_bus.v`, ready-made in `rtl/tinyml_uart_top.v` | `backend="uart:PORT"` (pyserial) |
| SPI master (MCU, Linux board) | `rtl/spi_bus.v`, ready-made in `rtl/tinyml_spi_top.v` | `backend="spi:0.0"` or `McuSpiBus(spi, cs)` |
| I2C master (MCU, Linux board) | `rtl/i2c_bus.v`, ready-made in `rtl/tinyml_i2c_top.v` | `backend="i2c:1:0x2A"` or `McuI2cBus(i2c)` |
| CPU behind AXI4-Lite (Zynq, soft core) | `rtl/axil_bus.v`, ready-made in `rtl/tinyml_axil_top.v` | `backend="axi:0x40000000"` or `AxiLiteBus(mem)` |
| anything else (a parallel bus from an MCU, …) | your bridge to the byte bus | a class with `write_block(addr, buf)` and `read_block(addr, n)`, passed as `load(path, backend=HwEngine(MyBus()))` |

A custom bus class runs on CPython or MicroPython. Frames of up to 128 bytes are enough; the runtime never needs more per call.

## Using the UART top on a board

1. Add `rtl/tinyml_uart_top.v`, `rtl/uart_bus.v`, `rtl/uart_rx.v`, `rtl/uart_tx.v`, `rtl/tmf_engine.v`, `rtl/tmf_lane.v`, `rtl/tmf_rq.v`, `rtl/tmf_requant.v`, `rtl/tmf_requant_pipe.v` (only needed with `RQ_MODE=1`) and `rtl/tmf_ram.v` to your FPGA project, with `tinyml_uart_top` as the top module (or instantiate it from your own top).
2. Set the parameters: `CLK_HZ` (your clock), `BAUD`, the RAM sizes (`W_AW` ≤ 15, `X_AW`/`Y_AW`/`P_AW` ≤ 12) to what the device has, and optionally `LANES` (1, 2, 4, 8) and `RQ_MODE` (0 serial, 1 pipelined; use 1 with several lanes). Sizes are totals over all banks.
3. In your vendor's constraint file, connect `clk`, `rst` (active high, e.g. a button), `uart_rx`, `uart_tx` and optionally `status[2:0]` to LEDs, and add the clock constraint.
4. On the PC: `tinyml_fpga.load("model.tmf", backend="uart:COM5", baud=921600)`. The driver reads the RAM sizes from the engine, so no software changes are needed.

## Simulating your top

`rtl/sim/tb_uart_top.v` replays a recorded real inference (`rtl/sim/gen_system_vectors.py`) through `tinyml_uart_top` and checks every byte read back. For a different top or RAM sizes, change the instance and set the same sizes in `AW` in `rtl/sim/gen_system_vectors.py` before recording.

## RAM budget

| RAM | Holds | Typical need |
|---|---|---|
| W | int8 weights, all resident layers | sum of K·N over conv/dw/fc layers |
| P | 16 B per output channel | sum of N |
| X | a batch of input vectors | at least K (dense) or K·N_chunk (depthwise) |
| Y | a batch of outputs | at least N_chunk |

Too-small RAMs still work, just slower: weights get streamed, layers split into channel chunks, and anything that cannot fit runs in software. `python -m tinyml_fpga info model.tmf --sizes W X Y P` shows the plan for a given build.
