// tinyml_uart_top.v - ready-to-use top: UART bridge + int8 engine, for any FPGA board.
// Driven from a PC with tinyml_fpga.load(..., backend="uart:PORT", baud=BAUD).
// Vendor-neutral: plain Verilog, inferred block RAM, no vendor primitives. To use it on a board,
// connect clk / rst / uart_rx / uart_tx (and optionally the status LEDs) in your pin constraints,
// and set CLK_HZ, BAUD and the RAM sizes to fit the device.

`default_nettype none

module tinyml_uart_top #(
    parameter CLK_HZ = 100000000,       // board clock in Hz
    parameter BAUD   = 921600,          // UART baud rate (pass the same value to UartBus)
    parameter W_AW   = 12,              // weight RAM address width    (bytes = 2^W_AW, max 15)
    parameter X_AW   = 10,              // input RAM address width     (max 12)
    parameter Y_AW   = 8,               // output RAM address width    (max 12)
    parameter P_AW   = 10,              // parameter RAM address width (16 B per channel, max 12)
    parameter LANES  = 1,               // output channels computed at once: 1, 2, 4 or 8
    parameter RQ_MODE = 0,              // requantiser: 0 serial (smallest), 1 pipelined (fastest, 4 multipliers)
    parameter HAS_QUEUE = 0,            // job descriptor queue
    parameter D_AW    = 9,              // descriptor RAM address width (64 B per descriptor)
    parameter HAS_PERF = 0,             // performance counters
    parameter HAS_WIN  = 0,             // input window generator (no host-side im2col)
    parameter HAS_FUSE = 0              // layer outputs can be written back into X
) (
    input  wire       clk,              // board clock
    input  wire       rst,              // reset, active high (e.g. a push button)
    input  wire       uart_rx,          // UART from the PC
    output wire       uart_tx,          // UART to the PC
    output wire [2:0] status            // {done, error, busy} for LEDs
);

    // Bus address.
    wire [15:0] addr;
    // Bus write strobe.
    wire        wr;
    // Bus write data.
    wire [7:0]  wdata;
    // Bus read strobe.
    wire        rd;
    // Bus read data.
    wire [7:0]  rdata;
    // Engine status byte.
    wire [7:0]  st;

    // Power-on reset counter.
    reg  [7:0] por = 8'd0;
    // Reset input synchroniser.
    reg  [1:0] rst_sync = 2'b00;
    // Internal reset: power-on or the reset input.
    wire       rst_i = (por != 8'hFF) | rst_sync[1];

    // Power-on counter and reset synchroniser.
    always @(posedge clk) begin
        // Count to the end value, then stay.
        if (por != 8'hFF) por <= por + 8'd1;
        // Synchronise the asynchronous reset input.
        rst_sync <= {rst_sync[0], rst};
    end

    // Done, error and busy bits for LEDs.
    assign status = {st[1], st[2], st[0]};

    // This contains the instantiation for the UART bridge
    uart_bus #(.CLKS_PER_BIT((CLK_HZ + BAUD / 2) / BAUD), .TIMEOUT(CLK_HZ / 100)) u_bridge (
        .clk(clk), .rst(rst_i), .rx(uart_rx), .tx(uart_tx),
        .bus_addr(addr), .bus_wr(wr), .bus_wdata(wdata), .bus_rd(rd), .bus_rdata(rdata));

    // This contains the instantiation for the int8 engine
    tmf_engine #(.W_AW(W_AW), .X_AW(X_AW), .Y_AW(Y_AW), .P_AW(P_AW), .LANES(LANES), .RQ_MODE(RQ_MODE),
                .HAS_QUEUE(HAS_QUEUE), .D_AW(D_AW), .HAS_PERF(HAS_PERF), .HAS_WIN(HAS_WIN), .HAS_FUSE(HAS_FUSE)) u_engine (
        .clk(clk), .rst(rst_i),
        .bus_addr(addr), .bus_wr(wr), .bus_wdata(wdata), .bus_rd(rd), .bus_rdata(rdata),
        .status_o(st));

endmodule

`default_nettype wire
