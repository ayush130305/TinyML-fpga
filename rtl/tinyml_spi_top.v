// tinyml_spi_top.v - ready-to-use top: SPI-slave bridge (mode 0) + int8 engine, for any FPGA board.
// Vendor-neutral plain Verilog. Same register map and frame rules as tinyml_uart_top.v.

`default_nettype none

module tinyml_spi_top #(
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
    parameter HAS_FUSE = 0,             // layer outputs can be written back into X
    parameter HAS_W4   = 0,             // 4-bit weights
    parameter HAS_CB   = 0              // codebook-compressed weights
) (
    input  wire       clk,              // board clock (SCK must be at most clk / 6)
    input  wire       rst,              // reset, active high (e.g. a push button)
    input  wire       spi_sck,          // SPI clock from the master (mode 0)
    input  wire       spi_cs_n,         // SPI chip select, active low
    input  wire       spi_mosi,         // SPI data from the master
    output wire       spi_miso,         // SPI data to the master (high impedance while not selected)
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

    // MISO output enable from the bridge.
    wire        miso_oe;
    // MISO value from the bridge.
    wire        miso_v;

    // MISO is released while the chip select is high.
    assign spi_miso = miso_oe ? miso_v : 1'bz;

    // This contains the instantiation for the SPI bridge
    spi_bus u_bridge (
        .clk(clk), .rst(rst_i), .sck(spi_sck), .cs_n(spi_cs_n), .mosi(spi_mosi), .miso(miso_v), .miso_oe(miso_oe),
        .bus_addr(addr), .bus_wr(wr), .bus_wdata(wdata), .bus_rd(rd), .bus_rdata(rdata));

    // This contains the instantiation for the int8 engine
    tmf_engine #(.W_AW(W_AW), .X_AW(X_AW), .Y_AW(Y_AW), .P_AW(P_AW), .LANES(LANES), .RQ_MODE(RQ_MODE),
                .HAS_QUEUE(HAS_QUEUE), .D_AW(D_AW), .HAS_PERF(HAS_PERF), .HAS_WIN(HAS_WIN), .HAS_FUSE(HAS_FUSE), .HAS_W4(HAS_W4), .HAS_CB(HAS_CB)) u_engine (
        .clk(clk), .rst(rst_i),
        .bus_addr(addr), .bus_wr(wr), .bus_wdata(wdata), .bus_rd(rd), .bus_rdata(rdata),
        .status_o(st));

endmodule

`default_nettype wire
