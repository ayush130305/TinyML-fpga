// axil_bus.v - AXI4-Lite-slave-to-byte-bus bridge: for FPGAs where the engine sits behind a CPU (Zynq PS, a soft
// core, a Linux board with an FPGA bridge). Each 32-bit word at AXI address A holds the four bytes at byte-bus
// addresses A..A+3, little endian, so a CPU can use plain byte accesses (or byte-strobed words) on the engine's
// register map: base + bus_addr. Writes honour WSTRB (one bus write per enabled byte); reads return all four bytes.
// Always answers OKAY. Reads and writes are served one at a time (write first).

`default_nettype none

module axil_bus (
    input  wire        clk,                   // AXI clock
    input  wire        rst,                   // synchronous reset, active high
    input  wire [15:0] awaddr,                // write address
    input  wire        awvalid,
    output reg         awready,
    input  wire [31:0] wdata,                 // write data
    input  wire [3:0]  wstrb,                 // byte enables
    input  wire        wvalid,
    output reg         wready,
    output wire [1:0]  bresp,                 // always OKAY
    output reg         bvalid,
    input  wire        bready,
    input  wire [15:0] araddr,                // read address
    input  wire        arvalid,
    output reg         arready,
    output reg  [31:0] rdata,                 // read data
    output wire [1:0]  rresp,                 // always OKAY
    output reg         rvalid,
    input  wire        rready,
    output reg  [15:0] bus_addr,              // byte address
    output reg         bus_wr,                // write strobe
    output reg  [7:0]  bus_wdata,             // write data
    output reg         bus_rd,                // read strobe
    input  wire [7:0]  bus_rdata              // read data
);

    // States.
    localparam [2:0] S_IDLE = 3'd0, S_WR = 3'd1, S_WRESP = 3'd2, S_RSET = 3'd3, S_RSTB = 3'd4, S_RRESP = 3'd5;

    assign bresp = 2'b00;
    assign rresp = 2'b00;

    // State and byte index 0..3.
    reg [2:0]  st;
    reg [1:0]  k;
    // Hold counter for the read address setup.
    reg [1:0]  wt;
    // Latched word address, data and strobes.
    reg [15:0] a;
    reg [31:0] d;
    reg [3:0]  s;

    always @(posedge clk) begin
        // Strobes and one-clock handshakes default low.
        bus_wr  <= 1'b0;
        bus_rd  <= 1'b0;
        awready <= 1'b0;
        wready  <= 1'b0;
        arready <= 1'b0;
        if (rst) begin
            st <= S_IDLE; bvalid <= 1'b0; rvalid <= 1'b0;
        end else begin
            case (st)
                // Take a write (address and data together) or a read.
                S_IDLE: begin
                    k <= 2'd0;
                    if (awvalid && wvalid) begin
                        a <= {awaddr[15:2], 2'b00}; d <= wdata; s <= wstrb;
                        awready <= 1'b1; wready <= 1'b1; st <= S_WR;
                    end else if (arvalid) begin
                        a <= {araddr[15:2], 2'b00}; arready <= 1'b1; wt <= 2'd0; st <= S_RSET;
                        bus_addr <= {araddr[15:2], 2'b00};
                    end
                end
                // One enabled byte per clock.
                S_WR: begin
                    if (s[k]) begin
                        bus_addr <= {a[15:2], k}; bus_wdata <= d[8*k +: 8]; bus_wr <= 1'b1;
                    end
                    k <= k + 2'd1;
                    if (k == 2'd3) begin st <= S_WRESP; bvalid <= 1'b1; end
                end
                // Write response.
                S_WRESP: if (bvalid && bready) begin bvalid <= 1'b0; st <= S_IDLE; end
                // Hold the address two clocks.
                S_RSET: begin
                    wt <= wt + 2'd1;
                    if (wt == 2'd1) st <= S_RSTB;
                end
                // Strobe and capture byte k.
                S_RSTB: begin
                    bus_rd <= 1'b1;
                    rdata[8*k +: 8] <= bus_rdata;
                    k <= k + 2'd1;
                    wt <= 2'd0;
                    if (k == 2'd3) begin st <= S_RRESP; rvalid <= 1'b1; end
                    else begin bus_addr <= {a[15:2], k + 2'd1}; st <= S_RSET; end
                end
                // Read response.
                S_RRESP: if (rvalid && rready) begin rvalid <= 1'b0; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
