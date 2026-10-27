// tmf_ram.v - simple dual-port byte RAM with a registered read port.
// One write port and one read port on the same clock, written in the standard style that FPGA
// synthesis tools infer as block RAM. Read data appears one clock after the read address.

`default_nettype none

module tmf_ram #(
    parameter AW = 10                 // address width: depth is 2^AW bytes
) (
    input  wire          clk,         // clock for both ports
    input  wire          we,          // write enable
    input  wire [AW-1:0] waddr,       // write address
    input  wire [7:0]    wdata,       // write data
    input  wire [AW-1:0] raddr,       // read address
    output reg  [7:0]    q            // read data, registered (valid one clock after raddr)
);

    // Storage array.
    reg [7:0] mem [0:(1<<AW)-1];

    // RAM process: synchronous write and synchronous read.
    always @(posedge clk) begin
        // Store the write byte when enabled.
        if (we) mem[waddr] <= wdata;
        // Register the byte at the read address.
        q <= mem[raddr];
    end

endmodule

`default_nettype wire
