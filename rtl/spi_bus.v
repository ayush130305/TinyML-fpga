// spi_bus.v - SPI-slave-to-byte-bus bridge (mode 0, MSB first): lets a microcontroller or a Linux board with a
// plain SPI master drive tmf_engine (or any byte-bus peripheral). Same bus contract as uart_bus.v.
// Frame (chip select low for the whole frame):
//   byte 0 : CMD   bit7 = 1 read / 0 write, bits[6:0] = n - 1 (informational: chip select high ends the frame)
//   byte 1 : address [15:8]       byte 2 : address [7:0]
//   write  : data bytes follow, one bus write each, the address auto-increments
//   read   : one dummy byte (the bridge fetches the first byte meanwhile), then the master clocks out the data
// Needs SCK no faster than clk / 6 (the clock edges are sampled by the fabric clock).

`default_nettype none

module spi_bus (
    input  wire        clk,                   // fabric clock
    input  wire        rst,                   // synchronous reset, active high
    input  wire        sck,                   // SPI clock from the master
    input  wire        cs_n,                  // chip select, active low
    input  wire        mosi,                  // master out, slave in
    output wire        miso,                  // master in, slave out (drive it only while cs_n is low)
    output wire        miso_oe,               // 1 while the bridge is selected
    output reg  [15:0] bus_addr,              // byte address
    output reg         bus_wr,                // write strobe
    output reg  [7:0]  bus_wdata,             // write data
    output reg         bus_rd,                // read strobe
    input  wire [7:0]  bus_rdata              // read data
);

    // Frame phases.
    localparam [2:0] P_CMD = 3'd0, P_AH = 3'd1, P_AL = 3'd2, P_WR = 3'd3, P_DUMMY = 3'd4, P_RD = 3'd5;

    // Synchronisers for the asynchronous pins (three stages of sck to also keep the previous value).
    reg [2:0] sck_q;
    reg [1:0] cs_q, mosi_q;
    // Synchronised clock edges and levels.
    wire rise = sck_q[1] & ~sck_q[2];
    wire fall = ~sck_q[1] & sck_q[2];
    wire sel  = ~cs_q[1];
    wire mosi_s = mosi_q[1];

    // Frame state.
    reg [2:0] phase;
    reg       is_rd;
    // Bit counter and shift registers.
    reg [2:0] bitcnt;
    reg [7:0] rxsh, txsh;
    // Next byte to send, applied on the falling edge after the byte ends (MISO must hold until then).
    reg [7:0] txn;
    reg       ldp;
    // The byte just completed.
    wire [7:0] rxb = {rxsh[6:0], mosi_s};
    // Byte-complete event: the 8th rising edge of a selected frame.
    wire byte_done = sel && rise && (bitcnt == 3'd7);

    // Read-ahead fetcher: fetch_go starts a bus read of bus_addr; fvalid says fdata holds the byte.
    reg       fetch_go;
    reg [2:0] fst;
    reg       fvalid;
    reg [7:0] fdata;

    // Output shift register drives MISO.
    assign miso    = txsh[7];
    assign miso_oe = sel;

    // Frame decoder and bus master.
    always @(posedge clk) begin
        // Strobes default low.
        bus_wr <= 1'b0;
        bus_rd <= 1'b0;
        // Synchronise the pins.
        sck_q  <= {sck_q[1:0], sck};
        cs_q   <= {cs_q[0], cs_n};
        mosi_q <= {mosi_q[0], mosi};
        // Address advances after each write strobe.
        if (bus_wr) bus_addr <= bus_addr + 16'd1;
        // Read-ahead fetcher: two clocks of address setup, then capture and move on.
        case (fst)
            3'd0: if (fetch_go) begin fst <= 3'd1; fetch_go <= 1'b0; fvalid <= 1'b0; end
            3'd1: fst <= 3'd2;
            3'd2: begin
                // Strobe and capture (the address has been stable for two clocks).
                bus_rd <= 1'b1;
                fdata  <= bus_rdata;
                fst    <= 3'd3;
            end
            3'd3: begin
                // Next address for the following fetch; the byte is ready.
                bus_addr <= bus_addr + 16'd1;
                fvalid   <= 1'b1;
                fst      <= 3'd0;
            end
            default: fst <= 3'd0;
        endcase
        // Chip select high: back to the start of a frame.
        if (rst || !sel) begin
            phase    <= P_CMD;
            bitcnt   <= 3'd0;
            txsh     <= 8'h00;
            ldp      <= 1'b0;
            fetch_go <= 1'b0;
            fvalid   <= 1'b0;
            if (rst) fst <= 3'd0;
        end else begin
            // Rising edge: sample MOSI.
            if (rise) begin
                rxsh   <= {rxsh[6:0], mosi_s};
                bitcnt <= bitcnt + 3'd1;
            end
            // Falling edge: load the pending byte, else present the next bit.
            if (fall) begin
                if (ldp) begin txsh <= txn; ldp <= 1'b0; end
                else if (bitcnt != 3'd0) txsh <= {txsh[6:0], 1'b0};
            end
            // A byte is complete.
            if (byte_done) begin
                case (phase)
                    // Command byte.
                    P_CMD: begin is_rd <= rxb[7]; phase <= P_AH; end
                    // Address high.
                    P_AH: begin bus_addr[15:8] <= rxb; phase <= P_AL; end
                    // Address low; reads start fetching their first byte.
                    P_AL: begin
                        bus_addr[7:0] <= rxb;
                        phase <= is_rd ? P_DUMMY : P_WR;
                        fetch_go <= is_rd;
                    end
                    // Write data: one bus write.
                    P_WR: begin bus_wdata <= rxb; bus_wr <= 1'b1; end
                    // The dummy byte ends: the first data byte goes to the shifter, the next fetch starts.
                    P_DUMMY: begin txn <= fdata; ldp <= 1'b1; phase <= P_RD; fetch_go <= 1'b1; end
                    // Data bytes: load the prefetched byte, fetch the next.
                    P_RD: begin txn <= fdata; ldp <= 1'b1; fetch_go <= 1'b1; end
                    default: phase <= P_CMD;
                endcase
            end
        end
    end

endmodule

`default_nettype wire
