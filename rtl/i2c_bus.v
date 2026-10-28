// i2c_bus.v - I2C-slave-to-byte-bus bridge (7-bit address, standard / fast mode, no clock stretching).
// Same register pointer model as most I2C chips:
//   write : START  addr+W  addr_hi  addr_lo  data...  STOP        data bytes are bus writes, address auto-increments
//   read  : START  addr+W  addr_hi  addr_lo  RESTART  addr+R  data...(master NACKs the last one)  STOP
//           after a read the pointer keeps counting, so a later plain read continues where it stopped
// Every byte the slave receives is acknowledged; a wrong slave address is not acknowledged.
// Needs SCL no faster than clk / 16 (the pins are sampled by the fabric clock). sda_low = 1 pulls SDA low.

`default_nettype none

module i2c_bus #(
    parameter [6:0] I2C_ADDR = 7'h2A           // 7-bit slave address
) (
    input  wire        clk,                   // fabric clock
    input  wire        rst,                   // synchronous reset, active high
    input  wire        scl,                   // I2C clock
    input  wire        sda_in,                // SDA as seen on the pin
    output reg         sda_low,               // 1 = pull SDA low (open drain)
    output reg  [15:0] bus_addr,              // byte address
    output reg         bus_wr,                // write strobe
    output reg  [7:0]  bus_wdata,             // write data
    output reg         bus_rd,                // read strobe
    input  wire [7:0]  bus_rdata              // read data
);

    // Transfer phases.
    localparam [2:0] P_IDLE = 3'd0, P_ADDR = 3'd1, P_AH = 3'd2, P_AL = 3'd3, P_WR = 3'd4, P_TX = 3'd5;

    // Pin synchronisers (the third stage gives the previous value).
    reg [2:0] scl_q, sda_q;
    // Edges and levels, in the fabric clock domain.
    wire scl_s  = scl_q[1];
    wire rise   = scl_q[1] & ~scl_q[2];
    wire fall   = ~scl_q[1] & scl_q[2];
    wire sda_s  = sda_q[1];
    wire start  = scl_s & sda_q[2] & ~sda_q[1];
    wire stop   = scl_s & ~sda_q[2] & sda_q[1];

    // Phase, bit counter (0..9: 8 data bits then the acknowledge bit), shift registers.
    reg [2:0] phase;
    reg [3:0] bitcnt;
    reg [7:0] rxsh, txb;
    // The byte just received.
    wire [7:0] rxb = {rxsh[6:0], sda_s};
    // The slave acknowledges the byte just received.
    reg       ack_pend;
    // Master NACK seen after a transmitted byte, and "a byte has been sent in this read".
    reg       m_nack, txsent;

    // Read-ahead fetcher (same sequence as spi_bus.v).
    reg       fetch_go;
    reg [1:0] fst;
    reg [7:0] fdata;

    // Transfer decoder and bus master.
    always @(posedge clk) begin
        // Strobes default low.
        bus_wr <= 1'b0;
        bus_rd <= 1'b0;
        // Synchronise the pins.
        scl_q <= {scl_q[1:0], scl};
        sda_q <= {sda_q[1:0], sda_in};
        // The address moves on after every write strobe.
        if (bus_wr) bus_addr <= bus_addr + 16'd1;
        // Fetcher: hold the address two clocks, strobe and capture, advance.
        case (fst)
            2'd0: if (fetch_go) begin fst <= 2'd1; fetch_go <= 1'b0; end
            2'd1: fst <= 2'd2;
            2'd2: begin bus_rd <= 1'b1; fdata <= bus_rdata; fst <= 2'd3; end
            default: begin bus_addr <= bus_addr + 16'd1; fst <= 2'd0; end
        endcase
        if (rst) begin
            phase <= P_IDLE; bitcnt <= 4'd0; sda_low <= 1'b0; fst <= 2'd0; fetch_go <= 1'b0;
            ack_pend <= 1'b0; m_nack <= 1'b0; txsent <= 1'b0;
        end else if (start) begin
            // START or repeated START: a new address byte follows.
            phase <= P_ADDR; bitcnt <= 4'd0; sda_low <= 1'b0; ack_pend <= 1'b0;
        end else if (stop) begin
            // STOP: idle.
            phase <= P_IDLE; bitcnt <= 4'd0; sda_low <= 1'b0;
        end else if (phase != P_IDLE) begin
            // Rising SCL: the master's data bit (or acknowledge) is valid.
            if (rise) begin
                bitcnt <= bitcnt + 4'd1;
                if (bitcnt < 4'd8) begin
                    // Receiving phases shift the bit in.
                    if (phase != P_TX) rxsh <= {rxsh[6:0], sda_s};
                    // The eighth bit completes a received byte.
                    if (bitcnt == 4'd7 && phase != P_TX) begin
                        case (phase)
                            // Address byte: ours? then write (pointer) or read (data out).
                            P_ADDR: begin
                                ack_pend <= (rxb[7:1] == I2C_ADDR);
                                if (rxb[7:1] != I2C_ADDR) phase <= P_IDLE;
                                else if (rxb[0]) begin
                                    // Read: fetch the byte at the pointer now, it goes out after the acknowledge.
                                    phase <= P_TX; txsent <= 1'b0; m_nack <= 1'b0; fetch_go <= 1'b1;
                                end else phase <= P_AH;
                            end
                            // Pointer high byte.
                            P_AH: begin bus_addr[15:8] <= rxb; ack_pend <= 1'b1; phase <= P_AL; end
                            // Pointer low byte.
                            P_AL: begin bus_addr[7:0] <= rxb; ack_pend <= 1'b1; phase <= P_WR; end
                            // Data byte: one bus write.
                            P_WR: begin bus_wdata <= rxb; bus_wr <= 1'b1; ack_pend <= 1'b1; end
                            default: ;
                        endcase
                    end
                end else if (bitcnt == 4'd8 && phase == P_TX && txsent) begin
                    // Ninth clock of a transmitted byte: the master's acknowledge (1 = NACK).
                    m_nack <= sda_s;
                end
            end
            // Falling SCL: drive the next bit.
            if (fall) begin
                if (bitcnt == 4'd8) begin
                    // Start of the ninth clock: acknowledge a received byte, or let the master answer.
                    sda_low <= (phase != P_TX) && ack_pend;
                end else if (bitcnt == 4'd9) begin
                    // End of the ninth clock.
                    bitcnt <= 4'd0;
                    if (phase == P_TX) begin
                        if (txsent && m_nack) begin
                            // The master has all it wants.
                            phase <= P_IDLE; sda_low <= 1'b0;
                        end else begin
                            // Next byte out: bit 7 now, the fetch for the byte after it starts.
                            txb <= fdata; sda_low <= ~fdata[7]; txsent <= 1'b1; fetch_go <= 1'b1;
                        end
                    end else sda_low <= 1'b0;
                end else if (phase == P_TX && bitcnt != 4'd0) begin
                    // Data bit 6 .. 0.
                    sda_low <= ~txb[3'd7 - bitcnt[2:0]];
                end
            end
        end
    end

endmodule

`default_nettype wire
