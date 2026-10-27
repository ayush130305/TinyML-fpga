// uart_bus.v - UART-to-byte-bus bridge: lets a PC drive tmf_engine (or any byte-bus peripheral).
// Frame format:
//   byte 0 : CMD   bit7 = 1 read / 0 write, bits[6:0] = number of bytes - 1
//   byte 1 : address [15:8]       byte 2 : address [7:0]
//   write  : n data bytes follow; after the last one is committed the bridge sends 0x55
//   read   : the bridge sends n data bytes
// The address auto-increments. Bus timing (see tmf_engine.v): the address is held for
// two clocks before the one-clock read strobe, and read data is sampled in the strobe clock.
// A frame that stalls for TIMEOUT clocks is dropped, so a confused host can always resynchronise.

`default_nettype none

module uart_bus #(
    parameter CLKS_PER_BIT = 868,             // clock / baud
    parameter TIMEOUT      = 1000000          // clocks of silence that abort a frame
) (
    input  wire        clk,                   // clock
    input  wire        rst,                   // synchronous reset, active high
    input  wire        rx,                    // UART input from the host
    output wire        tx,                    // UART output to the host
    output reg  [15:0] bus_addr,              // byte address
    output reg         bus_wr,                // write strobe
    output reg  [7:0]  bus_wdata,             // write data
    output reg         bus_rd,                // read strobe
    input  wire [7:0]  bus_rdata              // read data
);

    // Bridge states.
    localparam [3:0] B_CMD = 4'd0, B_AH = 4'd1, B_AL = 4'd2, B_WD = 4'd3, B_WACK = 4'd4,
                     B_RSET = 4'd5, B_RSTB = 4'd6, B_RSEND = 4'd7, B_RWAIT = 4'd8;
    // Width of the timeout counter.
    localparam TW = $clog2(TIMEOUT + 1);

    // Received byte.
    wire [7:0] rx_data;
    // Received byte strobe.
    wire       rx_valid;
    // Transmitter busy.
    wire       tx_busy;
    // Byte to send.
    reg  [7:0] tx_data;
    // Send strobe.
    reg        tx_send;
    // State.
    reg  [3:0] st;
    // Frame is a read.
    reg        is_rd;
    // Bytes left after the current one.
    reg  [6:0] left;
    // Wait counter for the read address setup.
    reg  [1:0] wt;
    // Silence counter for the frame timeout.
    reg  [TW-1:0] idle;

    // This contains the instantiation for u_rx
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (.clk(clk), .rst(rst), .rx(rx), .data(rx_data), .valid(rx_valid));
    // This contains the instantiation for u_tx
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (.clk(clk), .rst(rst), .data(tx_data), .send(tx_send), .tx(tx), .busy(tx_busy));

    // Frame decoder and bus master.
    always @(posedge clk) begin
        // Write strobe defaults low.
        bus_wr  <= 1'b0;
        // Read strobe defaults low.
        bus_rd  <= 1'b0;
        // Send strobe defaults low.
        tx_send <= 1'b0;
        // Silence counter: cleared by traffic, counts while a frame is open.
        idle    <= (rx_valid || st == B_CMD) ? {TW{1'b0}} : idle + 1'b1;
        // Reset, or a stalled frame: back to waiting for a command.
        if (rst || idle == TIMEOUT) begin
            // Wait for a command byte.
            st <= B_CMD;
        end else begin
            // State dispatch.
            case (st)
                // Command byte.
                B_CMD: if (rx_valid) begin
                    // Direction.
                    is_rd <= rx_data[7];
                    // Count minus one.
                    left  <= rx_data[6:0];
                    // Address high next.
                    st    <= B_AH;
                end
                // Address high byte.
                B_AH: if (rx_valid) begin
                    // Store it.
                    bus_addr[15:8] <= rx_data;
                    // Address low next.
                    st <= B_AL;
                end
                // Address low byte.
                B_AL: if (rx_valid) begin
                    // Store it.
                    bus_addr[7:0] <= rx_data;
                    // Read: set up the first address; write: receive data.
                    st <= is_rd ? B_RSET : B_WD;
                    // Address setup counter.
                    wt <= 2'd0;
                end
                // Each received data byte is one bus write.
                B_WD: if (rx_valid) begin
                    // Data.
                    bus_wdata <= rx_data;
                    // Strobe next clock edge (address and data are registered together).
                    bus_wr    <= 1'b1;
                    // Last byte: acknowledge, else advance.
                    if (left == 7'd0) st <= B_WACK;
                    else left <= left - 7'd1;
                end else if (bus_wr) begin
                    // Advance the address after the strobe.
                    bus_addr <= bus_addr + 16'd1;
                end
                // Send the write acknowledge once the last strobe has happened.
                B_WACK: if (!bus_wr && !tx_busy) begin
                    // Acknowledge byte.
                    tx_data <= 8'h55;
                    // Send it.
                    tx_send <= 1'b1;
                    // Frame complete.
                    st      <= B_CMD;
                end
                // Hold the address two clocks (registered RAM outputs settle).
                B_RSET: begin
                    // Count.
                    wt <= wt + 2'd1;
                    // Then strobe.
                    if (wt == 2'd1) st <= B_RSTB;
                end
                // Read strobe: data is sampled in this clock.
                B_RSTB: begin
                    // Strobe.
                    bus_rd  <= 1'b1;
                    // Capture.
                    tx_data <= bus_rdata;
                    // Send it.
                    st      <= B_RSEND;
                end
                // Start sending when the transmitter is free.
                B_RSEND: if (!tx_busy) begin
                    // Send strobe.
                    tx_send <= 1'b1;
                    // Wait for it to be taken.
                    st      <= B_RWAIT;
                end
                // Transmitter accepted the byte (busy rises next clock): next byte or done.
                B_RWAIT: if (tx_busy) begin
                    // Last byte sent.
                    if (left == 7'd0) st <= B_CMD;
                    else begin
                        // One fewer.
                        left     <= left - 7'd1;
                        // Next address.
                        bus_addr <= bus_addr + 16'd1;
                        // Setup counter.
                        wt       <= 2'd0;
                        // Set up the next read.
                        st       <= B_RSET;
                    end
                end
                // Unused encodings recover.
                default: st <= B_CMD;
            endcase
        end
    end

endmodule

`default_nettype wire
