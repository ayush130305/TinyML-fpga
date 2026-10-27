// tmf_engine.v - int8 dot-product engine for tinyml_fpga (conv / depthwise / fully-connected layers).
//
// For every input vector v (0..NVEC-1) and output channel o (0..N-1):
//   acc = bias[o] + sum_k (X[v*XSTRIDE + (DW ? o*K : 0) + k] + IN_OFF) * W[WBASE + o*K + k]
//   Y[v*N + o] = requant(acc)           (MODE.RAW: Y[(v*N+o)*4 .. +3] = acc, little endian)
// One MAC per clock (3-stage pipeline), exact TFLite requantisation in tmf_requant.v.
//
// It sits on a simple byte bus (driven by uart_bus.v, or any bridge of your own):
//   bus_addr stable at least one clock before bus_rd; bus_rdata is valid in the bus_rd clock;
//   bus_wr is a one-clock strobe with bus_addr/bus_wdata. Register map: see tinyml_fpga/engine.py.
//   0x100-0x111 config (RW), 0x11F CTRL (W), 0x120 STATUS, 0x121-0x124 CYCLES, 0x130-0x135 ID (RO)
//   0x1000 X, 0x2000 Y (RO from the bus), 0x3000 P (16 B per channel), 0x8000 W
// Bus writes to config and memories are ignored while the engine is busy.

`default_nettype none

module tmf_engine #(
    parameter W_AW = 12,                 // weight RAM address width   (bytes = 2^W_AW, max 15)
    parameter X_AW = 10,                 // input RAM address width    (max 12)
    parameter Y_AW = 8,                  // output RAM address width   (max 12)
    parameter P_AW = 10                  // parameter RAM address width (16 B per channel, max 12)
) (
    input  wire        clk,              // clock
    input  wire        rst,              // synchronous reset, active high
    input  wire [15:0] bus_addr,         // byte address
    input  wire        bus_wr,           // write strobe
    input  wire [7:0]  bus_wdata,        // write data
    input  wire        bus_rd,           // read strobe (reads have no side effects, kept for the bus contract)
    output reg  [7:0]  bus_rdata,        // read data
    output wire [7:0]  status_o          // STATUS byte (bit0 busy, bit1 done, bit2 error) for LEDs or a status register
);

    // ---------------------------------------------------------------- constants
    // Memory sizes in bytes.
    localparam [32:0] WSIZE = 33'd1 << W_AW;
    // Input RAM size.
    localparam [32:0] XSIZE = 33'd1 << X_AW;
    // Output RAM size.
    localparam [32:0] YSIZE = 33'd1 << Y_AW;
    // Parameter RAM size.
    localparam [32:0] PSIZE = 33'd1 << P_AW;

    // Sequencer states.
    localparam [3:0] S_IDLE  = 4'd0;     // waiting for CTRL.start
    localparam [3:0] S_CHK   = 4'd1;     // bounds check for the next output
    localparam [3:0] S_PAR   = 4'd2;     // fetch the 9 parameter bytes
    localparam [3:0] S_INIT  = 4'd3;     // load bias / multiplier / shift
    localparam [3:0] S_DOT   = 4'd4;     // issue K multiply-accumulates
    localparam [3:0] S_DRAIN = 4'd5;     // let the MAC pipeline empty
    localparam [3:0] S_RQ    = 4'd6;     // wait for the requantiser
    localparam [3:0] S_RAW   = 4'd7;     // write 4 raw accumulator bytes
    localparam [3:0] S_NEXT  = 4'd8;     // advance to the next output

    // ---------------------------------------------------------------- bus decode
    // Address is in the config/control register block.
    wire sel_reg = (bus_addr[15:8] == 8'h01);
    // Address is in the X window.
    wire sel_x   = (bus_addr[15:12] == 4'h1);
    // Address is in the Y window.
    wire sel_y   = (bus_addr[15:12] == 4'h2);
    // Address is in the P window.
    wire sel_p   = (bus_addr[15:12] == 4'h3);
    // Address is in the W window.
    wire sel_w   = bus_addr[15];

    // ---------------------------------------------------------------- configuration registers
    // Raw config bytes 0x100..0x111.
    reg  [7:0] cfg [0:17];
    // Dot-product length.
    wire [15:0] cK       = {cfg[1],  cfg[0]};
    // Output channels.
    wire [15:0] cN       = {cfg[3],  cfg[2]};
    // Vectors per job.
    wire [15:0] cNVEC    = {cfg[5],  cfg[4]};
    // Bytes between vectors.
    wire [15:0] cXSTRIDE = {cfg[7],  cfg[6]};
    // Weight base offset.
    wire [15:0] cWBASE   = {cfg[9],  cfg[8]};
    // Parameter base channel.
    wire [15:0] cPBASE   = {cfg[11], cfg[10]};
    // Input offset (9 significant bits).
    wire signed [15:0] cINOFF = {cfg[13], cfg[12]};
    // Output zero point.
    wire signed [7:0] cZP    = cfg[14];
    // Clamp low.
    wire signed [7:0] cAMIN  = cfg[15];
    // Clamp high.
    wire signed [7:0] cAMAX  = cfg[16];
    // Depthwise mode bit.
    wire cDW  = cfg[17][0];
    // Raw int32 output mode bit.
    wire cRAW = cfg[17][1];

    // ---------------------------------------------------------------- status
    // Engine running.
    reg busy;
    // Last job finished.
    reg done;
    // Last job failed (empty job or memory overrun).
    reg err;
    // Clock counter of the current / last job.
    reg [31:0] cycles;
    // STATUS byte.
    assign status_o = {5'd0, err, done, busy};

    // ---------------------------------------------------------------- sequencer registers
    // Current state.
    reg [3:0]  st;
    // Vector index.
    reg [15:0] vi;
    // Output channel index.
    reg [15:0] oi;
    // MAC issue counter.
    reg [15:0] ki;
    // Parameter byte counter.
    reg [3:0]  pc;
    // Raw byte counter.
    reg [1:0]  rb;
    // Start of the current vector in X.
    reg [32:0] xvec;
    // Start of the current channel's X slice.
    reg [32:0] xrow;
    // Start of the current channel's weight row.
    reg [32:0] wrow;
    // X read pointer during the dot product.
    reg [32:0] xp;
    // W read pointer during the dot product.
    reg [32:0] wp;
    // Byte address of the current parameter record.
    reg [32:0] pp;
    // Y write pointer.
    reg [32:0] yp;
    // Parameter bytes, shifted in LSB first: bias[31:0], mult[63:32], shift[71:64].
    reg [71:0] par;
    // Accumulator.
    reg signed [31:0] acc;
    // Pipeline stage 1 valid (RAM data available).
    reg v1;
    // Pipeline stage 2 valid (product available).
    reg v2;
    // Pipeline product register.
    reg signed [17:0] prod;
    // Input byte plus the input offset (-255 .. 255).
    wire signed [8:0]  xoff = $signed(x_q) + $signed(cINOFF[8:0]);
    // Signed 9 x 8 product.
    wire signed [16:0] mul  = xoff * $signed(w_q);

    // ---------------------------------------------------------------- RAMs
    // RAM read data.
    wire [7:0] x_q, w_q, p_q, y_q;
    // Y write enable from the sequencer.
    reg        y_we;
    // Y write address.
    reg [Y_AW-1:0] y_wa;
    // Y write data.
    reg [7:0]  y_wd;
    // Bus write is allowed (engine idle).
    wire bus_wr_ok = bus_wr & ~busy;

    // This contains the instantiation for the X RAM (input vectors)
    tmf_ram #(.AW(X_AW)) u_x (.clk(clk), .we(bus_wr_ok & sel_x), .waddr(bus_addr[X_AW-1:0]), .wdata(bus_wdata),
                               .raddr(busy ? xp[X_AW-1:0] : bus_addr[X_AW-1:0]), .q(x_q));
    // This contains the instantiation for the W RAM (weights)
    tmf_ram #(.AW(W_AW)) u_w (.clk(clk), .we(bus_wr_ok & sel_w), .waddr(bus_addr[W_AW-1:0]), .wdata(bus_wdata),
                               .raddr(busy ? wp[W_AW-1:0] : bus_addr[W_AW-1:0]), .q(w_q));
    // This contains the instantiation for the P RAM (per-channel parameters)
    tmf_ram #(.AW(P_AW)) u_p (.clk(clk), .we(bus_wr_ok & sel_p), .waddr(bus_addr[P_AW-1:0]), .wdata(bus_wdata),
                               .raddr(busy ? (pp[P_AW-1:0] + {{(P_AW-4){1'b0}}, pc}) : bus_addr[P_AW-1:0]), .q(p_q));
    // This contains the instantiation for the Y RAM (outputs)
    tmf_ram #(.AW(Y_AW)) u_y (.clk(clk), .we(y_we), .waddr(y_wa), .wdata(y_wd),
                               .raddr(bus_addr[Y_AW-1:0]), .q(y_q));

    // ---------------------------------------------------------------- requantiser
    // Start pulse to the requantiser.
    reg        rq_start;
    // Requantiser finished.
    wire       rq_done;
    // Requantiser result.
    wire [7:0] rq_y;

    // This contains the instantiation for u_rq
    tmf_requant u_rq (.clk(clk), .rst(rst), .start(rq_start), .acc(acc), .mult(par[63:32]), .shift(par[71:64]),
                      .zp(cZP), .act_min(cAMIN), .act_max(cAMAX), .done(rq_done), .y(rq_y));

    // ---------------------------------------------------------------- bounds of the next output
    // Dot length, zero-extended to the pointer width.
    wire [32:0] k33  = {17'd0, cK};
    // Vector stride, zero-extended to the pointer width.
    wire [32:0] xs33 = {17'd0, cXSTRIDE};
    // Bytes per output.
    wire [32:0] ob     = cRAW ? 33'd4 : 33'd1;
    // The next output fits in every memory.
    wire        fits   = (xrow + k33 <= XSIZE) && (wrow + k33 <= WSIZE) && (pp + 33'd16 <= PSIZE) && (yp + ob <= YSIZE);
    // Last channel of a vector.
    wire        last_o = (oi == cN - 16'd1);
    // Last vector of the job.
    wire        last_v = (vi == cNVEC - 16'd1);

    // ---------------------------------------------------------------- bus read mux
    // Read data: registers combinationally, memories from their registered outputs.
    always @* begin
        // Default for unmapped addresses.
        bus_rdata = 8'h00;
        // Config read-back.
        if (sel_reg && bus_addr[7:0] < 8'h12) bus_rdata = cfg[bus_addr[4:0]];
        // STATUS.
        if (sel_reg && bus_addr[7:0] == 8'h20) bus_rdata = status_o;
        // CYCLES byte 0.
        if (sel_reg && bus_addr[7:0] == 8'h21) bus_rdata = cycles[7:0];
        // CYCLES byte 1.
        if (sel_reg && bus_addr[7:0] == 8'h22) bus_rdata = cycles[15:8];
        // CYCLES byte 2.
        if (sel_reg && bus_addr[7:0] == 8'h23) bus_rdata = cycles[23:16];
        // CYCLES byte 3.
        if (sel_reg && bus_addr[7:0] == 8'h24) bus_rdata = cycles[31:24];
        // MAGIC.
        if (sel_reg && bus_addr[7:0] == 8'h30) bus_rdata = 8'h4D;
        // VERSION.
        if (sel_reg && bus_addr[7:0] == 8'h31) bus_rdata = 8'h01;
        // W address width.
        if (sel_reg && bus_addr[7:0] == 8'h32) bus_rdata = W_AW;
        // X address width.
        if (sel_reg && bus_addr[7:0] == 8'h33) bus_rdata = X_AW;
        // Y address width.
        if (sel_reg && bus_addr[7:0] == 8'h34) bus_rdata = Y_AW;
        // P address width.
        if (sel_reg && bus_addr[7:0] == 8'h35) bus_rdata = P_AW;
        // X window.
        if (sel_x) bus_rdata = x_q;
        // Y window.
        if (sel_y) bus_rdata = y_q;
        // P window.
        if (sel_p) bus_rdata = p_q;
        // W window.
        if (sel_w) bus_rdata = w_q;
    end

    // ---------------------------------------------------------------- config writes
    // Loop index for the register reset.
    integer i;
    // Config register file: bus writes when idle.
    always @(posedge clk) begin
        // Reset clears every config byte.
        if (rst) begin
            // Clear all 18 bytes.
            for (i = 0; i < 18; i = i + 1) cfg[i] <= 8'h00;
        end else if (bus_wr_ok && sel_reg && bus_addr[7:0] < 8'h12) begin
            // Store the addressed byte.
            cfg[bus_addr[4:0]] <= bus_wdata;
        end
    end

    // ---------------------------------------------------------------- MAC pipeline
    // Stage 2 and 3 of the multiply-accumulate; stage 1 is the RAM read itself.
    always @(posedge clk) begin
        // Stage 2: (x + in_off) * w, 9-bit x 8-bit signed.
        if (v1) prod <= {mul[16], mul};
        // Stage 2 valid follows stage 1.
        v2 <= v1;
        // Stage 1 valid: an address was issued this clock.
        v1 <= (st == S_DOT);
    end

    // ---------------------------------------------------------------- sequencer
    // Job control: start, per-output loop, bounds checks, writes, status.
    always @(posedge clk) begin
        // Y write strobe defaults low.
        y_we     <= 1'b0;
        // Requantiser start defaults low.
        rq_start <= 1'b0;
        // Count clocks while busy.
        if (busy) cycles <= cycles + 32'd1;
        // Stage 3 of the MAC pipeline: accumulate the product.
        if (v2) acc <= acc + {{14{prod[17]}}, prod};
        // Reset to idle with clear status.
        if (rst) begin
            // Idle state.
            st   <= S_IDLE;
            // Not busy.
            busy <= 1'b0;
            // No result.
            done <= 1'b0;
            // No error.
            err  <= 1'b0;
        end else begin
            // A start while a job is running is a driver bug: flag it (the running job continues).
            if (bus_wr && sel_reg && bus_addr[7:0] == 8'h1F && bus_wdata[0] && busy) err <= 1'b1;
            // Clear-status command from the bus (only meaningful when idle).
            if (bus_wr && sel_reg && bus_addr[7:0] == 8'h1F && bus_wdata[7] && !busy) begin
                // Clear done.
                done <= 1'b0;
                // Clear error.
                err  <= 1'b0;
            end
            // State dispatch.
            case (st)
                // Wait for CTRL.start.
                S_IDLE: if (bus_wr && sel_reg && bus_addr[7:0] == 8'h1F && bus_wdata[0]) begin
                    // Empty job: error at once.
                    if (cK == 16'd0 || cN == 16'd0 || cNVEC == 16'd0) begin
                        // Flag the error.
                        err  <= 1'b1;
                        // And finish.
                        done <= 1'b1;
                    end else begin
                        // Running.
                        busy   <= 1'b1;
                        // New job: clear status.
                        done   <= 1'b0;
                        // No error yet.
                        err    <= 1'b0;
                        // Restart the clock counter.
                        cycles <= 32'd0;
                        // First vector.
                        vi     <= 16'd0;
                        // First channel.
                        oi     <= 16'd0;
                        // First vector starts at X[0].
                        xvec   <= 33'd0;
                        // First channel's X slice.
                        xrow   <= 33'd0;
                        // First weight row.
                        wrow   <= {17'd0, cWBASE};
                        // First parameter record (16 bytes per channel).
                        pp     <= {13'd0, cPBASE, 4'd0};
                        // First output byte.
                        yp     <= 33'd0;
                        // Check bounds first.
                        st     <= S_CHK;
                    end
                end
                // Every memory access of this output must be in range.
                S_CHK: begin
                    // Out of range: stop with an error.
                    if (!fits) begin
                        // Flag.
                        err  <= 1'b1;
                        // Finished.
                        done <= 1'b1;
                        // Not running.
                        busy <= 1'b0;
                        // Back to idle.
                        st   <= S_IDLE;
                    end else begin
                        // Fetch the parameters.
                        pc <= 4'd0;
                        // Next state.
                        st <= S_PAR;
                    end
                end
                // Read 9 parameter bytes; the byte addressed in clock n arrives in clock n+1.
                S_PAR: begin
                    // Shift in the byte addressed one clock earlier.
                    if (pc != 4'd0) par <= {p_q, par[71:8]};
                    // Next byte.
                    pc <= pc + 4'd1;
                    // After the 9th byte is captured, load the operands.
                    if (pc == 4'd9) st <= S_INIT;
                end
                // Load the operands and start the dot product.
                S_INIT: begin
                    // Accumulator starts at the bias.
                    acc <= par[31:0];
                    // X read pointer.
                    xp  <= xrow;
                    // W read pointer.
                    wp  <= wrow;
                    // MAC counter.
                    ki  <= 16'd0;
                    // Go.
                    st  <= S_DOT;
                end
                // One RAM read pair per clock; the pipeline adds the products.
                S_DOT: begin
                    // Next X byte.
                    xp <= xp + 33'd1;
                    // Next W byte.
                    wp <= wp + 33'd1;
                    // Count the issue.
                    ki <= ki + 16'd1;
                    // Last issue: wait for the pipeline.
                    if (ki == cK - 16'd1) st <= S_DRAIN;
                end
                // Pipeline empties two clocks after the last issue.
                S_DRAIN: if (!v1 && !v2) begin
                    // Raw mode: write the accumulator bytes.
                    if (cRAW) begin
                        // Byte 0 first.
                        rb <= 2'd0;
                        // Raw write state.
                        st <= S_RAW;
                    end else begin
                        // Start the requantiser.
                        rq_start <= 1'b1;
                        // Wait for it.
                        st <= S_RQ;
                    end
                end
                // Requantiser running.
                S_RQ: if (rq_done) begin
                    // Write the int8 result.
                    y_we <= 1'b1;
                    // At the current output position.
                    y_wa <= yp[Y_AW-1:0];
                    // Result byte.
                    y_wd <= rq_y;
                    // Advance the output pointer.
                    yp   <= yp + 33'd1;
                    // Next output.
                    st   <= S_NEXT;
                end
                // Four little-endian accumulator bytes.
                S_RAW: begin
                    // Write strobe.
                    y_we <= 1'b1;
                    // Current output byte.
                    y_wa <= yp[Y_AW-1:0];
                    // Byte rb of the accumulator.
                    y_wd <= acc[8*rb +: 8];
                    // Advance the output pointer.
                    yp   <= yp + 33'd1;
                    // Next byte.
                    rb   <= rb + 2'd1;
                    // After byte 3, next output.
                    if (rb == 2'd3) st <= S_NEXT;
                end
                // Step to the next channel / vector, or finish.
                S_NEXT: begin
                    // Last channel of this vector.
                    if (last_o) begin
                        // Last vector too: the job is complete.
                        if (last_v) begin
                            // Done.
                            done <= 1'b1;
                            // Not running.
                            busy <= 1'b0;
                            // Idle.
                            st   <= S_IDLE;
                        end else begin
                            // Next vector.
                            vi   <= vi + 16'd1;
                            // First channel again.
                            oi   <= 16'd0;
                            // Next vector start.
                            xvec <= xvec + xs33;
                            // Its first slice.
                            xrow <= xvec + xs33;
                            // Back to the first weight row.
                            wrow <= {17'd0, cWBASE};
                            // Back to the first parameter record.
                            pp   <= {13'd0, cPBASE, 4'd0};
                            // Check the new output.
                            st   <= S_CHK;
                        end
                    end else begin
                        // Next channel.
                        oi   <= oi + 16'd1;
                        // Next weight row.
                        wrow <= wrow + k33;
                        // Next parameter record.
                        pp   <= pp + 33'd16;
                        // Depthwise channels use their own X slice.
                        if (cDW) xrow <= xrow + k33;
                        // Check the new output.
                        st   <= S_CHK;
                    end
                end
                // Unused encodings recover to idle.
                default: st <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
