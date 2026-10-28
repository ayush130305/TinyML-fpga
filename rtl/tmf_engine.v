// tmf_engine.v - int8 dot-product engine for tinyml_fpga (conv / depthwise / fully-connected layers).
//
// For every input vector v (0..NVEC-1) and output channel o (0..N-1):
//   acc = bias[o] + sum_k (X[v*XSTRIDE + (DW ? o*K : 0) + k] + IN_OFF) * W[row(o) + k]
//   Y[v*N + o] = requant(acc)           (MODE.RAW: Y[(v*N+o)*4 .. +3] = acc, little endian)
// LANES output channels are computed at once (they share the input stream and each reads its own weight
// RAM bank). Finished accumulators are handed to a background unit that requantises them and writes Y
// while the next group is already being multiplied, so the requantiser's latency is hidden.
//
// Weight and parameter RAMs are split into LANES banks selected by the TOP bits of the W / P address
// window. Channel o lives in bank (o % LANES) at local row (o / LANES), so a layer of N channels needs
// ceil(N / LANES) rows per bank. WBASE and PBASE are local (per-bank) offsets. With LANES = 1 this is the
// plain contiguous layout. The driver (tinyml_fpga/engine.py) hides the layout.
//
// It sits on a simple byte bus (driven by uart_bus.v, or any bridge of your own):
//   bus_addr stable at least one clock before bus_rd; bus_rdata is valid in the bus_rd clock;
//   bus_wr is a one-clock strobe with bus_addr/bus_wdata. Register map: see tinyml_fpga/engine.py.
//   0x100-0x111 config (RW), 0x11F CTRL (W), 0x120 STATUS, 0x121-0x124 CYCLES, 0x130-0x137 ID (RO)
//   0x1000 X, 0x2000 Y (RO from the bus), 0x3000 P (16 B per channel), 0x8000 W
// Bus writes to config and memories are ignored while the engine is busy.

`default_nettype none

module tmf_engine #(
    parameter W_AW    = 12,              // weight RAM address width, all banks together (bytes = 2^W_AW, max 15)
    parameter X_AW    = 10,              // input RAM address width    (max 12)
    parameter Y_AW    = 8,               // output RAM address width   (max 12)
    parameter P_AW    = 10,              // parameter RAM address width, all banks together (16 B per channel, max 12)
    parameter LANES   = 1,               // output channels computed at once: 1, 2, 4 or 8
    parameter RQ_MODE = 0                // requantiser: 0 serial (smallest), 1 pipelined (needs 4 multipliers)
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
    // Bank-select bits: log2(LANES).
    localparam LB = (LANES >= 8) ? 3 : (LANES >= 4) ? 2 : (LANES >= 2) ? 1 : 0;
    // Address width inside one weight bank.
    localparam WA = W_AW - LB;
    // Address width inside one parameter bank.
    localparam PA = P_AW - LB;
    // Weight bank size in bytes.
    localparam [32:0] BWSIZE = 33'd1 << WA;
    // Input RAM size.
    localparam [32:0] XSIZE = 33'd1 << X_AW;
    // Output RAM size.
    localparam [32:0] YSIZE = 33'd1 << Y_AW;
    // Parameter bank size in bytes.
    localparam [32:0] BPSIZE = 33'd1 << PA;
    // LANES as a 4-bit constant.
    localparam [3:0] LANES4 = LANES;
    // Feature bits reported in the ID block.
    localparam [7:0] FEATURES = {7'd0, (RQ_MODE == 1)};

    // Sequencer states.
    localparam [3:0] S_IDLE  = 4'd0;     // waiting for CTRL.start
    localparam [3:0] S_CHK   = 4'd1;     // bounds check for the next group
    localparam [3:0] S_PAR   = 4'd2;     // fetch the 9 parameter bytes of every lane
    localparam [3:0] S_INIT  = 4'd3;     // load biases
    localparam [3:0] S_DOT   = 4'd4;     // issue K multiply-accumulates
    localparam [3:0] S_DRAIN = 4'd5;     // let the MAC pipeline empty, then hand the group to the output unit
    localparam [3:0] S_NEXT  = 4'd6;     // advance to the next group
    localparam [3:0] S_FIN   = 4'd7;     // wait for the output unit, then finish
    localparam [3:0] S_ERRW  = 4'd8;     // wait for the output unit, then report an error

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
    // Weight bank addressed by the bus.
    wire [2:0] bank_w = (bus_addr >> WA) & (LANES - 1);
    // Parameter bank addressed by the bus.
    wire [2:0] bank_p = (bus_addr >> PA) & (LANES - 1);

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
    // Weight base offset (local, per bank).
    wire [15:0] cWBASE   = {cfg[9],  cfg[8]};
    // Parameter base record (local, per bank).
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
    // First output channel of the current group.
    reg [15:0] oi;
    // MAC issue counter.
    reg [15:0] ki;
    // Parameter byte counter.
    reg [3:0]  pc;
    // Start of the current vector in X.
    reg [32:0] xvec;
    // Start of the current group's X slice.
    reg [32:0] xrow;
    // Local weight row start of the current group (per bank).
    reg [32:0] wrow;
    // X read pointer during the dot product.
    reg [32:0] xp;
    // Local W read pointer during the dot product (same for every bank).
    reg [32:0] wp;
    // Local byte address of the current parameter record (same for every bank).
    reg [32:0] pp;
    // Output bytes scheduled so far (checked against the Y size before each group).
    reg [32:0] ys;
    // Pipeline stage 1 valid (RAM data available).
    reg v1;
    // Pipeline stage 2 valid (product available).
    reg v2;

    // ---------------------------------------------------------------- group geometry
    // Dot length, zero-extended to the pointer width.
    wire [32:0] k33  = {17'd0, cK};
    // Vector stride, zero-extended to the pointer width.
    wire [32:0] xs33 = {17'd0, cXSTRIDE};
    // Channels left in this vector.
    wire [15:0] left = cN - oi;
    // Channels in this group: one for depthwise, otherwise up to LANES.
    wire [3:0]  cnt  = cDW ? 4'd1 : ((left < LANES) ? left[3:0] : LANES4);
    // Lane of the first channel of the group (depthwise channels use lane o % LANES).
    wire [2:0]  l0   = oi[2:0] & (LANES - 1);
    // Output bytes of this group (four per channel in raw mode).
    wire [32:0] gob  = cRAW ? {27'd0, cnt, 2'b00} : {29'd0, cnt};
    // The next group fits in every memory.
    wire        fits = (xrow + k33 <= XSIZE) && (wrow + k33 <= BWSIZE) && (pp + 33'd16 <= BPSIZE) && (ys + gob <= YSIZE);
    // Last group of a vector.
    wire        last_o = ({1'b0, oi} + {13'd0, cnt} >= {1'b0, cN});
    // Last vector of the job.
    wire        last_v = (vi == cNVEC - 16'd1);

    // ---------------------------------------------------------------- RAMs and lanes
    // RAM read data of the X RAM and the Y RAM.
    wire [7:0] x_q, y_q;
    // Weight and parameter bytes of every bank, packed.
    wire [8*LANES-1:0] w_q, p_q;
    // Lane results: accumulators, multipliers, shifts, packed.
    wire [32*LANES-1:0] acc_w, mult_w;
    wire [8*LANES-1:0]  shift_w;
    // Y write enable from the output unit.
    reg        y_we;
    // Y write address.
    reg [Y_AW-1:0] y_wa;
    // Y write data.
    reg [7:0]  y_wd;
    // Bus write is allowed (engine idle).
    wire bus_wr_ok = bus_wr & ~busy;
    // Input byte plus the input offset (-255 .. 255), shared by all lanes.
    wire signed [8:0]  xoff = $signed(x_q) + $signed(cINOFF[8:0]);
    // The MAC pipeline advances while the lane controls below are valid.
    wire par_en   = busy && (st == S_PAR) && (pc != 4'd0);
    // Bias load strobe.
    wire acc_init = (st == S_INIT);
    // Parameter RAM read address (local, shared by all banks).
    wire [32:0] p_ra = pp + {29'd0, pc};

    // This contains the instantiation for the X RAM (input vectors)
    tmf_ram #(.AW(X_AW)) u_x (.clk(clk), .we(bus_wr_ok & sel_x), .waddr(bus_addr[X_AW-1:0]), .wdata(bus_wdata),
                               .raddr(busy ? xp[X_AW-1:0] : bus_addr[X_AW-1:0]), .q(x_q));
    // This contains the instantiation for the Y RAM (outputs)
    tmf_ram #(.AW(Y_AW)) u_y (.clk(clk), .we(y_we), .waddr(y_wa), .wdata(y_wd),
                               .raddr(bus_addr[Y_AW-1:0]), .q(y_q));

    // One lane per output channel of a group: weight bank, parameter bank and MAC.
    genvar gl;
    generate
        for (gl = 0; gl < LANES; gl = gl + 1) begin : lane
            // This contains the instantiation for this lane's weight bank
            tmf_ram #(.AW(WA)) u_w (.clk(clk), .we(bus_wr_ok & sel_w & (bank_w == gl)), .waddr(bus_addr[WA-1:0]),
                                    .wdata(bus_wdata), .raddr(busy ? wp[WA-1:0] : bus_addr[WA-1:0]),
                                    .q(w_q[8*gl +: 8]));
            // This contains the instantiation for this lane's parameter bank
            tmf_ram #(.AW(PA)) u_p (.clk(clk), .we(bus_wr_ok & sel_p & (bank_p == gl)), .waddr(bus_addr[PA-1:0]),
                                    .wdata(bus_wdata),
                                    .raddr(busy ? p_ra[PA-1:0] : bus_addr[PA-1:0]),
                                    .q(p_q[8*gl +: 8]));
            // This contains the instantiation for this lane's MAC
            tmf_lane u_lane (.clk(clk), .par_en(par_en), .p_byte(p_q[8*gl +: 8]), .acc_init(acc_init), .v1(v1), .v2(v2),
                             .xoff(xoff), .w_byte(w_q[8*gl +: 8]), .acc_o(acc_w[32*gl +: 32]),
                             .mult_o(mult_w[32*gl +: 32]), .shift_o(shift_w[8*gl +: 8]));
        end
    endgenerate

    // ---------------------------------------------------------------- output unit (background requantiser)
    // Group staging: accumulators, multipliers and shifts copied at hand-over.
    reg [32*LANES-1:0] s_acc, s_mult;
    reg [8*LANES-1:0]  s_shift;
    // Output unit has a group to finish.
    reg        sbusy;
    // First lane of the staged group.
    reg [2:0]  s_l0;
    // Channels in the staged group.
    reg [3:0]  s_cnt;
    // Channels issued to the requantiser (or raw channels completed).
    reg [3:0]  s_iss;
    // Results written.
    reg [3:0]  s_wr;
    // Raw byte counter.
    reg [1:0]  raw_b;
    // Y write pointer (bytes).
    reg [32:0] wy;
    // Hand-over of a finished group to the output unit.
    wire do_hand = (st == S_DRAIN) && !v1 && !v2 && !sbusy;
    // Lane being issued.
    wire [2:0] s_lane = s_l0 + s_iss[2:0];
    // Requantiser interface.
    wire        rq_ready, rq_done;
    wire [7:0]  rq_y;
    // Issue the next staged channel into the requantiser.
    wire        rq_start = sbusy && !cRAW && (s_iss < s_cnt) && rq_ready;

    // Operands of the staged channel being issued.
    wire signed [31:0] rq_acc  = s_acc[32*s_lane +: 32];
    wire        [31:0] rq_mult = s_mult[32*s_lane +: 32];
    wire signed [7:0]  rq_sh   = s_shift[8*s_lane +: 8];

    // This contains the instantiation for the requantiser
    tmf_rq #(.MODE(RQ_MODE)) u_rq (.clk(clk), .rst(rst), .start(rq_start), .acc(rq_acc), .mult(rq_mult), .shift(rq_sh),
                                   .zp(cZP), .act_min(cAMIN), .act_max(cAMAX), .ready(rq_ready), .done(rq_done), .y(rq_y));

    // Output unit: requantise or copy staged accumulators and write Y, one channel after another.
    always @(posedge clk) begin
        // Y write strobe defaults low.
        y_we <= 1'b0;
        // Reset empties the unit.
        if (rst) begin
            // Nothing staged.
            sbusy <= 1'b0;
        end else if (do_hand) begin
            // Take over the group staged this clock.
            sbusy <= 1'b1;
            // Remember the first lane.
            s_l0  <= l0;
            // And how many channels.
            s_cnt <= cnt;
            // Nothing issued yet.
            s_iss <= 4'd0;
            // Nothing written yet.
            s_wr  <= 4'd0;
            // First raw byte.
            raw_b <= 2'd0;
        end else if (sbusy) begin
            // Raw mode: write the four bytes of each accumulator, one per clock.
            if (cRAW) begin
                // Write strobe.
                y_we  <= 1'b1;
                // At the output pointer.
                y_wa  <= wy[Y_AW-1:0];
                // Byte raw_b of the current lane's accumulator.
                y_wd  <= s_acc[32*s_lane + 8*raw_b +: 8];
                // Advance the output pointer.
                wy    <= wy + 33'd1;
                // Next byte.
                raw_b <= raw_b + 2'd1;
                // After byte 3 the channel is complete.
                if (raw_b == 2'd3) begin
                    // Next channel.
                    s_iss <= s_iss + 4'd1;
                    // Last channel: the unit is free after this write.
                    if (s_iss == s_cnt - 4'd1) sbusy <= 1'b0;
                end
            end else begin
                // Count issues (the start pulse itself is the combinational rq_start).
                if (rq_start) s_iss <= s_iss + 4'd1;
                // A requantised byte arrives: write it.
                if (rq_done) begin
                    // Write strobe.
                    y_we  <= 1'b1;
                    // At the output pointer.
                    y_wa  <= wy[Y_AW-1:0];
                    // Result byte.
                    y_wd  <= rq_y;
                    // Advance the output pointer.
                    wy    <= wy + 33'd1;
                    // One more result.
                    s_wr  <= s_wr + 4'd1;
                    // Last result of the group: the unit is free after this write.
                    if (s_wr == s_cnt - 4'd1) sbusy <= 1'b0;
                end
            end
        end
        // Job start rewinds the output pointer (start is decoded again in the sequencer).
        if (!busy && bus_wr && sel_reg && bus_addr[7:0] == 8'h1F && bus_wdata[0]) wy <= 33'd0;
    end

    // Staging registers capture the lanes at hand-over.
    always @(posedge clk) begin
        // Copy accumulators, multipliers and shifts.
        if (do_hand) begin
            // Accumulators.
            s_acc   <= acc_w;
            // Multipliers.
            s_mult  <= mult_w;
            // Shifts.
            s_shift <= shift_w;
        end
    end

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
        if (sel_reg && bus_addr[7:0] == 8'h31) bus_rdata = 8'h02;
        // W address width.
        if (sel_reg && bus_addr[7:0] == 8'h32) bus_rdata = W_AW;
        // X address width.
        if (sel_reg && bus_addr[7:0] == 8'h33) bus_rdata = X_AW;
        // Y address width.
        if (sel_reg && bus_addr[7:0] == 8'h34) bus_rdata = Y_AW;
        // P address width.
        if (sel_reg && bus_addr[7:0] == 8'h35) bus_rdata = P_AW;
        // Lanes.
        if (sel_reg && bus_addr[7:0] == 8'h36) bus_rdata = LANES;
        // Feature bits.
        if (sel_reg && bus_addr[7:0] == 8'h37) bus_rdata = FEATURES;
        // X window.
        if (sel_x) bus_rdata = x_q;
        // Y window.
        if (sel_y) bus_rdata = y_q;
        // P window: the addressed bank.
        if (sel_p) bus_rdata = p_q[8*bank_p +: 8];
        // W window: the addressed bank.
        if (sel_w) bus_rdata = w_q[8*bank_w +: 8];
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

    // ---------------------------------------------------------------- MAC pipeline control
    // Stage valid bits follow the issue of RAM addresses.
    always @(posedge clk) begin
        // Stage 2 valid follows stage 1.
        v2 <= v1;
        // Stage 1 valid: an address was issued this clock.
        v1 <= (st == S_DOT);
    end

    // ---------------------------------------------------------------- sequencer
    // Job control: start, per-group loop, bounds checks, status.
    always @(posedge clk) begin
        // Count clocks while busy.
        if (busy) cycles <= cycles + 32'd1;
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
                        // First group's X slice.
                        xrow   <= 33'd0;
                        // First weight row.
                        wrow   <= {17'd0, cWBASE};
                        // First parameter record (16 bytes per channel).
                        pp     <= {13'd0, cPBASE, 4'd0};
                        // No output scheduled yet.
                        ys     <= 33'd0;
                        // Check bounds first.
                        st     <= S_CHK;
                    end
                end
                // Every memory access of this group must be in range.
                S_CHK: begin
                    // Out of range: stop with an error once the output unit is idle.
                    if (!fits) begin
                        // Wait, then report.
                        st <= S_ERRW;
                    end else begin
                        // Fetch the parameters.
                        pc <= 4'd0;
                        // Next state.
                        st <= S_PAR;
                    end
                end
                // Read 9 parameter bytes per lane; the byte addressed in clock n arrives in clock n+1.
                S_PAR: begin
                    // Next byte (the lanes shift it in through par_en).
                    pc <= pc + 4'd1;
                    // After the 9th byte is captured, load the operands.
                    if (pc == 4'd9) st <= S_INIT;
                end
                // Load the operands and start the dot product.
                S_INIT: begin
                    // X read pointer.
                    xp  <= xrow;
                    // W read pointer.
                    wp  <= wrow;
                    // MAC counter.
                    ki  <= 16'd0;
                    // Go (the lanes load their biases through acc_init).
                    st  <= S_DOT;
                end
                // One RAM read per clock; the lanes add the products.
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
                // Pipeline empties two clocks after the last issue; then the group goes to the output unit.
                S_DRAIN: if (do_hand) begin
                    // The group's output bytes are now scheduled.
                    ys <= ys + gob;
                    // Move on.
                    st <= S_NEXT;
                end
                // Step to the next group / vector, or finish.
                S_NEXT: begin
                    // Last group of this vector.
                    if (last_o) begin
                        // Last vector too: the job is complete once the output unit is empty.
                        if (last_v) begin
                            // Wait for the output unit.
                            st <= S_FIN;
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
                            // Check the new group.
                            st   <= S_CHK;
                        end
                    end else begin
                        // Next group.
                        oi <= oi + {12'd0, cnt};
                        // Depthwise: next channel has its own X slice.
                        if (cDW) xrow <= xrow + k33;
                        // Local row advances after the last lane of a bank row (always for non-depthwise groups).
                        if (!cDW || ((oi[2:0] & (LANES - 1)) == (LANES - 1))) begin
                            // Next weight row.
                            wrow <= wrow + k33;
                            // Next parameter record.
                            pp   <= pp + 33'd16;
                        end
                        // Check the new group.
                        st <= S_CHK;
                    end
                end
                // Wait for the last results to be written, then finish.
                S_FIN: if (!sbusy) begin
                    // Done.
                    done <= 1'b1;
                    // Not running.
                    busy <= 1'b0;
                    // Idle.
                    st   <= S_IDLE;
                end
                // Error exit after the output unit has drained.
                S_ERRW: if (!sbusy) begin
                    // Flag.
                    err  <= 1'b1;
                    // Finished.
                    done <= 1'b1;
                    // Not running.
                    busy <= 1'b0;
                    // Idle.
                    st   <= S_IDLE;
                end
                // Unused encodings recover to idle.
                default: st <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
