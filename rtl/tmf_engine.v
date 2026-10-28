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
//   0x100-0x115 config (RW), 0x118 QCNT, 0x119 QDONE, 0x11F CTRL (W), 0x120 STATUS, 0x121-0x124 CYCLES,
//   0x130-0x138 ID (RO), 0x140-0x15E window descriptor (RW, HAS_WIN), 0x160-0x16F performance counters (RO)
//   0x1000 X, 0x2000 Y (RO from the bus), 0x3000 P (16 B per channel), 0x4000 D (job descriptors), 0x8000 W
// With HAS_QUEUE a list of job descriptors (the same bytes as the config block, 64 B per slot) can be written
// to D and run back to back with one CTRL = 0x02, which saves the host a configure / start / poll round trip
// per job. HAS_PERF adds counters for MAC clocks, stall clocks, output bytes and finished jobs.
// HAS_W4 stores weights as signed 4-bit values, two per W byte (low nibble = even index), halving weight RAM.
// HAS_WIN adds an input-window generator: with MODE bit 2 the engine walks the (padded, strided, dilated)
// convolution window of every output pixel straight out of an NHWC image in X, so the host no longer builds
// im2col patches. HAS_FUSE lets a job write its outputs into X (MODE bit 3) instead of Y, so the next layer of
// a queue reads them in place and a whole run of layers needs one start.
// Bus writes to config and memories are ignored while the engine is busy.

`default_nettype none

module tmf_engine #(
    parameter W_AW    = 12,              // weight RAM address width, all banks together (bytes = 2^W_AW, max 15)
    parameter X_AW    = 10,              // input RAM address width    (max 12)
    parameter Y_AW    = 8,               // output RAM address width   (max 12)
    parameter P_AW    = 10,              // parameter RAM address width, all banks together (16 B per channel, max 12)
    parameter LANES   = 1,               // output channels computed at once: 1, 2, 4 or 8
    parameter RQ_MODE = 0,               // requantiser: 0 serial (smallest), 1 pipelined (needs 4 multipliers)
    parameter HAS_QUEUE = 0,             // job descriptor queue (CTRL 0x02 runs QCNT descriptors from D)
    parameter D_AW    = 9,               // descriptor RAM address width (64 B per descriptor, max 12)
    parameter HAS_PERF = 0,              // performance counters at 0x160
    parameter HAS_WIN  = 0,              // input window generator (MODE bit 2)
    parameter HAS_FUSE = 0,              // outputs can be written into X (MODE bit 3)
    parameter HAS_W4   = 0               // 4-bit weights, two per byte (MODE bit 4)
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
    localparam [7:0] FEATURES = {1'b0, (HAS_PERF != 0), 1'b0, (HAS_W4 != 0), (HAS_FUSE != 0), (HAS_WIN != 0), (HAS_QUEUE != 0), (RQ_MODE == 1)};
    // Descriptor RAM size in bytes.
    localparam [32:0] DSIZE = 33'd1 << D_AW;
    // Configuration bytes held in a descriptor.
    localparam NCFG = (HAS_WIN != 0) ? 53 : 22;
    // Width of the output write address (the outputs may go to X).
    localparam YW = (HAS_FUSE != 0 && X_AW > Y_AW) ? X_AW : Y_AW;

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
    localparam [3:0] S_LD    = 4'd9;     // load the next descriptor from D into the config registers
    localparam [3:0] S_JSTART = 4'd10;   // validate the config and set up the job

    // ---------------------------------------------------------------- bus decode
    // Address is in the config/control register block.
    wire sel_reg = (bus_addr[15:8] == 8'h01);
    // Address is in the X window.
    wire sel_x   = (bus_addr[15:12] == 4'h1);
    // Address is in the Y window.
    wire sel_y   = (bus_addr[15:12] == 4'h2);
    // Address is in the P window.
    wire sel_p   = (bus_addr[15:12] == 4'h3);
    // Address is in the D (descriptor) window.
    wire sel_d   = (bus_addr[15:12] == 4'h4);
    // Address is in the W window.
    wire sel_w   = bus_addr[15];
    // Address is in the window descriptor block 0x140..0x15E.
    wire win_reg = (HAS_WIN != 0) && sel_reg && (bus_addr[7:6] == 2'b01) && (bus_addr[5:0] <= 6'd30);
    // Address is a config byte; its index in the descriptor-ordered array (0x100.. = 0..21, 0x140.. = 22..52).
    wire cfg_hit = (sel_reg && (bus_addr[7:0] < 8'd22)) || win_reg;
    // Index of the addressed config byte.
    wire [5:0] cfg_idx = win_reg ? (6'd22 + bus_addr[5:0]) : bus_addr[5:0];
    // Weight bank addressed by the bus.
    wire [2:0] bank_w = (bus_addr >> WA) & (LANES - 1);
    // Parameter bank addressed by the bus.
    wire [2:0] bank_p = (bus_addr >> PA) & (LANES - 1);

    // ---------------------------------------------------------------- configuration registers
    // Raw config bytes 0x100..0x115 (descriptor order).
    reg  [7:0] cfg [0:NCFG-1];
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
    // Window mode bit (HAS_WIN).
    wire cWIN  = (HAS_WIN != 0) && cfg[17][2];
    // 4-bit weights: each W byte holds two, low nibble first (HAS_W4).
    wire cW4   = (HAS_W4 != 0) && cfg[17][4];
    // Write the outputs into X (HAS_FUSE).
    wire cOUTX = (HAS_FUSE != 0) && cfg[17][3];
    // Input base offset in X.
    wire [15:0] cXBASE = {cfg[19], cfg[18]};
    // Output base offset in Y.
    wire [15:0] cYBASE = {cfg[21], cfg[20]};
    // Number of descriptors to run (queue mode).
    reg  [7:0] cQCNT;
    // Window descriptor bytes (zero without HAS_WIN).
    wire [7:0] wf [0:30];
    genvar gi;
    generate
        for (gi = 0; gi < 31; gi = gi + 1) begin : g_wf
            if (HAS_WIN != 0) begin : g_on
                // Byte gi of the window block.
                assign wf[gi] = cfg[22 + gi];
            end else begin : g_off
                // No window generator: zero.
                assign wf[gi] = 8'h00;
            end
        end
    endgenerate
    // Image height, width, channels.
    wire signed [15:0] cH   = {wf[1],  wf[0]};
    wire signed [15:0] cWW  = {wf[3],  wf[2]};
    wire        [15:0] cC   = {wf[5],  wf[4]};
    // Kernel height and width.
    wire [7:0] cKH = wf[6];
    wire [7:0] cKW = wf[7];
    // Dilation (rows, columns), stride (rows, columns), depth multiplier.
    wire [7:0] cDH = wf[8];
    wire [7:0] cDWD = wf[9];
    wire [7:0] cSH = wf[10];
    wire [7:0] cSW = wf[11];
    wire [7:0] cDM = wf[12];
    // Output pixels per row.
    wire [15:0] cOW = {wf[14], wf[13]};
    // Input row and column of the first window's top-left tap.
    wire signed [15:0] cIY0 = {wf[16], wf[15]};
    wire signed [15:0] cIX0 = {wf[18], wf[17]};
    // X address of that tap (XBASE included).
    wire signed [15:0] cORG = {wf[20], wf[19]};
    // Byte steps: one output column, one output row, one kernel column, one kernel row.
    wire [15:0] cSWC  = {wf[22], wf[21]};
    wire [15:0] cSHWC = {wf[24], wf[23]};
    wire [15:0] cKXS  = {wf[26], wf[25]};
    wire [15:0] cKYS  = {wf[28], wf[27]};
    // Bytes of image the job may touch.
    wire [15:0] cXLEN = {wf[30], wf[29]};

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
    // Queue mode: the current job comes from a descriptor.
    reg        q_run;
    // Descriptor being run.
    reg [7:0]  qi;
    // Descriptors finished in the last queue run.
    reg [7:0]  qdone;
    // Descriptor byte counter while loading.
    reg [5:0]  ld;
    // Descriptor RAM read data.
    wire [7:0] d_q;
    // Window generator state (HAS_WIN): row and column of the current tap, of the pixel's first tap, and the
    // X address of the pixel (vbase), of its first row (vrow), of the current kernel column (xks) and row (xrs).
    reg signed [15:0] iy, ix, iy0, ix0;
    // (16 bits are enough: X is at most 4096 bytes and the arithmetic wraps like the address does.)
    reg [15:0] vbase, vrow, xks, xrs;
    // Channel inside a kernel tap, kernel column, output column in the row.
    reg [15:0] wc;
    reg [7:0]  wkx;
    reg [15:0] wox;
    // Depthwise: input channel offset of the current output channel and position inside its depth multiplier.
    reg [15:0] icoff;
    reg [7:0]  dmc;
    // Pipeline stage 1 valid (RAM data available).
    reg v1;
    // Pipeline stage 2 valid (product available).
    reg v2;

    // ---------------------------------------------------------------- group geometry
    // Dot length, zero-extended to the pointer width.
    wire [32:0] k33  = {17'd0, cK};
    // Weight bytes per output channel row: K, or ceil(K / 2) for 4-bit weights.
    wire [32:0] kb33 = cW4 ? ((k33 + 33'd1) >> 1) : k33;
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
    // The window or the vector slice fits in X.
    wire        x_ok = cWIN ? ({17'd0, cXBASE} + {17'd0, cXLEN} <= XSIZE) : (xrow + k33 <= XSIZE);
    // The output goes to X or to Y.
    wire        y_ok = (ys + gob <= (cOUTX ? XSIZE : YSIZE));
    // The next group fits in every memory.
    wire        fits = x_ok && (wrow + kb33 <= BWSIZE) && (pp + 33'd16 <= BPSIZE) && y_ok;
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
    reg [YW-1:0] y_wa;
    // Y write data.
    reg [7:0]  y_wd;
    // Bus write is allowed (engine idle).
    wire bus_wr_ok = bus_wr & ~busy;
    // Input byte plus the input offset (-255 .. 255), shared by all lanes.
    wire signed [8:0]  xoff_r = $signed(x_q) + $signed(cINOFF[8:0]);
    // Window tap outside the image (registered with the RAM read): it contributes zero.
    reg pad1;
    // The byte the lanes multiply.
    wire signed [8:0]  xoff = pad1 ? 9'sd0 : xoff_r;
    // The MAC pipeline advances while the lane controls below are valid.
    wire par_en   = busy && (st == S_PAR) && (pc != 4'd0);
    // Bias load strobe.
    wire acc_init = (st == S_INIT);
    // Parameter RAM read address (local, shared by all banks).
    wire [32:0] p_ra = pp + {29'd0, pc};

    // Output address zero-extended so it can address X.
    wire [11:0] y_wa12 = y_wa;
    // This contains the instantiation for the X RAM (input vectors)
    tmf_ram #(.AW(X_AW)) u_x (.clk(clk), .we((bus_wr_ok & sel_x) | (y_we & cOUTX)),
                               .waddr((HAS_FUSE != 0 && busy) ? y_wa12[X_AW-1:0] : bus_addr[X_AW-1:0]),
                               .wdata((HAS_FUSE != 0 && busy) ? y_wd : bus_wdata),
                               .raddr(busy ? xp[X_AW-1:0] : bus_addr[X_AW-1:0]), .q(x_q));
    // This contains the instantiation for the Y RAM (outputs)
    tmf_ram #(.AW(Y_AW)) u_y (.clk(clk), .we(y_we & ~cOUTX), .waddr(y_wa12[Y_AW-1:0]), .wdata(y_wd),
                               .raddr(bus_addr[Y_AW-1:0]), .q(y_q));

    // Which nibble of the weight byte the lanes get (registered with the RAM read).
    reg nsel;
    always @(posedge clk) nsel <= ki[0];
    // Tap outside the image (only in window mode).
    wire pad_now = cWIN && (iy < 16'sd0 || iy >= cH || ix < 16'sd0 || ix >= cWW);
    // Register it with the RAM read.
    always @(posedge clk) pad1 <= pad_now && (st == S_DOT);

    // Descriptor RAM (only with HAS_QUEUE).
    generate
        if (HAS_QUEUE != 0) begin : g_d
            // This contains the instantiation for the descriptor RAM
            tmf_ram #(.AW(D_AW)) u_d (.clk(clk), .we(bus_wr_ok & sel_d), .waddr(bus_addr[D_AW-1:0]), .wdata(bus_wdata),
                                       .raddr(busy ? {qi[D_AW-7:0], ld} : bus_addr[D_AW-1:0]), .q(d_q));
        end else begin : g_nod
            // No queue: the descriptor window reads as zero.
            assign d_q = 8'h00;
        end
    endgenerate

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
            // The weight the MAC multiplies: the whole byte, or one sign-extended nibble.
            wire [3:0] wnib = nsel ? w_q[8*gl+4 +: 4] : w_q[8*gl +: 4];
            wire [7:0] wsel = cW4 ? {{4{wnib[3]}}, wnib} : w_q[8*gl +: 8];
            // This contains the instantiation for this lane's MAC
            tmf_lane u_lane (.clk(clk), .par_en(par_en), .p_byte(p_q[8*gl +: 8]), .acc_init(acc_init), .v1(v1), .v2(v2),
                             .xoff(xoff), .w_byte(wsel), .acc_o(acc_w[32*gl +: 32]),
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
    // The job's registers are being initialised this clock (valid config).
    wire job_init = (st == S_JSTART) && !(cK == 16'd0 || cN == 16'd0 || cNVEC == 16'd0);
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
                y_wa  <= wy[YW-1:0];
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
                    y_wa  <= wy[YW-1:0];
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
        // Job set-up rewinds the output pointer to the job's Y base.
        if (job_init) wy <= {17'd0, cYBASE};
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

    // ---------------------------------------------------------------- performance counters
    // Clocks spent issuing multiply-accumulates, clocks waited for the output unit, output bytes, finished jobs.
    reg [31:0] p_mac, p_stall, p_out, p_jobs;
    // The four counters packed for the bus read mux (zero without HAS_PERF).
    wire [127:0] perf_bus = (HAS_PERF != 0) ? {p_jobs, p_out, p_stall, p_mac} : 128'd0;
    // CTRL register write strobe.
    wire ctl_wr = bus_wr && sel_reg && bus_addr[7:0] == 8'h1F;

    // Counters: cleared by a start command, counting while the engine works.
    always @(posedge clk) begin
        // Only built with HAS_PERF.
        if (HAS_PERF != 0) begin
            // A start command from idle clears them.
            if (st == S_IDLE && ctl_wr && (bus_wdata[0] || bus_wdata[1])) begin
                // Clear.
                p_mac <= 32'd0; p_stall <= 32'd0; p_out <= 32'd0; p_jobs <= 32'd0;
            end else begin
                // A multiply-accumulate address is issued.
                if (st == S_DOT) p_mac <= p_mac + 32'd1;
                // The group is complete but the output unit is still busy with the previous one.
                if (st == S_DRAIN && !v1 && !v2 && sbusy) p_stall <= p_stall + 32'd1;
                // One byte written to Y.
                if (y_we) p_out <= p_out + 32'd1;
                // A job completes.
                if (st == S_FIN && !sbusy) p_jobs <= p_jobs + 32'd1;
            end
        end
    end

    // ---------------------------------------------------------------- bus read mux
    // Read data: registers combinationally, memories from their registered outputs.
    always @* begin
        // Default for unmapped addresses.
        bus_rdata = 8'h00;
        // Config read-back (the window block is write-only: it reads as zero).
        if (cfg_hit && !win_reg) bus_rdata = cfg[cfg_idx];
        // Descriptor count.
        if (sel_reg && bus_addr[7:0] == 8'h18) bus_rdata = cQCNT;
        // Descriptors finished in the last queue run.
        if (sel_reg && bus_addr[7:0] == 8'h19) bus_rdata = qdone;
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
        // Descriptor RAM address width (0 without a queue).
        if (sel_reg && bus_addr[7:0] == 8'h38) bus_rdata = (HAS_QUEUE != 0) ? D_AW : 0;
        // Performance counters, little endian: MAC, STALL, OUT, JOBS.
        if (sel_reg && bus_addr[7:4] == 4'h6) bus_rdata = perf_bus[8*bus_addr[3:0] +: 8];
        // D window.
        if (sel_d) bus_rdata = d_q;
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
    // A descriptor byte is being written this clock (it arrives one clock after its address).
    wire        ld_we = (HAS_QUEUE != 0) && (st == S_LD) && (ld != 6'd0);
    // Single write port of the config file: a descriptor load or a bus write.
    wire        cw_en = ld_we | (bus_wr_ok & cfg_hit);
    // Index written.
    wire [5:0]  cw_ix = ld_we ? (ld - 6'd1) : cfg_idx;
    // Data written.
    wire [7:0]  cw_d  = ld_we ? d_q : bus_wdata;
    // Config register file: bus writes when idle, descriptor loads while a queue runs.
    always @(posedge clk) begin
        // Reset clears every config byte.
        if (rst) begin
            // Clear all bytes.
            for (i = 0; i < NCFG; i = i + 1) cfg[i] <= 8'h00;
            // And the descriptor count.
            cQCNT <= 8'd0;
        end else begin
            // Store the addressed config byte.
            if (cw_en) cfg[cw_ix] <= cw_d;
            // Store the descriptor count.
            if (bus_wr_ok && sel_reg && bus_addr[7:0] == 8'h18) cQCNT <= bus_wdata;
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
            if (ctl_wr && (bus_wdata[0] || bus_wdata[1]) && busy) err <= 1'b1;
            // Clear-status command from the bus (only meaningful when idle).
            if (ctl_wr && bus_wdata[7] && !busy) begin
                // Clear done.
                done <= 1'b0;
                // Clear error.
                err  <= 1'b0;
            end
            // State dispatch.
            case (st)
                // Wait for CTRL.start (0x01 single job, 0x02 descriptor queue).
                S_IDLE: if (ctl_wr && bus_wdata[0]) begin
                    // Running.
                    busy   <= 1'b1;
                    // New job: clear status.
                    done   <= 1'b0;
                    // No error yet.
                    err    <= 1'b0;
                    // Restart the clock counter.
                    cycles <= 32'd0;
                    // The job comes from the config registers.
                    q_run  <= 1'b0;
                    // Validate and set up.
                    st     <= S_JSTART;
                end else if (HAS_QUEUE != 0 && ctl_wr && bus_wdata[1]) begin
                    // Nothing to run, or more descriptors than fit in D: error at once.
                    if (cQCNT == 8'd0 || {19'd0, cQCNT, 6'd0} > DSIZE) begin
                        // Flag the error.
                        err  <= 1'b1;
                        // And finish.
                        done <= 1'b1;
                    end else begin
                        // Running.
                        busy   <= 1'b1;
                        // New run: clear status.
                        done   <= 1'b0;
                        // No error yet.
                        err    <= 1'b0;
                        // Restart the clock counter.
                        cycles <= 32'd0;
                        // Jobs come from descriptors.
                        q_run  <= 1'b1;
                        // First descriptor.
                        qi     <= 8'd0;
                        // None finished yet.
                        qdone  <= 8'd0;
                        // Start loading its bytes.
                        ld     <= 6'd0;
                        // Load.
                        st     <= S_LD;
                    end
                end
                // Copy the descriptor into the config registers (the config RAM write is in the config block).
                S_LD: begin
                    // Next byte (the address of byte n is issued in the clock with ld == n).
                    ld <= ld + 6'd1;
                    // The last byte is written in the clock with ld == NCFG.
                    if (ld == NCFG) st <= S_JSTART;
                end
                // Validate the config and set up the job registers.
                S_JSTART: begin
                    // Empty job: error and done at once.
                    if (cK == 16'd0 || cN == 16'd0 || cNVEC == 16'd0) begin
                        // Flag the error.
                        err  <= 1'b1;
                        // Finished.
                        done <= 1'b1;
                        // Not running.
                        busy <= 1'b0;
                        // Idle.
                        st   <= S_IDLE;
                    end else begin
                        // First vector.
                        vi     <= 16'd0;
                        // First channel.
                        oi     <= 16'd0;
                        // First vector starts at the X base.
                        xvec   <= {17'd0, cXBASE};
                        // First group's X slice.
                        xrow   <= {17'd0, cXBASE};
                        // First weight row.
                        wrow   <= {17'd0, cWBASE};
                        // First parameter record (16 bytes per channel).
                        pp     <= {13'd0, cPBASE, 4'd0};
                        // Output scheduling starts at the Y base.
                        ys     <= {17'd0, cYBASE};
                        // Window mode: first pixel.
                        if (cWIN) begin
                            // X address of its first tap.
                            vbase <= cORG;
                            vrow  <= cORG;
                            // Its image row and column.
                            iy0   <= cIY0;
                            ix0   <= cIX0;
                            // First output column.
                            wox   <= 16'd0;
                            // First depthwise channel.
                            icoff <= 16'd0;
                            dmc   <= 8'd0;
                        end
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
                    xp  <= cWIN ? {17'd0, vbase + icoff} : xrow;
                    // Window generator: first tap of the pixel.
                    if (cWIN) begin
                        xks <= vbase + icoff;
                        xrs <= vbase + icoff;
                        iy  <= iy0;
                        ix  <= ix0;
                        wc  <= 16'd0;
                        wkx <= 8'd0;
                    end
                    // W read pointer.
                    wp  <= wrow;
                    // MAC counter.
                    ki  <= 16'd0;
                    // Go (the lanes load their biases through acc_init).
                    st  <= S_DOT;
                end
                // One RAM read per clock; the lanes add the products.
                S_DOT: begin
                    // Next X byte: the next channel, kernel column or kernel row of the window, or the next byte.
                    if (!cWIN) begin
                        // Plain vectors.
                        xp <= xp + 33'd1;
                    end else if (!cDW && wc != cC - 16'd1) begin
                        // Next channel of this tap.
                        wc <= wc + 16'd1;
                        xp <= xp + 33'd1;
                    end else if (wkx != cKW - 8'd1) begin
                        // Next kernel column.
                        wc  <= 16'd0;
                        wkx <= wkx + 8'd1;
                        ix  <= ix + $signed({8'd0, cDWD});
                        xks <= xks + cKXS;
                        xp  <= {17'd0, xks + cKXS};
                    end else begin
                        // Next kernel row.
                        wc  <= 16'd0;
                        wkx <= 8'd0;
                        ix  <= ix0;
                        iy  <= iy + $signed({8'd0, cDH});
                        xrs <= xrs + cKYS;
                        xks <= xrs + cKYS;
                        xp  <= {17'd0, xrs + cKYS};
                    end
                    // Next W byte (4-bit weights: every second clock).
                    wp <= wp + (cW4 ? {32'd0, ki[0]} : 33'd1);
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
                            // Window mode: next output pixel (right, or the start of the next row).
                            if (cWIN) begin
                                // Depthwise channel counters restart.
                                icoff <= 16'd0;
                                dmc   <= 8'd0;
                                // Last column: down by the row stride.
                                if (wox == cOW - 16'd1) begin
                                    wox   <= 16'd0;
                                    iy0   <= iy0 + $signed({8'd0, cSH});
                                    ix0   <= cIX0;
                                    vrow  <= vrow + cSHWC;
                                    vbase <= vrow + cSHWC;
                                end else begin
                                    wox   <= wox + 16'd1;
                                    ix0   <= ix0 + $signed({8'd0, cSW});
                                    vbase <= vbase + cSWC;
                                end
                            end
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
                        // Window depthwise: next input channel after DM output channels.
                        if (cWIN && cDW) begin
                            // Depth multiplier exhausted.
                            if (dmc == cDM - 8'd1) begin
                                dmc   <= 8'd0;
                                icoff <= icoff + 16'd1;
                            end else begin
                                dmc <= dmc + 8'd1;
                            end
                        end
                        // Local row advances after the last lane of a bank row (always for non-depthwise groups).
                        if (!cDW || ((oi[2:0] & (LANES - 1)) == (LANES - 1))) begin
                            // Next weight row.
                            wrow <= wrow + kb33;
                            // Next parameter record.
                            pp   <= pp + 33'd16;
                        end
                        // Check the new group.
                        st <= S_CHK;
                    end
                end
                // Wait for the last results to be written, then finish (or load the next descriptor).
                S_FIN: if (!sbusy) begin
                    // Another descriptor to run.
                    if (q_run && ({1'b0, qi} + 9'd1 < {1'b0, cQCNT})) begin
                        // Count the finished one.
                        qdone <= qdone + 8'd1;
                        // Select the next.
                        qi    <= qi + 8'd1;
                        // Start loading its bytes.
                        ld    <= 6'd0;
                        // Load.
                        st    <= S_LD;
                    end else begin
                        // Count the finished one.
                        if (q_run) qdone <= qdone + 8'd1;
                        // Done.
                        done <= 1'b1;
                        // Not running.
                        busy <= 1'b0;
                        // Idle.
                        st   <= S_IDLE;
                    end
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
