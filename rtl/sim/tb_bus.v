// tb_bus.v - end-to-end testbench for the SPI, I2C and AXI4-Lite paths (tinyml_*_top = bridge + engine).
// Replays system_vectors.hex (a recorded real model inference, see gen_system_vectors.py) through a master model
// of the selected bus, so the bridge, the engine and the PC/MCU-side frame rules are all checked.
// BUS = 1 SPI, 2 I2C, 3 AXI4-Lite. The engine uses the default RAM sizes because the stream reads them back.

`timescale 1ns/1ps
`default_nettype none

module tb_bus #(
    parameter BUS     = 1,               // bus under test
    parameter LANES   = 1,               // engine lanes (must match gen_system_vectors.py)
    parameter RQ_MODE = 0,               // requantiser (must match gen_system_vectors.py)
    parameter HAS_QUEUE = 0,             // descriptor queue (must match gen_system_vectors.py)
    parameter HAS_PERF = 0,              // performance counters (must match gen_system_vectors.py)
    parameter HAS_WIN = 0,               // window generator (must match gen_system_vectors.py)
    parameter HAS_FUSE = 0,              // write-back to X (must match gen_system_vectors.py)
    parameter HAS_W4 = 0,                // 4-bit weights (must match gen_system_vectors.py)
    parameter HAS_CB = 0                 // codebook weights (must match gen_system_vectors.py)
);

    // Half period of the slow bus clocks in fabric clocks.
    localparam HP = 8;
    // I2C slave address under test.
    localparam [6:0] IA = 7'h2A;

    // 100 MHz clock.
    reg clk = 1'b0;
    always #5 clk = ~clk;

    // SPI master signals.
    reg  sck = 1'b0, cs_n = 1'b1, mosi = 1'b0;
    wire miso;
    // I2C lines: master drives open drain through m_sda_low, wired-AND with the slave on the pulled-up net.
    reg  scl = 1'b1, m_sda_low = 1'b0;
    tri1 sda;
    assign sda = m_sda_low ? 1'b0 : 1'bz;
    // AXI master signals.
    reg  [15:0] awaddr = 0, araddr = 0;
    reg         awvalid = 0, wvalid = 0, bready = 0, arvalid = 0, rready = 0;
    reg  [31:0] wdata = 0;
    reg  [3:0]  wstrb = 0;
    wire        awready, wready, bvalid, arready, rvalid;
    wire [31:0] rdata;
    wire [1:0]  bresp, rresp;
    wire [2:0]  led;

    // The device under test.
    generate
        if (BUS == 1) begin : g_spi
            tinyml_spi_top #(.W_AW(12), .X_AW(10), .Y_AW(8), .P_AW(10), .LANES(LANES), .RQ_MODE(RQ_MODE), .HAS_QUEUE(HAS_QUEUE), .D_AW(9),
                .HAS_PERF(HAS_PERF), .HAS_WIN(HAS_WIN), .HAS_FUSE(HAS_FUSE), .HAS_W4(HAS_W4), .HAS_CB(HAS_CB)) dut (
                .clk(clk), .rst(1'b0), .spi_sck(sck), .spi_cs_n(cs_n), .spi_mosi(mosi), .spi_miso(miso), .status(led));
        end else if (BUS == 2) begin : g_i2c
            tinyml_i2c_top #(.I2C_ADDR(IA), .W_AW(12), .X_AW(10), .Y_AW(8), .P_AW(10), .LANES(LANES), .RQ_MODE(RQ_MODE), .HAS_QUEUE(HAS_QUEUE), .D_AW(9),
                .HAS_PERF(HAS_PERF), .HAS_WIN(HAS_WIN), .HAS_FUSE(HAS_FUSE), .HAS_W4(HAS_W4), .HAS_CB(HAS_CB)) dut (
                .clk(clk), .rst(1'b0), .i2c_scl(scl), .i2c_sda(sda), .status(led));
        end else begin : g_axi
            tinyml_axil_top #(.W_AW(12), .X_AW(10), .Y_AW(8), .P_AW(10), .LANES(LANES), .RQ_MODE(RQ_MODE), .HAS_QUEUE(HAS_QUEUE), .D_AW(9),
                .HAS_PERF(HAS_PERF), .HAS_WIN(HAS_WIN), .HAS_FUSE(HAS_FUSE), .HAS_W4(HAS_W4), .HAS_CB(HAS_CB)) dut (
                .clk(clk), .rst(1'b0),
                .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
                .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
                .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
                .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
                .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
                .status(led));
        end
    endgenerate

    // ---------------------------------------------------------------- SPI master
    // One byte, MSB first, mode 0.
    task spi_byte(input [7:0] t, output [7:0] r);
        integer k;
        begin
            for (k = 7; k >= 0; k = k - 1) begin
                mosi = t[k]; repeat (HP) @(posedge clk);
                sck = 1'b1;  repeat (HP / 2) @(posedge clk);
                r[k] = miso; repeat (HP / 2) @(posedge clk);
                sck = 1'b0;
            end
        end
    endtask

    // ---------------------------------------------------------------- I2C master
    // START, or repeated START (SCL is low or high on entry).
    task i2c_start;
        begin
            m_sda_low = 1'b0; repeat (HP) @(posedge clk);
            scl = 1'b1;       repeat (HP) @(posedge clk);
            m_sda_low = 1'b1; repeat (HP) @(posedge clk);
            scl = 1'b0;       repeat (HP) @(posedge clk);
        end
    endtask
    // STOP.
    task i2c_stop;
        begin
            m_sda_low = 1'b1; repeat (HP) @(posedge clk);
            scl = 1'b1;       repeat (HP) @(posedge clk);
            m_sda_low = 1'b0; repeat (2 * HP) @(posedge clk);
        end
    endtask
    // Write one byte, return the slave's acknowledge (0 = ACK).
    task i2c_wbyte(input [7:0] b, output nack);
        integer k;
        begin
            for (k = 7; k >= 0; k = k - 1) begin
                m_sda_low = ~b[k]; repeat (HP) @(posedge clk);
                scl = 1'b1;        repeat (HP) @(posedge clk);
                scl = 1'b0;
            end
            // Acknowledge clock: release SDA and sample.
            m_sda_low = 1'b0; repeat (HP) @(posedge clk);
            scl = 1'b1;       repeat (HP / 2) @(posedge clk);
            nack = sda;       repeat (HP / 2) @(posedge clk);
            scl = 1'b0;       repeat (HP) @(posedge clk);
        end
    endtask
    // Read one byte; the master acknowledges it unless last is set.
    task i2c_rbyte(input last, output [7:0] b);
        integer k;
        begin
            m_sda_low = 1'b0;
            for (k = 7; k >= 0; k = k - 1) begin
                repeat (HP) @(posedge clk);
                scl = 1'b1;  repeat (HP / 2) @(posedge clk);
                b[k] = sda;  repeat (HP / 2) @(posedge clk);
                scl = 1'b0;
            end
            // Master's acknowledge (low) or NACK (released).
            m_sda_low = ~last; repeat (HP) @(posedge clk);
            scl = 1'b1;        repeat (HP) @(posedge clk);
            scl = 1'b0;        repeat (HP / 2) @(posedge clk);
            m_sda_low = 1'b0;  repeat (HP / 2) @(posedge clk);
        end
    endtask

    // ---------------------------------------------------------------- AXI master
    // One 32-bit write.
    task axi_w32(input [15:0] ad, input [31:0] d, input [3:0] s);
        begin
            awaddr = ad; awvalid = 1'b1; wdata = d; wstrb = s; wvalid = 1'b1; bready = 1'b1;
            @(posedge clk);
            while (!(awready && wready)) @(posedge clk);
            awvalid = 1'b0; wvalid = 1'b0;
            while (!bvalid) @(posedge clk);
            @(posedge clk);
            bready = 1'b0;
        end
    endtask
    // One 32-bit read.
    reg [31:0] r32;
    task axi_r32(input [15:0] ad);
        begin
            araddr = ad; arvalid = 1'b1; rready = 1'b1;
            @(posedge clk);
            while (!arready) @(posedge clk);
            arvalid = 1'b0;
            while (!rvalid) @(posedge clk);
            r32 = rdata;
            @(posedge clk);
            rready = 1'b0;
        end
    endtask

    // ---------------------------------------------------------------- generic frames
    // Data to write and data read back.
    reg [7:0] wb [0:127];
    reg [7:0] rb [0:127];
    // Scratch.
    reg [7:0] t8;
    reg       nk;
    reg [15:0] wa;
    reg [3:0]  ws;
    reg [31:0] wd;
    integer    j, e, lastw;

    // Write n bytes of wb[] at ad.
    task bwrite(input [15:0] ad, input integer n);
        integer i;
        begin
            case (BUS)
                1: begin
                    cs_n = 1'b0; repeat (HP) @(posedge clk);
                    spi_byte(n - 1, t8); spi_byte(ad[15:8], t8); spi_byte(ad[7:0], t8);
                    for (i = 0; i < n; i = i + 1) spi_byte(wb[i], t8);
                    repeat (HP) @(posedge clk);
                    cs_n = 1'b1; repeat (4 * HP) @(posedge clk);
                end
                2: begin
                    i2c_start; i2c_wbyte({IA, 1'b0}, nk);
                    if (nk) begin $display("FAIL i2c: address not acknowledged"); e = e + 1; end
                    i2c_wbyte(ad[15:8], nk); i2c_wbyte(ad[7:0], nk);
                    for (i = 0; i < n; i = i + 1) i2c_wbyte(wb[i], nk);
                    i2c_stop;
                end
                default: begin
                    // Group the bytes by aligned word.
                    i = 0;
                    while (i < n) begin
                        wa = ad + i; wa[1:0] = 2'b00; ws = 4'b0000; wd = 32'd0;
                        while (i < n && ((ad + i) >> 2) == (wa >> 2)) begin
                            ws[(ad + i) & 3] = 1'b1;
                            wd[8 * ((ad + i) & 3) +: 8] = wb[i];
                            i = i + 1;
                        end
                        axi_w32(wa, wd, ws);
                    end
                end
            endcase
        end
    endtask

    // Read n bytes at ad into rb[].
    task bread(input [15:0] ad, input integer n);
        integer i;
        begin
            case (BUS)
                1: begin
                    cs_n = 1'b0; repeat (HP) @(posedge clk);
                    spi_byte(8'h80 | (n - 1), t8); spi_byte(ad[15:8], t8); spi_byte(ad[7:0], t8); spi_byte(8'h00, t8);
                    for (i = 0; i < n; i = i + 1) spi_byte(8'h00, rb[i]);
                    repeat (HP) @(posedge clk);
                    cs_n = 1'b1; repeat (4 * HP) @(posedge clk);
                end
                2: begin
                    i2c_start; i2c_wbyte({IA, 1'b0}, nk); i2c_wbyte(ad[15:8], nk); i2c_wbyte(ad[7:0], nk);
                    i2c_start; i2c_wbyte({IA, 1'b1}, nk);
                    for (i = 0; i < n; i = i + 1) i2c_rbyte(i == n - 1, rb[i]);
                    i2c_stop;
                end
                default: begin
                    for (i = 0; i < n; i = i + 1) begin
                        if (i == 0 || ((ad + i) & 3) == 0) axi_r32(ad + i);
                        rb[i] = r32[8 * ((ad + i) & 3) +: 8];
                    end
                end
            endcase
        end
    endtask

    // ---------------------------------------------------------------- replay
    reg  [31:0] vec [0:524287];
    integer pc, errors, checks, frames, n, i, polls;
    reg [15:0] a;

    initial begin
        $readmemh("system_vectors.hex", vec);
        errors = 0; checks = 0; frames = 0; pc = 0; e = 0;
        repeat (400) @(posedge clk);
        while (vec[pc][31:24] != 8'hFF) begin
            a = vec[pc][23:8];
            n = vec[pc][7:0] + 1;
            frames = frames + 1;
            case (vec[pc][31:24])
                8'h10: begin
                    for (i = 0; i < n; i = i + 1) wb[i] = vec[pc + 1 + i][7:0];
                    bwrite(a, n);
                    pc = pc + n;
                end
                8'h11: begin
                    bread(a, n);
                    for (i = 0; i < n; i = i + 1) begin
                        checks = checks + 1;
                        if (rb[i] !== vec[pc + 1 + i][7:0]) begin
                            if (errors < 20) $display("FAIL frame %0d addr %h: got %h expected %h", frames, a + i, rb[i], vec[pc + 1 + i][7:0]);
                            errors = errors + 1;
                        end
                    end
                    pc = pc + n;
                end
                8'h12: begin
                    polls = 0; rb[0] = 8'h00;
                    while (!rb[0][1] && polls < 100000) begin bread(16'h0120, 1); polls = polls + 1; end
                    checks = checks + 1;
                    if (rb[0] !== 8'h02) begin $display("FAIL frame %0d: status %h", frames, rb[0]); errors = errors + 1; end
                end
                default: begin $display("FAIL bad word %h at %0d", vec[pc], pc); errors = errors + 1; end
            endcase
            pc = pc + 1;
        end
        errors = errors + e;
        if (errors == 0) $display("PASS tb_bus[%0d]: %0d frames, %0d bytes checked", BUS, frames, checks);
        else             $display("FAIL tb_bus[%0d]: %0d errors in %0d checks", BUS, errors, checks);
        $finish;
    end

    // Watchdog.
    initial begin #4000000000; $display("FAIL tb_bus[%0d]: timeout", BUS); $finish; end

endmodule

`default_nettype wire
