// tb_requant.v - self-checking testbench for both requantisers (tmf_rq MODE 0 and MODE 1).
// Replays requant_vectors.hex (written by gen_requant_vectors.py from the Python reference): one case is
// 4 words acc, mult, {shift,zp,lo,hi}, expected y. Mode 0 takes one case at a time; mode 1 gets a new
// case every clock (or with gaps) and must return the results in order, LAT clocks later.

`timescale 1ns/1ps
`default_nettype none

module tb_requant;

    // Clock.
    reg clk = 1'b0;
    // Reset.
    reg rst = 1'b1;
    // 100 MHz clock.
    always #5 clk = ~clk;

    // Case memory: 4 words per case.
    reg [31:0] mem [0:4*65536-1];
    // Number of cases (first word).
    integer ncase;
    // Error count and checks.
    integer errors, checks;

    // Operand bus shared by both devices.
    reg               st0, st1;
    reg signed [31:0] acc;
    reg        [31:0] mult;
    reg signed [7:0]  shift, zp, lo, hi;
    // Device outputs.
    wire              rdy0, rdy1, dn0, dn1;
    wire signed [7:0] y0, y1;

    // This contains the instantiation for the serial requantiser under test
    tmf_rq #(.MODE(0)) d0 (.clk(clk), .rst(rst), .start(st0), .acc(acc), .mult(mult), .shift(shift), .zp(zp),
                           .act_min(lo), .act_max(hi), .ready(rdy0), .done(dn0), .y(y0));
    // This contains the instantiation for the pipelined requantiser under test
    tmf_rq #(.MODE(1)) d1 (.clk(clk), .rst(rst), .start(st1), .acc(acc), .mult(mult), .shift(shift), .zp(zp),
                           .act_min(lo), .act_max(hi), .ready(rdy1), .done(dn1), .y(y1));

    // Expected results queue for the pipelined device.
    reg signed [7:0] expq [0:65535];
    // Queue pointers.
    integer wq, rq1;
    // Case index being issued to mode 0 / mode 1.
    integer i0, i1;
    // Gap pattern: mode 1 inserts idle clocks between some cases.
    integer gap;
    // Clock counter.
    integer cyc;

    // Check the pipelined results as they arrive.
    always @(posedge clk) if (!rst && dn1) begin
        // Compare with the expected value in order.
        if (y1 !== expq[rq1]) begin
            if (errors < 10) $display("FAIL pipe case %0d: got %0d expected %0d", rq1, y1, expq[rq1]);
            errors = errors + 1;
        end
        // Count.
        checks = checks + 1; rq1 = rq1 + 1;
    end

    // Main sequence.
    initial begin
        // Load vectors.
        $readmemh("requant_vectors.hex", mem);
        // Case count.
        ncase = mem[0];
        // Counters.
        errors = 0; checks = 0; wq = 0; rq1 = 0; st0 = 0; st1 = 0; gap = 0;
        // Reset.
        repeat (4) @(posedge clk);
        rst = 1'b0;
        // ---- mode 0, one case at a time ----
        for (i0 = 0; i0 < ncase; i0 = i0 + 1) begin
            // Present operands on the falling edge and pulse start for one clock.
            @(negedge clk);
            acc = mem[1+4*i0]; mult = mem[2+4*i0]; {shift, zp, lo, hi} = mem[3+4*i0];
            // Wait until ready.
            while (!rdy0) @(negedge clk);
            st0 = 1'b1; @(negedge clk); st0 = 1'b0;
            // Wait for done.
            cyc = 0;
            while (!dn0 && cyc < 200) begin @(negedge clk); cyc = cyc + 1; end
            // Compare (done is a registered pulse, y valid in the same clock).
            if (y0 !== $signed(mem[4+4*i0][7:0])) begin
                if (errors < 10) $display("FAIL serial case %0d: got %0d expected %0d", i0, y0, $signed(mem[4+4*i0][7:0]));
                errors = errors + 1;
            end
            checks = checks + 1;
        end
        // ---- mode 1, streamed with some gaps ----
        for (i1 = 0; i1 < ncase; i1 = i1 + 1) begin
            // New operands each clock (falling edge), occasionally a gap clock first.
            @(negedge clk);
            if (i1 % 7 == 3) begin st1 = 1'b0; @(negedge clk); end
            acc = mem[1+4*i1]; mult = mem[2+4*i1]; {shift, zp, lo, hi} = mem[3+4*i1];
            expq[wq] = mem[4+4*i1][7:0]; wq = wq + 1;
            st1 = 1'b1;
        end
        // Drain the pipeline.
        @(negedge clk); st1 = 1'b0;
        repeat (20) @(negedge clk);
        // Verdict.
        if (rq1 != ncase) begin
            $display("FAIL pipe returned %0d results for %0d cases", rq1, ncase);
            errors = errors + 1;
        end
        if (errors == 0) $display("PASS tb_requant: %0d cases x 2 requantisers, %0d checks", ncase, checks);
        else             $display("FAIL tb_requant: %0d errors", errors);
        $finish;
    end

endmodule

`default_nettype wire
