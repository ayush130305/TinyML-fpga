// tb_engine.v - self-checking testbench for tmf_engine.v.
// Replays engine_vectors.hex (written by gen_engine_vectors.py from the Python register model)
// through the byte bus with the bus timing rules (address stable before the read strobe) and
// checks every output byte, the STATUS byte and the cycle counter.

`timescale 1ns/1ps
`default_nettype none

module tb_engine #(
    parameter LANES   = 1,               // engine lanes under test (must match gen_engine_vectors.py)
    parameter RQ_MODE = 0                // requantiser under test (must match gen_engine_vectors.py)
);

    // Clock.
    reg         clk = 1'b0;
    // Reset.
    reg         rst = 1'b1;
    // Bus address.
    reg  [15:0] addr = 16'h0;
    // Bus write strobe.
    reg         wr = 1'b0;
    // Bus write data.
    reg  [7:0]  wdata = 8'h0;
    // Bus read strobe.
    reg         rd = 1'b0;
    // Bus read data.
    wire [7:0]  rdata;
    // Engine status byte.
    wire [7:0]  status;

    // Command stream.
    reg  [31:0] vec [0:262143];
    // Stream index.
    integer pc;
    // Error count.
    integer errors;
    // Checks performed.
    integer checks;
    // Jobs run.
    integer jobs;
    // Scratch read value.
    reg  [7:0]  r;
    // Cycle counter read back.
    reg  [31:0] cyc;
    // Poll counter.
    integer polls;

    // 25 MHz clock (period 40 ns).
    always #20 clk = ~clk;

    // This contains the instantiation for the device under test
    tmf_engine #(.W_AW(10), .X_AW(9), .Y_AW(8), .P_AW(8), .LANES(LANES), .RQ_MODE(RQ_MODE)) dut (
        .clk(clk), .rst(rst), .bus_addr(addr), .bus_wr(wr), .bus_wdata(wdata), .bus_rd(rd),
        .bus_rdata(rdata), .status_o(status));

    // Bus write of one byte: address and data first, then a one-clock strobe.
    task bus_write(input [15:0] a, input [7:0] d);
        begin
            // Present address and data.
            @(negedge clk); addr = a; wdata = d;
            // Strobe for one clock.
            @(negedge clk); wr = 1'b1;
            // Release.
            @(negedge clk); wr = 1'b0;
        end
    endtask

    // Bus read of one byte: address held two clocks, sample in the strobe clock (bus timing rules).
    task bus_read(input [15:0] a, output [7:0] d);
        begin
            // Present the address.
            @(negedge clk); addr = a;
            // Let the registered RAM output settle.
            @(negedge clk); rd = 1'b1;
            // Sample in the strobe clock.
            d = rdata;
            // Release.
            @(negedge clk); rd = 1'b0;
        end
    endtask

    // Main sequence.
    initial begin
        // Load the stream.
        $readmemh("engine_vectors.hex", vec);
        // Counters.
        errors = 0; checks = 0; jobs = 0; pc = 0;
        // Hold reset a few clocks.
        repeat (4) @(negedge clk);
        rst = 1'b0;
        // Walk the stream.
        while (vec[pc][31:24] != 8'hFF) begin
            // Dispatch on the opcode.
            case (vec[pc][31:24])
                // Write a byte.
                8'h01: bus_write(vec[pc][23:8], vec[pc][7:0]);
                // Start and wait for done.
                8'h02: begin
                    // Start command.
                    bus_write(16'h011F, 8'h01);
                    // Poll STATUS.done (bounded).
                    polls = 0;
                    r = 8'h00;
                    while (!r[1] && polls < 200000) begin
                        bus_read(16'h0120, r);
                        polls = polls + 1;
                    end
                    // Count the job.
                    jobs = jobs + 1;
                    // Hung job.
                    if (!r[1]) begin
                        $display("FAIL job %0d: never finished", jobs);
                        errors = errors + 1;
                    end
                end
                // Expect a byte.
                8'h03: begin
                    // Read it.
                    bus_read(vec[pc][23:8], r);
                    // Count the check.
                    checks = checks + 1;
                    // Compare.
                    if (r !== vec[pc][7:0]) begin
                        if (errors < 20) $display("FAIL job %0d addr %h: got %h expected %h", jobs, vec[pc][23:8], r, vec[pc][7:0]);
                        errors = errors + 1;
                    end
                end
                // Expect STATUS.
                8'h04: begin
                    // Read it.
                    bus_read(16'h0120, r);
                    // Count the check.
                    checks = checks + 1;
                    // Compare.
                    if (r !== vec[pc][7:0]) begin
                        $display("FAIL job %0d status: got %h expected %h", jobs, r, vec[pc][7:0]);
                        errors = errors + 1;
                    end
                end
                // Expect CYCLES (next word).
                8'h05: begin
                    // Four bytes, little endian.
                    bus_read(16'h0121, cyc[7:0]);
                    bus_read(16'h0122, cyc[15:8]);
                    bus_read(16'h0123, cyc[23:16]);
                    bus_read(16'h0124, cyc[31:24]);
                    // Skip to the value word.
                    pc = pc + 1;
                    // Count the check.
                    checks = checks + 1;
                    // Compare.
                    if (cyc !== vec[pc]) begin
                        if (errors < 20) $display("FAIL job %0d cycles: got %0d expected %0d", jobs, cyc, vec[pc]);
                        errors = errors + 1;
                    end
                end
                // Unknown opcode.
                default: begin
                    $display("FAIL bad vector word %h at %0d", vec[pc], pc);
                    errors = errors + 1;
                end
            endcase
            // Next word.
            pc = pc + 1;
        end
        // Verdict.
        if (errors == 0) $display("PASS tb_engine (lanes %0d, rq_mode %0d): %0d jobs, %0d checks", LANES, RQ_MODE, jobs, checks);
        else             $display("FAIL tb_engine (lanes %0d, rq_mode %0d): %0d errors in %0d checks", LANES, RQ_MODE, errors, checks);
        $finish;
    end

endmodule

`default_nettype wire
