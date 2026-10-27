// tb_uart_top.v - end-to-end testbench for the UART path (tinyml_uart_top = uart_bus + engine).
// Replays system_vectors.hex (a recorded real model inference) through a UART BFM,
// so the bridge, the engine and the PC-side frame format (tinyml_fpga.buses.UartBus) are all checked.
// The engine is built with the default RAM sizes because the recorded stream reads them back.

`timescale 1ns/1ps
`default_nettype none

module tb_uart_top;

    // Clocks per UART bit in this simulation (100 MHz / 12.5 Mbaud).
    localparam CPB = 8;

    // 100 MHz clock.
    reg        clk = 1'b0;
    // Host -> FPGA serial line.
    reg        rx = 1'b1;
    // FPGA -> host serial line.
    wire       tx;
    // LEDs.
    wire [2:0] led;

    // Command stream.
    reg  [31:0] vec [0:524287];
    // Stream index.
    integer pc;
    // Counters.
    integer errors, checks, frames;
    // Loop indices.
    integer i, n, polls;
    // Received byte.
    reg  [7:0] r;
    // Frame address.
    reg [15:0] a;

    // Clock: period 10 ns.
    always #5 clk = ~clk;

    // This contains the instantiation for the board top
    tinyml_uart_top #(.CLK_HZ(100000000), .BAUD(12500000), .W_AW(12), .X_AW(10), .Y_AW(8), .P_AW(10)) dut (
        .clk(clk), .rst(1'b0), .uart_rx(rx), .uart_tx(tx), .status(led));

    // Send one byte to the FPGA (start bit, 8 data bits LSB first, stop bit).
    task usend(input [7:0] b);
        integer k;
        begin
            // Start bit.
            rx = 1'b0; repeat (CPB) @(posedge clk);
            // Data bits.
            for (k = 0; k < 8; k = k + 1) begin rx = b[k]; repeat (CPB) @(posedge clk); end
            // Stop bit.
            rx = 1'b1; repeat (CPB) @(posedge clk);
        end
    endtask

    // Receive one byte from the FPGA (with a timeout).
    task urecv(output [7:0] b);
        integer k, t;
        begin
            // Wait for the start bit.
            t = 0;
            while (tx === 1'b1 && t < 200000) begin @(posedge clk); t = t + 1; end
            // Middle of the start bit.
            repeat (CPB / 2) @(posedge clk);
            // Data bits.
            for (k = 0; k < 8; k = k + 1) begin repeat (CPB) @(posedge clk); b[k] = tx; end
            // Stop bit.
            repeat (CPB) @(posedge clk);
            // Timeout.
            if (t >= 200000) begin b = 8'hXX; $display("FAIL uart receive timeout"); end
        end
    endtask

    // Read k bytes at ad into rb[].
    reg [7:0] rb [0:127];
    // UART read frame.
    task uread(input [15:0] ad, input integer k);
        integer j;
        begin
            // Header.
            usend(8'h80 | (k - 1)); usend(ad[15:8]); usend(ad[7:0]);
            // Data back.
            for (j = 0; j < k; j = j + 1) urecv(rb[j]);
        end
    endtask

    // Main sequence.
    initial begin
        // Load the stream.
        $readmemh("system_vectors.hex", vec);
        // Counters.
        errors = 0; checks = 0; frames = 0; pc = 0;
        // Power-on reset.
        repeat (300) @(posedge clk);
        // Walk the stream.
        while (vec[pc][31:24] != 8'hFF) begin
            // Frame address and length.
            a = vec[pc][23:8];
            n = vec[pc][7:0] + 1;
            // Count frames.
            frames = frames + 1;
            // Dispatch.
            case (vec[pc][31:24])
                // Write frame, then the acknowledge.
                8'h10: begin
                    // Header.
                    usend(8'h00 | (n - 1)); usend(a[15:8]); usend(a[7:0]);
                    // Data.
                    for (i = 1; i <= n; i = i + 1) usend(vec[pc + i][7:0]);
                    // Acknowledge.
                    urecv(r);
                    checks = checks + 1;
                    if (r !== 8'h55) begin
                        $display("FAIL frame %0d: write ack %h", frames, r);
                        errors = errors + 1;
                    end
                    // Skip the data words.
                    pc = pc + n;
                end
                // Read frame with expected bytes.
                8'h11: begin
                    // Read.
                    uread(a, n);
                    // Compare.
                    for (i = 0; i < n; i = i + 1) begin
                        checks = checks + 1;
                        if (rb[i] !== vec[pc + 1 + i][7:0]) begin
                            if (errors < 20) $display("FAIL frame %0d addr %h: got %h expected %h", frames, a + i, rb[i], vec[pc + 1 + i][7:0]);
                            errors = errors + 1;
                        end
                    end
                    // Skip the expected words.
                    pc = pc + n;
                end
                // Poll STATUS.
                8'h12: begin
                    // Bounded polling.
                    polls = 0;
                    rb[0] = 8'h00;
                    while (!rb[0][1] && polls < 100000) begin
                        uread(16'h0120, 1);
                        polls = polls + 1;
                    end
                    // Done without error.
                    checks = checks + 1;
                    if (rb[0] !== 8'h02) begin
                        $display("FAIL frame %0d: status %h", frames, rb[0]);
                        errors = errors + 1;
                    end
                end
                // Unknown opcode.
                default: begin
                    $display("FAIL bad word %h at %0d", vec[pc], pc);
                    errors = errors + 1;
                end
            endcase
            // Next word.
            pc = pc + 1;
        end
        // Verdict.
        if (errors == 0) $display("PASS tb_uart_top: %0d frames, %0d bytes checked", frames, checks);
        else             $display("FAIL tb_uart_top: %0d errors in %0d checks", errors, checks);
        $finish;
    end

endmodule

`default_nettype wire
