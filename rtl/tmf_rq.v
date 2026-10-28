// tmf_rq.v - requantiser selector: one interface, two implementations (parameter MODE).
//   MODE 0: tmf_requant.v       bit-serial, no multiplier, about 38 + ts/8 + ts%8 clocks per result, one at a time
//   MODE 1: tmf_requant_pipe.v  four 17x17 multipliers, fixed latency, one result per clock
// Both give identical results for every input (the testbench tb_requant.v checks that).
// Issue a result with `start` while `ready` is high; `done` pulses when `y` is valid. Results come out in order.

`default_nettype none

module tmf_rq #(
    parameter MODE = 0                   // 0 serial (smallest), 1 pipelined (fastest)
) (
    input  wire               clk,       // clock
    input  wire               rst,       // synchronous reset, active high
    input  wire               start,     // issue one operand set (only while ready)
    input  wire signed [31:0] acc,       // int32 accumulator
    input  wire        [31:0] mult,      // fixed-point multiplier
    input  wire signed [7:0]  shift,     // TFLite shift
    input  wire signed [7:0]  zp,        // output zero point
    input  wire signed [7:0]  act_min,   // clamp low
    input  wire signed [7:0]  act_max,   // clamp high
    output wire               ready,     // a new operand set can be issued
    output wire               done,      // result valid this clock
    output wire signed [7:0]  y          // int8 result
);

    // Pick the implementation at elaboration time.
    generate
        if (MODE == 0) begin : g_serial
            // This contains the instantiation for the serial requantiser
            tmf_requant u_rq (.clk(clk), .rst(rst), .start(start), .acc(acc), .mult(mult), .shift(shift), .zp(zp),
                              .act_min(act_min), .act_max(act_max), .ready(ready), .done(done), .y(y));
        end else begin : g_pipe
            // This contains the instantiation for the pipelined requantiser
            tmf_requant_pipe u_rq (.clk(clk), .rst(rst), .start(start), .acc(acc), .mult(mult), .shift(shift), .zp(zp),
                                   .act_min(act_min), .act_max(act_max), .ready(ready), .done(done), .y(y));
        end
    endgenerate

endmodule

`default_nettype wire
