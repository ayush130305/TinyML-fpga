// tmf_requant.v - exact TFLite int8 requantisation, sequential (no hardware multiplier needed).
//   y = clamp( ((acc * mult + 2^(ts-1)) >>> ts) + zp , act_min, act_max ),   ts = 31 - shift
// This is TFLite's single-rounding MultiplyByQuantizedMultiplier, bit-exact with LiteRT.
// The 32x31 multiply is shift-and-add (32 clocks); the variable right shift is done in steps of
// 8 and then 1 bit (at most 7 + 7 clocks). One 64-bit adder is shared by the multiply, the rounding
// constant and the zero point. Clocks from start to done: 1 + 32 + 1 + (ts/8 + ts%8 + 1) + 1 + 1.

`default_nettype none

module tmf_requant (
    input  wire               clk,      // clock
    input  wire               rst,      // synchronous reset, active high
    input  wire               start,    // one-clock pulse: latch the inputs and begin
    input  wire signed [31:0] acc,      // int32 accumulator
    input  wire        [31:0] mult,     // fixed-point multiplier (0 .. 2^31-1)
    input  wire signed [7:0]  shift,    // TFLite shift (-31 .. 30)
    input  wire signed [7:0]  zp,       // output zero point
    input  wire signed [7:0]  act_min,  // clamp low
    input  wire signed [7:0]  act_max,  // clamp high
    output reg                done,     // one-clock pulse: y is valid
    output reg  signed [7:0]  y         // int8 result
);

    // Phase encodings.
    localparam [2:0] P_IDLE = 3'd0;     // waiting for start
    localparam [2:0] P_MUL  = 3'd1;     // shift-and-add multiply
    localparam [2:0] P_RND  = 3'd2;     // add the rounding constant
    localparam [2:0] P_SHR  = 3'd3;     // arithmetic right shift by ts
    localparam [2:0] P_ZP   = 3'd4;     // add the zero point
    localparam [2:0] P_OUT  = 3'd5;     // clamp, report

    // Current phase.
    reg [2:0]         ph;
    // Running 64-bit product / sum.
    reg signed [63:0] p;
    // Multiplicand, shifted left each step (sign-extended accumulator).
    reg signed [63:0] mc;
    // Multiplier bits still to consume.
    reg [31:0]        ml;
    // Multiply step counter.
    reg [5:0]         n;
    // Remaining right-shift distance.
    reg [6:0]         ts;
    // Latched zero point and clamp limits.
    reg signed [7:0]  zp_q, lo_q, hi_q;
    // Second adder operand: multiplicand (multiply), rounding constant, or zero point.
    wire signed [63:0] addend = (ph == P_MUL) ? (ml[0] ? mc : 64'sd0)
                              : (ph == P_RND) ? (64'sd1 <<< (ts - 7'd1))
                              :                 $signed({{56{zp_q[7]}}, zp_q});
    // The one shared 64-bit adder.
    wire signed [63:0] sum = p + addend;
    // p fits in int8 (bits 63..7 all equal the sign).
    wire               fits8 = (p[63:7] == {57{p[63]}});
    // Clamp test inside int8: below the low limit.
    wire               below = $signed(p[7:0]) < lo_q;
    // Clamp test inside int8: above the high limit.
    wire               above = $signed(p[7:0]) > hi_q;

    // Requantiser sequencer: one phase step per clock.
    always @(posedge clk) begin
        // done is a single-clock pulse by default.
        done <= 1'b0;
        // Reset returns to idle.
        if (rst) begin
            // Idle phase.
            ph <= P_IDLE;
        end else begin
            // Phase dispatch.
            case (ph)
                // Wait for a start pulse and latch the operands.
                P_IDLE: if (start) begin
                    // Clear the product.
                    p    <= 64'sd0;
                    // Sign-extended multiplicand.
                    mc   <= {{32{acc[31]}}, acc};
                    // Multiplier bits.
                    ml   <= mult;
                    // 32 multiply steps.
                    n    <= 6'd0;
                    // Total right shift 31 - shift (1 .. 62).
                    ts   <= 7'd31 - {{1{shift[7]}}, shift[5:0]};
                    // Keep the output parameters.
                    zp_q <= zp;
                    // Keep the clamp low limit.
                    lo_q <= act_min;
                    // Keep the clamp high limit.
                    hi_q <= act_max;
                    // Start multiplying.
                    ph   <= P_MUL;
                end
                // One multiplier bit per clock.
                P_MUL: begin
                    // Add the shifted multiplicand where the multiplier bit is 1 (addend is 0 otherwise).
                    p  <= sum;
                    // Next multiplicand weight.
                    mc <= mc <<< 1;
                    // Next multiplier bit.
                    ml <= ml >> 1;
                    // Count the step.
                    n  <= n + 6'd1;
                    // After 32 steps the product is complete.
                    if (n == 6'd31) ph <= P_RND;
                end
                // Rounding: add 2^(ts-1).
                P_RND: begin
                    // Half of the output step.
                    p  <= sum;
                    // Shift next.
                    ph <= P_SHR;
                end
                // Arithmetic right shift, 8 bits at a time while possible, then 1 bit.
                P_SHR: begin
                    // Big steps first.
                    if (ts >= 7'd8) begin
                        // Shift by 8.
                        p  <= p >>> 8;
                        // 8 fewer to go.
                        ts <= ts - 7'd8;
                    end else if (ts != 7'd0) begin
                        // Shift by 1.
                        p  <= p >>> 1;
                        // One fewer to go.
                        ts <= ts - 7'd1;
                    end else begin
                        // Shift finished.
                        ph <= P_ZP;
                    end
                end
                // Add the output zero point.
                P_ZP: begin
                    // p + zp.
                    p  <= sum;
                    // Clamp next.
                    ph <= P_OUT;
                end
                // Clamp and report: outside int8 the sign decides, inside compare 8-bit values.
                P_OUT: begin
                    // Far below or below the low limit.
                    if ((!fits8 && p[63]) || (fits8 && below))       y <= lo_q;
                    // Far above or above the high limit.
                    else if ((!fits8 && !p[63]) || (fits8 && above)) y <= hi_q;
                    // In range.
                    else                                              y <= p[7:0];
                    // Result valid next clock.
                    done <= 1'b1;
                    // Back to idle.
                    ph   <= P_IDLE;
                end
                // Unused encodings recover to idle.
                default: ph <= P_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
