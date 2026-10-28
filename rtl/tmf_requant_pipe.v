// tmf_requant_pipe.v - exact TFLite int8 requantisation, fully pipelined (one result per clock).
//   y = clamp( ((acc * mult + 2^(ts-1)) >>> ts) + zp , act_min, act_max ),   ts = 31 - shift
// Same function as tmf_requant.v (bit-exact), but built from four unsigned 16x16 multipliers and a two-step
// barrel shifter, so a new operand set can enter every clock. Latency is LAT clocks from start to done.
// Use it when several output lanes finish together (LANES > 1) or the layers are short.
// The sign is handled separately: |acc| * mult is formed from unsigned 16-bit halves,
//   |acc| = ah*2^16 + al, mult = mh*2^16 + ml:   |acc|*mult = ah*mh*2^32 + (ah*ml + al*mh)*2^16 + al*ml,
// then negated again if acc was negative, so every multiplier is a plain unsigned 16x16 (one DSP block).

`default_nettype none

module tmf_requant_pipe (
    input  wire               clk,      // clock
    input  wire               rst,      // synchronous reset, active high
    input  wire               start,    // operands valid this clock (one operand set per clock is accepted)
    input  wire signed [31:0] acc,      // int32 accumulator
    input  wire        [31:0] mult,     // fixed-point multiplier (0 .. 2^31-1)
    input  wire signed [7:0]  shift,    // TFLite shift (-31 .. 30)
    input  wire signed [7:0]  zp,       // output zero point
    input  wire signed [7:0]  act_min,  // clamp low
    input  wire signed [7:0]  act_max,  // clamp high
    output wire               ready,    // always 1: a new operand set can enter every clock
    output reg                done,     // result valid this clock
    output reg  signed [7:0]  y         // int8 result
);

    // Number of clocks between start and done.
    localparam LAT = 10;

    // The unit never stalls.
    assign ready = 1'b1;

    // Side data per operand set: {ts[6:0], zp, act_min, act_max}, carried down the pipeline.
    reg [30:0] sd [1:9];
    // Valid bits, one per stage 1..9.
    reg [8:0]  v;
    // Right-shift distance 31 - shift (1 .. 62), as in tmf_requant.v.
    wire [6:0] ts_c = 7'd31 - {{1{shift[7]}}, shift[5:0]};

    // Stage 1: magnitude and sign of the accumulator, multiplier.
    reg        [31:0] a1;
    reg        [31:0] m1;
    reg               sg1;
    // Stage 2: the four unsigned partial products.
    reg        [31:0] pa2, pb2, pc2, pd2;
    reg               sg2;
    // Stage 3: middle products added.
    reg        [32:0] m3;
    reg        [31:0] pa3, pd3;
    reg               sg3;
    // Stage 4: low and middle parts combined.
    reg        [63:0] t4;
    reg        [31:0] pd4;
    reg               sg4;
    // Stage 5: full magnitude product.
    reg        [63:0] p5;
    reg               sg5;
    // Stage 6: product with its sign restored.
    reg signed [63:0] q6;
    // Stage 7: plus the rounding constant 2^(ts-1).
    reg signed [63:0] r7;
    // Stage 8: shifted right by 8 * ts[5:3].
    reg signed [63:0] r8;
    // Stage 9: shifted right by ts[2:0].
    reg signed [63:0] r9;
    // Rounding constant for stage 7 (ts is at least 1).
    wire signed [63:0] rc6 = 64'sd1 <<< (sd[6][30:24] - 7'd1);
    // Stage 10 sum with the zero point (combinational view of r9).
    wire signed [63:0] s9 = r9 + $signed({{56{sd[9][23]}}, sd[9][23:16]});
    // The sum fits in int8.
    wire               fits8 = (s9[63:7] == {57{s9[63]}});
    // Clamp tests for values inside int8 range.
    wire               below = $signed(s9[7:0]) < $signed(sd[9][15:8]);
    wire               above = $signed(s9[7:0]) > $signed(sd[9][7:0]);

    // Loop index for the side-data shift.
    integer i;
    // Pipeline: one stage per clock, all stages advance every clock.
    always @(posedge clk) begin
        // Side data enters, then moves down one stage per clock.
        sd[1] <= {ts_c, zp, act_min, act_max};
        // Shift the rest.
        for (i = 2; i <= 9; i = i + 1) sd[i] <= sd[i-1];
        // Stage 1: magnitude and sign (|-2^31| = 2^31 still fits 32 unsigned bits).
        a1  <= acc[31] ? (~acc + 32'd1) : acc;
        // Sign.
        sg1 <= acc[31];
        // Multiplier.
        m1  <= mult;
        // Stage 2: the four 16x16 unsigned products.
        pa2 <= a1[15:0]  * m1[15:0];
        pb2 <= a1[15:0]  * m1[31:16];
        pc2 <= a1[31:16] * m1[15:0];
        pd2 <= a1[31:16] * m1[31:16];
        // Sign moves on.
        sg2 <= sg1;
        // Stage 3: add the two middle products; pass the others.
        m3  <= {1'b0, pb2} + {1'b0, pc2};
        pa3 <= pa2;
        pd3 <= pd2;
        sg3 <= sg2;
        // Stage 4: low product plus the middle product shifted by 16.
        t4  <= {32'd0, pa3} + ({31'd0, m3} << 16);
        pd4 <= pd3;
        sg4 <= sg3;
        // Stage 5: add the high product shifted by 32.
        p5  <= t4 + ({32'd0, pd4} << 32);
        sg5 <= sg4;
        // Stage 6: restore the sign.
        q6  <= sg5 ? -$signed(p5) : $signed(p5);
        // Stage 7: rounding constant.
        r7  <= q6 + rc6;
        // Stage 8: coarse arithmetic shift by 0, 8, ... 56.
        r8  <= r7 >>> {sd[7][29:27], 3'b000};
        // Stage 9: fine arithmetic shift by 0 .. 7.
        r9  <= r8 >>> sd[8][26:24];
        // Valid bits march through; reset clears them.
        v   <= rst ? 9'd0 : {v[7:0], start};
        // Stage 10: zero point, clamp, report.
        done <= v[8] & ~rst;
        // Far below or below the low limit.
        if ((!fits8 && s9[63]) || (fits8 && below))       y <= $signed(sd[9][15:8]);
        // Far above or above the high limit.
        else if ((!fits8 && !s9[63]) || (fits8 && above)) y <= $signed(sd[9][7:0]);
        // In range.
        else                                               y <= s9[7:0];
    end

endmodule

`default_nettype wire
