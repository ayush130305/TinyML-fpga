// tmf_lane.v - one MAC lane of the engine: parameter shift register, multiply stage and accumulator.
// Several lanes work on consecutive output channels at once. All lanes see the same input byte (the
// engine's X stream) and each reads its own weight byte and its own parameter bytes from its RAM bank.
// The multiply is 3-stage pipelined together with the RAM read (stage 1 = RAM output, 2 = product, 3 = accumulate).

`default_nettype none

module tmf_lane (
    input  wire               clk,      // clock
    input  wire               par_en,   // shift one parameter byte in (LSB first)
    input  wire        [7:0]  p_byte,   // parameter byte from this lane's P bank
    input  wire               acc_init, // load the accumulator with the bias
    input  wire               v1,       // pipeline stage 1 valid: RAM data available
    input  wire               v2,       // pipeline stage 2 valid: product available
    input  wire signed [8:0]  xoff,     // input byte plus the input offset (shared by all lanes)
    input  wire        [7:0]  w_byte,   // weight byte from this lane's W bank
    output wire signed [31:0] acc_o,    // accumulator
    output wire        [31:0] mult_o,   // fixed-point multiplier of the channel
    output wire        [7:0]  shift_o   // shift of the channel
);

    // Parameter bytes, shifted in LSB first: bias[31:0], mult[63:32], shift[71:64].
    reg  [71:0]        par;
    // Accumulator.
    reg  signed [31:0] acc;
    // Pipeline product register.
    reg  signed [17:0] prod;
    // Signed 9 x 8 product.
    wire signed [16:0] mul = xoff * $signed(w_byte);

    // Outputs.
    assign acc_o   = acc;
    assign mult_o  = par[63:32];
    assign shift_o = par[71:64];

    // Lane datapath: parameter capture, product, accumulation.
    always @(posedge clk) begin
        // Shift the addressed parameter byte in.
        if (par_en)   par  <= {p_byte, par[71:8]};
        // Stage 2: (x + in_off) * w.
        if (v1)       prod <= {mul[16], mul};
        // Stage 3: accumulate the product.
        if (v2)       acc  <= acc + {{14{prod[17]}}, prod};
        // Start every output from its bias (never overlaps v2).
        if (acc_init) acc  <= par[31:0];
    end

endmodule

`default_nettype wire
