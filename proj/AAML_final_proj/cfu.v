/*
cmd_payload_function_id[9:3]:
0 -> Lab02
1 -> reset
2 -> logistic
3 -> softmax
4 -> TPU input
5 -> perform TPU
6 -> TPU output
7 -> reset_C_buf // delete
8 -> softmax
9 -> softmax
10 -> sum_prods
11 -> add_sums
12 -> $signed(cmd_payload_inputs_0)+$signed(cmd_payload_inputs_1)
*/

module Cfu (
    input               cmd_valid,
    output              cmd_ready,
    input      [9:0]    cmd_payload_function_id,
    input      [31:0]   cmd_payload_inputs_0,
    input      [31:0]   cmd_payload_inputs_1,
    
    output reg          rsp_valid,
    input               rsp_ready,
    output reg [31:0]   rsp_payload_outputs_0,
    input               reset,
    input               clk
);

    //////////////////////fullyconnect/////////////////////////

    localparam InputOffset = $signed(9'd128);
    // SIMD multiply step:
    wire signed [15:0] prod_0, prod_1, prod_2, prod_3;
    assign prod_0 = ($signed(cmd_payload_inputs_0[7 : 0]) + InputOffset)* $signed(cmd_payload_inputs_1[7 : 0]);
    assign prod_1 = ($signed(cmd_payload_inputs_0[15 : 8]) + InputOffset)* $signed(cmd_payload_inputs_1[15 : 8]);
    assign prod_2 = ($signed(cmd_payload_inputs_0[23 : 16]) + InputOffset)* $signed(cmd_payload_inputs_1[23 :16]);
    assign prod_3 = ($signed(cmd_payload_inputs_0[31: 24]) + InputOffset)* $signed(cmd_payload_inputs_1[31: 24]);


    wire signed [15:0] add_a, add_b, add_c, add_d;
    assign add_a = ($signed(cmd_payload_inputs_0[7 : 0])) + $signed(cmd_payload_inputs_1[7 : 0]);
    assign add_b = ($signed(cmd_payload_inputs_0[15 : 8])) + $signed(cmd_payload_inputs_1[15 : 8]);
    assign add_c = ($signed(cmd_payload_inputs_0[23 : 16])) + $signed(cmd_payload_inputs_1[23 :16]);
    assign add_d = ($signed(cmd_payload_inputs_0[31: 24])) + $signed(cmd_payload_inputs_1[31: 24]);

    // assign prod_0 = ($signed(cmd_payload_inputs_0[7 : 0]))* $signed(cmd_payload_inputs_1[7 : 0]);
    // assign prod_1 = ($signed(cmd_payload_inputs_0[15 : 8]))* $signed(cmd_payload_inputs_1[15 : 8]);
    // assign prod_2 = ($signed(cmd_payload_inputs_0[23 : 16]))* $signed(cmd_payload_inputs_1[23 :16]);
    // assign prod_3 = ($signed(cmd_payload_inputs_0[31: 24]))* $signed(cmd_payload_inputs_1[31: 24]);

    wire signed [31:0] sum_prods;
    assign sum_prods = prod_0 + prod_1 + prod_2 + prod_3;
    wire signed [31:0] add_sums;
    assign add_sums = add_a + add_b + add_c + add_d;

    
    reg [31:0] EXP_LOOKUP_TABLE [0:63];

    initial begin
        EXP_LOOKUP_TABLE[0] = 32'h7fffffff;
        EXP_LOOKUP_TABLE[1] = 32'h63AFBE7A;
        EXP_LOOKUP_TABLE[2] = 32'h4DA2CBF1;
        EXP_LOOKUP_TABLE[3] = 32'h3C7681D7;
        EXP_LOOKUP_TABLE[4] = 32'h2F16AC6C;
        EXP_LOOKUP_TABLE[5] = 32'h24AC306E;
        EXP_LOOKUP_TABLE[6] = 32'h1C8F8772;
        EXP_LOOKUP_TABLE[7] = 32'h163E397E;
        EXP_LOOKUP_TABLE[8] = 32'h1152AAA3;
        EXP_LOOKUP_TABLE[9] = 32'h0D7DB8C7;
        EXP_LOOKUP_TABLE[10] = 32'h0A81C2E0;
        EXP_LOOKUP_TABLE[11] = 32'h082EC9C4;
        EXP_LOOKUP_TABLE[12] = 32'h065F6C33;
        EXP_LOOKUP_TABLE[13] = 32'h04F68DA1;
        EXP_LOOKUP_TABLE[14] = 32'h03DD8203;
        EXP_LOOKUP_TABLE[15] = 32'h0302A126;
        EXP_LOOKUP_TABLE[16] = 32'h02582AB7;
        EXP_LOOKUP_TABLE[17] = 32'h01D36911;
        EXP_LOOKUP_TABLE[18] = 32'h016C0504;
        EXP_LOOKUP_TABLE[19] = 32'h011B7FAD;
        EXP_LOOKUP_TABLE[20] = 32'h00DCC9FF;
        EXP_LOOKUP_TABLE[21] = 32'h00ABF35F;
        EXP_LOOKUP_TABLE[22] = 32'h0085EA52;
        EXP_LOOKUP_TABLE[23] = 32'h00684B19;
        EXP_LOOKUP_TABLE[24] = 32'h00513947;
        EXP_LOOKUP_TABLE[25] = 32'h003F41D2;
        EXP_LOOKUP_TABLE[26] = 32'h003143C3;
        EXP_LOOKUP_TABLE[27] = 32'h00265E0C;
        EXP_LOOKUP_TABLE[28] = 32'h001DE16B;
        EXP_LOOKUP_TABLE[29] = 32'h0017455F;
        EXP_LOOKUP_TABLE[30] = 32'h00121F9B;
        EXP_LOOKUP_TABLE[31] = 32'h000E1D54;
        EXP_LOOKUP_TABLE[32] = 32'h000AFE10;
        EXP_LOOKUP_TABLE[33] = 32'h00088F98;
        EXP_LOOKUP_TABLE[34] = 32'h0006AAD0;
        EXP_LOOKUP_TABLE[35] = 32'h00053145;
        EXP_LOOKUP_TABLE[36] = 32'h00040B3C;
        EXP_LOOKUP_TABLE[37] = 32'h0003263E;
        EXP_LOOKUP_TABLE[38] = 32'h000273E7;
        EXP_LOOKUP_TABLE[39] = 32'h0001E902;
        EXP_LOOKUP_TABLE[40] = 32'h00017CD7;
        EXP_LOOKUP_TABLE[41] = 32'h00012899;
        EXP_LOOKUP_TABLE[42] = 32'h0000E6FE;
        EXP_LOOKUP_TABLE[43] = 32'h0000B3E5;
        EXP_LOOKUP_TABLE[44] = 32'h00008C1A;
        EXP_LOOKUP_TABLE[45] = 32'h00006D1C;
        EXP_LOOKUP_TABLE[46] = 32'h000054FA;
        EXP_LOOKUP_TABLE[47] = 32'h0000422E;
        EXP_LOOKUP_TABLE[48] = 32'h0000338A;
        EXP_LOOKUP_TABLE[49] = 32'h00002823;
        EXP_LOOKUP_TABLE[50] = 32'h00001F42;
        EXP_LOOKUP_TABLE[51] = 32'h00001858;
        EXP_LOOKUP_TABLE[52] = 32'h000012F6;
        EXP_LOOKUP_TABLE[53] = 32'h00000EC4;
        EXP_LOOKUP_TABLE[54] = 32'h00000B80;
        EXP_LOOKUP_TABLE[55] = 32'h000008F4;
        EXP_LOOKUP_TABLE[56] = 32'h000006F9;
        EXP_LOOKUP_TABLE[57] = 32'h0000056E;
        EXP_LOOKUP_TABLE[58] = 32'h0000043B;
        EXP_LOOKUP_TABLE[59] = 32'h0000034B;
        EXP_LOOKUP_TABLE[60] = 32'h00000290;
        EXP_LOOKUP_TABLE[61] = 32'h000001FF;
        EXP_LOOKUP_TABLE[62] = 32'h0000018E;
        EXP_LOOKUP_TABLE[63] = 32'h00000136;
    end

    reg [31:0] RECIPROCAL_LOOKUP_TABLE [0:31];

    initial begin
        RECIPROCAL_LOOKUP_TABLE[0] = 32'h7C1F07C1;
        RECIPROCAL_LOOKUP_TABLE[1] = 32'h78787878;
        RECIPROCAL_LOOKUP_TABLE[2] = 32'h75075075;
        RECIPROCAL_LOOKUP_TABLE[3] = 32'h71C71C71;
        RECIPROCAL_LOOKUP_TABLE[4] = 32'h6EB3E453;
        RECIPROCAL_LOOKUP_TABLE[5] = 32'h6BCA1AF2;
        RECIPROCAL_LOOKUP_TABLE[6] = 32'h69069069;
        RECIPROCAL_LOOKUP_TABLE[7] = 32'h66666666;
        RECIPROCAL_LOOKUP_TABLE[8] = 32'h63E7063E;
        RECIPROCAL_LOOKUP_TABLE[9] = 32'h61861861;
        RECIPROCAL_LOOKUP_TABLE[10] = 32'h5F417D05;
        RECIPROCAL_LOOKUP_TABLE[11] = 32'h5D1745D1;
        RECIPROCAL_LOOKUP_TABLE[12] = 32'h5B05B05B;
        RECIPROCAL_LOOKUP_TABLE[13] = 32'h590B2164;
        RECIPROCAL_LOOKUP_TABLE[14] = 32'h572620AE;
        RECIPROCAL_LOOKUP_TABLE[15] = 32'h55555555;
        RECIPROCAL_LOOKUP_TABLE[16] = 32'h5397829C;
        RECIPROCAL_LOOKUP_TABLE[17] = 32'h51EB851E;
        RECIPROCAL_LOOKUP_TABLE[18] = 32'h50505050;
        RECIPROCAL_LOOKUP_TABLE[19] = 32'h4EC4EC4E;
        RECIPROCAL_LOOKUP_TABLE[20] = 32'h4D4873EC;
        RECIPROCAL_LOOKUP_TABLE[21] = 32'h4BDA12F6;
        RECIPROCAL_LOOKUP_TABLE[22] = 32'h4A7904A7;
        RECIPROCAL_LOOKUP_TABLE[23] = 32'h49249249;
        RECIPROCAL_LOOKUP_TABLE[24] = 32'h47DC11F7;
        RECIPROCAL_LOOKUP_TABLE[25] = 32'h469EE584;
        RECIPROCAL_LOOKUP_TABLE[26] = 32'h456C797D;
        RECIPROCAL_LOOKUP_TABLE[27] = 32'h44444444;
        RECIPROCAL_LOOKUP_TABLE[28] = 32'h4325C53E;
        RECIPROCAL_LOOKUP_TABLE[29] = 32'h42108421;
        RECIPROCAL_LOOKUP_TABLE[30] = 32'h41041041;
        RECIPROCAL_LOOKUP_TABLE[31] = 32'h40000000;
    end

    wire [31:0] frac_bits = cmd_payload_inputs_0;
    wire [31:0] raw_input = cmd_payload_inputs_1;

    parameter [6:0] EXP_in   = 7'd8;
    parameter [6:0] RECIP_in = 7'd9;

    wire [31:0] exp_index;
    parameter [31:0] exp_table_size = 64;  // 2^6
    parameter [31:0] exp_table_offset_bit = 2;  // 因為輸入絕對值範圍為 0~16，所以offset bit設為 6 - 4 = 2

    assign exp_index = (~raw_input + 1) >> (frac_bits - exp_table_offset_bit);
    
    wire [31:0] recip_index;
    parameter [31:0] recip_table_size = 32; // 2^5
    parameter [31:0] recip_table_offset_bit = 5; // 因為輸入絕對值為 0~1，所以offset bit設為 5 - 0 = 5
    
    assign recip_index = raw_input >> (frac_bits - recip_table_offset_bit);


    //*****************************************************************************************************//
    parameter BUF_ADDR_BITS = 8;

    //*****************************************//
    //*************** wire, reg ***************//
    //*****************************************//
    // wire rst_n;
    wire in_valid;
    wire busy_TPU_out;
    reg [15:0] A_cfu_wr;
    reg [15:0] B_cfu_wr;
    reg [15:0] A_cfu_wr_buf;
    reg [15:0] B_cfu_wr_buf;
    wire A_wr_en, B_wr_en, C_wr_en;
    reg [BUF_ADDR_BITS-1 :0] A_input_index, B_input_index;
    wire [BUF_ADDR_BITS-3 :0] C_input_index;
    wire [BUF_ADDR_BITS-1 :0] A_index, B_index;
    wire [BUF_ADDR_BITS-3 :0] C_index;
    reg [31:0] A_input_data, B_input_data;
    wire [31:0] A_data_in, B_data_in;
    wire [31:0] A_data_out_0, A_data_out_1, A_data_out_2, A_data_out_3, A_data_out_4, A_data_out_5, A_data_out_6, A_data_out_7,
                A_data_out_8, A_data_out_9, A_data_out_10, A_data_out_11, A_data_out_12, A_data_out_13, A_data_out_14, A_data_out_15;
    wire [31:0] B_data_out_0, B_data_out_1, B_data_out_2, B_data_out_3, B_data_out_4, B_data_out_5, B_data_out_6, B_data_out_7,
                B_data_out_8, B_data_out_9, B_data_out_10, B_data_out_11, B_data_out_12, B_data_out_13, B_data_out_14, B_data_out_15;

    wire [127:0] C_input_data, C_data_in, C_data_out;
    reg signed [31:0] input_offset;
    // wire reset_C_buf;
    wire [9:0] input_k; 
    wire [7:0] input_M, input_N; 
    reg [6:0] fun_7;
    reg input_A_flag;
    reg [9:0] input_k_reg;
    integer i;

    //*****************************************//
    //*********** input A, B buffer ***********//
    //*****************************************//
    reg [9:0] cnt_input_index; // count k
    always @(posedge clk) begin
        if(reset)
            cnt_input_index <= 0;
        else if(cmd_valid && cmd_payload_function_id[9:3] == 1)
            cnt_input_index <= 0;
        else if(cmd_valid && cmd_payload_function_id[9:3] == 1)
            cnt_input_index <= 0;
        else if(cmd_valid && cmd_payload_function_id[9:3] == 4) begin
            if(input_A_flag) begin
                if(cnt_input_index == input_k_reg-1)
                    cnt_input_index <= 0;
                else
                    cnt_input_index <= cnt_input_index + 1;
            end
            else begin
                if(cnt_input_index+2 > input_k_reg-1)
                    cnt_input_index <= 0;
                else
                    cnt_input_index <= cnt_input_index + 2;
            end
        end
    end

    reg [1:0] count_which_input_round; // total 4 round
    always @(posedge clk) begin
        if(reset)
            count_which_input_round <= 0;
        else if(cmd_valid && cmd_payload_function_id[9:3] == 1)
            count_which_input_round <= 0;
        else if(cmd_valid && cmd_payload_function_id[9:3] == 4) begin
            if(input_A_flag) begin
                if(cnt_input_index == input_k_reg-1) // next round
                    count_which_input_round <= count_which_input_round + 1;
            end
            else begin
                if(cnt_input_index+2 > input_k_reg-1) // next round
                    count_which_input_round <= count_which_input_round + 1;
            end
            
        end
    end

    //*****************************************//
    //********* buffer input control **********//
    //*****************************************//
    // assign A_input_index = cnt_input_index;
    // assign B_input_index = cnt_input_index;
    always @(posedge clk) begin
        if(reset) begin
            A_input_index <= 0;
            B_input_index <= 0;
        end
        else begin
            A_input_index <= {count_which_input_round, cnt_input_index[9:4]};
            B_input_index <= {count_which_input_round, cnt_input_index[9:4]};
        end
    end
    // assign A_cfu_wr = cmd_valid && cmd_payload_function_id[9:3] == 4;
    // assign B_cfu_wr = cmd_valid && cmd_payload_function_id[9:3] == 4;
    // assign A_input_data = cmd_payload_inputs_0;
    // assign B_input_data = cmd_payload_inputs_1;
    always @(posedge clk) begin
        if(reset) begin
            A_input_data <= 0;
            B_input_data <= 0;
        end
        else begin
            A_input_data <= {cmd_payload_inputs_0[7:0], cmd_payload_inputs_0[15:8], cmd_payload_inputs_0[23:16], cmd_payload_inputs_0[31:24]};
            B_input_data <= {cmd_payload_inputs_1[7:0], cmd_payload_inputs_1[15:8], cmd_payload_inputs_1[23:16], cmd_payload_inputs_1[31:24]};
            // A_input_data <= cmd_payload_inputs_0;
            // B_input_data <= cmd_payload_inputs_1;
        end
    end

    // control BRAM write enable
    // A buffer
    always @(*) begin
        if(cmd_valid && cmd_payload_function_id[9:3] == 4 && input_A_flag)
            for(i=0; i<16; i=i+1)
                A_cfu_wr[i] = (cnt_input_index[3:0] == i) ? 1 : 0;
        else
            A_cfu_wr = 0;
    end
    always @(posedge clk) begin
        if(reset)
            A_cfu_wr_buf <= 0;
        else
            A_cfu_wr_buf <= A_cfu_wr;
    end
    // B buffer
    always @(*) begin
        if(cmd_valid && cmd_payload_function_id[9:3] == 4) begin
            if(input_A_flag) begin
                for(i=0; i<16; i=i+1)
                    B_cfu_wr[i] = (cnt_input_index[3:0] == i) ? 1 : 0;
            end
            else begin
                for(i=0; i<16; i=i+1)
                    B_cfu_wr[i] = (cnt_input_index[3:1] == i/2) ? 1 : 0;
            end
            
        end
        else
            B_cfu_wr = 0;
    end
    always @(posedge clk) begin
        if(reset)
            B_cfu_wr_buf <= 0;
        else
            B_cfu_wr_buf <= B_cfu_wr;
    end
    // check whether buffer A has to refresh
    always @(posedge clk) begin
        if(reset)
            input_A_flag <= 0;
        else if(cmd_valid && cmd_payload_function_id[9:3] == 1)
            input_A_flag <= cmd_payload_inputs_1[10];
    end
    // store k index
    always @(posedge clk) begin
        if(reset)
            input_k_reg <= 0;
        else if(cmd_valid && cmd_payload_function_id[9:3] == 1)
            input_k_reg <= cmd_payload_inputs_1[9:0];
    end

    //*****************************************//
    //************ Globle Buffer A ************//
    //*****************************************//
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_0(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[0]),
        .index      ((A_cfu_wr_buf[0] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_0)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_1(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[1]),
        .index      ((A_cfu_wr_buf[1] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_1)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_2(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[2]),
        .index      ((A_cfu_wr_buf[2] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_2)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_3(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[3]),
        .index      ((A_cfu_wr_buf[3] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_3)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_4(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[4]),
        .index      ((A_cfu_wr_buf[4] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_4)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_5(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[5]),
        .index      ((A_cfu_wr_buf[5] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_5)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_6(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[6]),
        .index      ((A_cfu_wr_buf[6] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_6)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_7(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[7]),
        .index      ((A_cfu_wr_buf[7] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_7)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_8(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[8]),
        .index      ((A_cfu_wr_buf[8] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_8)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_9(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[9]),
        .index      ((A_cfu_wr_buf[9] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_9)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_10(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[10]),
        .index      ((A_cfu_wr_buf[10] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_10)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_11(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[11]),
        .index      ((A_cfu_wr_buf[11] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_11)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_12(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[12]),
        .index      ((A_cfu_wr_buf[12] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_12)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_13(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[13]),
        .index      ((A_cfu_wr_buf[13] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_13)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_14(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[14]),
        .index      ((A_cfu_wr_buf[14] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_14)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_A_15(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (A_cfu_wr_buf[15]),
        .index      ((A_cfu_wr_buf[15] || fun_7 == 4) ? A_input_index : A_index),
        .data_in    (A_input_data),
        .data_out   (A_data_out_15)
    );

    //*****************************************//
    //************ Globle Buffer B ************//
    //*****************************************//
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_0(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[0]),
        .index      ((B_cfu_wr_buf[0] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (input_A_flag ? B_input_data : A_input_data),
        .data_out   (B_data_out_0)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_1(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[1]),
        .index      ((B_cfu_wr_buf[1] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (B_input_data),
        .data_out   (B_data_out_1)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_2(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[2]),
        .index      ((B_cfu_wr_buf[2] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (input_A_flag ? B_input_data : A_input_data),
        .data_out   (B_data_out_2)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_3(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[3]),
        .index      ((B_cfu_wr_buf[3] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (B_input_data),
        .data_out   (B_data_out_3)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_4(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[4]),
        .index      ((B_cfu_wr_buf[4] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (input_A_flag ? B_input_data : A_input_data),
        .data_out   (B_data_out_4)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_5(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[5]),
        .index      ((B_cfu_wr_buf[5] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (B_input_data),
        .data_out   (B_data_out_5)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_6(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[6]),
        .index      ((B_cfu_wr_buf[6] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (input_A_flag ? B_input_data : A_input_data),
        .data_out   (B_data_out_6)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_7(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[7]),
        .index      ((B_cfu_wr_buf[7] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (B_input_data),
        .data_out   (B_data_out_7)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_8(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[8]),
        .index      ((B_cfu_wr_buf[8] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (input_A_flag ? B_input_data : A_input_data),
        .data_out   (B_data_out_8)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_9(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[9]),
        .index      ((B_cfu_wr_buf[9] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (B_input_data),
        .data_out   (B_data_out_9)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_10(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[10]),
        .index      ((B_cfu_wr_buf[10] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (input_A_flag ? B_input_data : A_input_data),
        .data_out   (B_data_out_10)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_11(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[11]),
        .index      ((B_cfu_wr_buf[11] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (B_input_data),
        .data_out   (B_data_out_11)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_12(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[12]),
        .index      ((B_cfu_wr_buf[12] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (input_A_flag ? B_input_data : A_input_data),
        .data_out   (B_data_out_12)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_13(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[13]),
        .index      ((B_cfu_wr_buf[13] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (B_input_data),
        .data_out   (B_data_out_13)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_14(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[14]),
        .index      ((B_cfu_wr_buf[14] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (input_A_flag ? B_input_data : A_input_data),
        .data_out   (B_data_out_14)
    );
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS), // ADDR_BITS 12 -> generates 2^12 entries
        .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_B_15(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (B_cfu_wr_buf[15]),
        .index      ((B_cfu_wr_buf[15] || fun_7 == 4) ? B_input_index : B_index),
        .data_in    (B_input_data),
        .data_out   (B_data_out_15)
    );

    //*****************************************//
    //************ Globle Buffer C ************//
    //*****************************************//
    global_buffer_bram #(
        .ADDR_BITS(BUF_ADDR_BITS-2), // store 16*16 answer, 256/4 = 64 = 2^6
        .DATA_BITS(128)  // DATA_BITS 32 -> 32 bits for each entries
    )
    gbuff_C(
        .clk        (clk),
        .rst_n      (1'b1),
        .ram_en     (1'b1),
        .wr_en      (C_wr_en),
        .index      ((cmd_valid && (cmd_payload_function_id[9:3] == 6||cmd_payload_function_id[9:3] == 4)) || fun_7 == 6 || fun_7 == 4 ? C_input_index : C_index),
        .data_in    (C_data_in),
        .data_out   (C_data_out)
    );
    //*****************************************//
    //******** buffer C output control ********//
    //*****************************************//
    reg [BUF_ADDR_BITS-1 :0] cnt_output_index;
    reg [31:0] buf_C_out;
    always @(posedge clk) begin
        if(reset)
            cnt_output_index <= 0;
        else if(cmd_valid && (cmd_payload_function_id[9:3] == 1 || cmd_payload_function_id[9:3] == 5))
            cnt_output_index <= 0;
        else if(cmd_valid && (cmd_payload_function_id[9:3] == 6 || cmd_payload_function_id[9:3] == 4) && cnt_output_index!=255)
            cnt_output_index <= cnt_output_index + 1;
    end
    
    assign C_input_index = cnt_output_index[BUF_ADDR_BITS-1 :2];
    always @(*) begin
        case (cnt_output_index[1:0])
            2'd0: buf_C_out =  C_data_out[127-:32];
            2'd1: buf_C_out =  C_data_out[95-:32];
            2'd2: buf_C_out =  C_data_out[63-:32];
            2'd3: buf_C_out =  C_data_out[31-:32];
            default: buf_C_out = C_data_out[127-:32];
        endcase
    end
    

    //*****************************************//
    //****************** TPU ******************//
    //*****************************************//
    // input_offset
    always @(posedge clk) begin
        if(cmd_valid && cmd_payload_function_id[9:3] == 1)
            input_offset <= cmd_payload_inputs_0;
    end
    // control
    assign in_valid     = cmd_valid && (cmd_payload_function_id[9:3] == 5);
    // assign reset_C_buf  = cmd_valid && (cmd_payload_function_id[9:3] == 7);
    assign {input_k, input_M, input_N} = cmd_payload_inputs_0[31:6];

    TPU tpu_module(
    .clk(clk),
    .rst_n(cmd_valid && cmd_payload_function_id[9:3] == 1),

    .input_offset(input_offset),
    // .reset_C_buf(reset_C_buf),

    .in_valid(in_valid),
    .K(input_k),
    .M(input_M),
    .N(input_N),
    .busy(busy_TPU_out),

    .A_wr_en(A_wr_en),
    .A_index(A_index),
    .A_data_in(A_data_in),
    .A_data_out_0(A_data_out_0),
    .A_data_out_1(A_data_out_1),
    .A_data_out_2(A_data_out_2),
    .A_data_out_3(A_data_out_3),
    .A_data_out_4(A_data_out_4),
    .A_data_out_5(A_data_out_5),
    .A_data_out_6(A_data_out_6),
    .A_data_out_7(A_data_out_7),
    .A_data_out_8(A_data_out_8),
    .A_data_out_9(A_data_out_9),
    .A_data_out_10(A_data_out_10),
    .A_data_out_11(A_data_out_11),
    .A_data_out_12(A_data_out_12),
    .A_data_out_13(A_data_out_13),
    .A_data_out_14(A_data_out_14),
    .A_data_out_15(A_data_out_15),

    .B_wr_en(B_wr_en),
    .B_index(B_index),
    .B_data_in(B_data_in),
    .B_data_out_0(B_data_out_0),
    .B_data_out_1(B_data_out_1),
    .B_data_out_2(B_data_out_2),
    .B_data_out_3(B_data_out_3),
    .B_data_out_4(B_data_out_4),
    .B_data_out_5(B_data_out_5),
    .B_data_out_6(B_data_out_6),
    .B_data_out_7(B_data_out_7),
    .B_data_out_8(B_data_out_8),
    .B_data_out_9(B_data_out_9),
    .B_data_out_10(B_data_out_10),
    .B_data_out_11(B_data_out_11),
    .B_data_out_12(B_data_out_12),
    .B_data_out_13(B_data_out_13),
    .B_data_out_14(B_data_out_14),
    .B_data_out_15(B_data_out_15),

    .C_wr_en(C_wr_en),
    .C_index(C_index),
    .C_data_in(C_data_in),
    .C_data_out(C_data_out));

    always @(posedge clk) begin
        if(reset)
            fun_7 <= 0;
        else if(cmd_valid)
            fun_7 <= cmd_payload_function_id[9:3];
    end

    // Only not ready for a command when we have a response.
    assign cmd_ready = ~rsp_valid;

    always @(posedge clk) begin
        if (reset) 
            rsp_payload_outputs_0 <= 0;
        else if(cmd_valid) begin
            if(cmd_payload_function_id[9:3] == 1) // reset
                rsp_payload_outputs_0 <= 0;
            else if(cmd_payload_function_id[9:3] == 6 || cmd_payload_function_id[9:3] == 4) // output C
                rsp_payload_outputs_0 <= buf_C_out;
            else if(cmd_payload_function_id[9:3] == EXP_in)
                rsp_payload_outputs_0 <= EXP_LOOKUP_TABLE[(exp_index < exp_table_size) ? exp_index : exp_table_size - 1];
            else if(cmd_payload_function_id[9:3] == RECIP_in)
                rsp_payload_outputs_0 <= RECIPROCAL_LOOKUP_TABLE[(recip_index < recip_table_size) ? recip_index : recip_table_size - 1];
            else if(cmd_payload_function_id[9:3] == 10)
                rsp_payload_outputs_0 <= sum_prods;
            else if(cmd_payload_function_id[9:3] == 11)
                rsp_payload_outputs_0 <= add_sums;    
            else if(cmd_payload_function_id[9:3] == 12)
                rsp_payload_outputs_0 <= $signed(cmd_payload_inputs_0)+$signed(cmd_payload_inputs_1);
        end
        else if(rsp_valid)
            rsp_payload_outputs_0 <= 0;
    end

    // only trigger rsp_valid for 1 cycle when fun7 is ready
    reg busy_TPU_out_delay_1_cycle;
    always @(posedge clk) begin
        if(reset)
            busy_TPU_out_delay_1_cycle <= 0;
        else
            busy_TPU_out_delay_1_cycle <= busy_TPU_out;
    end

    always @(posedge clk) begin
        if(reset)
            rsp_valid <= 0;
        else if(cmd_valid) begin
            if(cmd_payload_function_id[9:3] == 1 || cmd_payload_function_id[9:3] == 4 || cmd_payload_function_id[9:3] == 6
            || cmd_payload_function_id[9:3] == EXP_in || cmd_payload_function_id[9:3] == RECIP_in
            || cmd_payload_function_id[9:3] == 10 || cmd_payload_function_id[9:3] == 11 || cmd_payload_function_id[9:3] == 12)
                rsp_valid <= 1;
        end 
        else if((!busy_TPU_out) && busy_TPU_out_delay_1_cycle && fun_7 == 5)
            rsp_valid <= 1;
        else if(rsp_valid)
            rsp_valid <= ~rsp_ready;
    end	   


endmodule




module TPU(
    clk,
    rst_n,

    input_offset,
    // reset_C_buf,

    in_valid,
    K,
    M,
    N,
    busy,

    A_wr_en,
    A_index,
    A_data_in,
    A_data_out_0,
    A_data_out_1,
    A_data_out_2,
    A_data_out_3,
    A_data_out_4,
    A_data_out_5,
    A_data_out_6,
    A_data_out_7,
    A_data_out_8,
    A_data_out_9,
    A_data_out_10,
    A_data_out_11,
    A_data_out_12,
    A_data_out_13,
    A_data_out_14,
    A_data_out_15,

    B_wr_en,
    B_index,
    B_data_in,
    B_data_out_0,
    B_data_out_1,
    B_data_out_2,
    B_data_out_3,
    B_data_out_4,
    B_data_out_5,
    B_data_out_6,
    B_data_out_7,
    B_data_out_8,
    B_data_out_9,
    B_data_out_10,
    B_data_out_11,
    B_data_out_12,
    B_data_out_13,
    B_data_out_14,
    B_data_out_15,

    C_wr_en,
    C_index,
    C_data_in,
    C_data_out
);

    parameter BUF_ADDR_BITS = 8;

    input clk;
    input rst_n;
    input signed [31:0] input_offset;
    // input reset_C_buf;
    input            in_valid;
    input [9:0]      K;
    input [7:0]      M;
    input [7:0]      N;
    output  reg      busy;

    output           A_wr_en;
    output [BUF_ADDR_BITS-1 :0]     A_index;
    output [31:0]    A_data_in;
    input  [31:0]    A_data_out_0;
    input  [31:0]    A_data_out_1;
    input  [31:0]    A_data_out_2;
    input  [31:0]    A_data_out_3;
    input  [31:0]    A_data_out_4;
    input  [31:0]    A_data_out_5;
    input  [31:0]    A_data_out_6;
    input  [31:0]    A_data_out_7;
    input  [31:0]    A_data_out_8;
    input  [31:0]    A_data_out_9;
    input  [31:0]    A_data_out_10;
    input  [31:0]    A_data_out_11;
    input  [31:0]    A_data_out_12;
    input  [31:0]    A_data_out_13;
    input  [31:0]    A_data_out_14;
    input  [31:0]    A_data_out_15;

    output           B_wr_en;
    output [BUF_ADDR_BITS-1 :0]     B_index;
    output [31:0]    B_data_in;
    input  [31:0]    B_data_out_0;
    input  [31:0]    B_data_out_1;
    input  [31:0]    B_data_out_2;
    input  [31:0]    B_data_out_3;
    input  [31:0]    B_data_out_4;
    input  [31:0]    B_data_out_5;
    input  [31:0]    B_data_out_6;
    input  [31:0]    B_data_out_7;
    input  [31:0]    B_data_out_8;
    input  [31:0]    B_data_out_9;
    input  [31:0]    B_data_out_10;
    input  [31:0]    B_data_out_11;
    input  [31:0]    B_data_out_12;
    input  [31:0]    B_data_out_13;
    input  [31:0]    B_data_out_14;
    input  [31:0]    B_data_out_15;

    output           C_wr_en;
    output [BUF_ADDR_BITS-3 :0]     C_index;
    output [127:0]   C_data_in;
    input  [127:0]   C_data_out;


    //*****************************************//
    //************** new design ***************//
    //*****************************************//
    reg [5:0] cnt_row;
    reg [1:0] cnt_A_round, cnt_B_round;
    reg signed [7:0] A_data [0:15][0:3];
    reg signed [7:0] B_data [0:15][0:3];
    reg signed [15:0] A_times_B [0:15][0:3][0:3];
    reg signed [31:0] Add_A_times_B [0:3][0:3];
    reg signed [31:0] C_data_buf [0:3][0:3];
    reg signed [31:0] C_data_store [0:3][0:3];
    reg [9:0] K_reg;
    wire [5:0] k_divide_16;
    integer i, j, k;
    reg [2:0] cnt_first_reset;
    reg [2:0] wait_4_cycle; // wait for C output if rows that calculated for one answer is fewer than 4
    reg [1:0] cnt_out_cycles;
    reg output_C_flag;
    reg [5:0] n_C_index;
    reg [31:0] A_data_out_0_buf, A_data_out_1_buf, A_data_out_2_buf, A_data_out_3_buf, A_data_out_4_buf, A_data_out_5_buf, A_data_out_6_buf, A_data_out_7_buf,
               A_data_out_8_buf, A_data_out_9_buf, A_data_out_10_buf, A_data_out_11_buf, A_data_out_12_buf, A_data_out_13_buf, A_data_out_14_buf, A_data_out_15_buf;
    reg [31:0] B_data_out_0_buf, B_data_out_1_buf, B_data_out_2_buf, B_data_out_3_buf, B_data_out_4_buf, B_data_out_5_buf, B_data_out_6_buf, B_data_out_7_buf,
               B_data_out_8_buf, B_data_out_9_buf, B_data_out_10_buf, B_data_out_11_buf, B_data_out_12_buf, B_data_out_13_buf, B_data_out_14_buf, B_data_out_15_buf;

    assign k_divide_16 = K_reg==27 ? 2 : K_reg[9:4]; // how many rows for storing k

    //*****************************************//
    //************ common control *************//
    //*****************************************//
    // get input
    always @(posedge clk) begin
        if(rst_n) begin
            K_reg <= 0;
        end
        else if(in_valid) begin
            K_reg <= K;
        end
    end

    always @(posedge clk) begin
        if(rst_n)
            cnt_row <= 0;
        else begin
            if(in_valid)
                cnt_row <= 0;
            else if(cnt_row == k_divide_16-1) begin
                if(k_divide_16 < 4) begin
                    if(wait_4_cycle == 4 - k_divide_16)
                        cnt_row <= 0;
                end
                else
                    cnt_row <= 0;
            end
            else
                cnt_row <= cnt_row + 1;
        end
    end

    always @(posedge clk) begin
        if(rst_n)
            cnt_first_reset <= 0;
        else if(in_valid)
            cnt_first_reset <= 1;
        else if(cnt_first_reset == 5)
            cnt_first_reset <= 0;
        else if(cnt_first_reset > 0) begin
            cnt_first_reset <= cnt_first_reset + 1;
        end
    end

    always @(posedge clk) begin
        if(rst_n)
            wait_4_cycle <= 0;
        else if(wait_4_cycle == 5) begin
            if(k_divide_16 < 4)
                wait_4_cycle <= 2;
            else
                wait_4_cycle <= 0;
        end
        else if(wait_4_cycle > 0) begin
            wait_4_cycle <= wait_4_cycle + 1;
        end
        else if(cnt_row == k_divide_16-1)
            wait_4_cycle <= wait_4_cycle + 1;
    end

    //*****************************************//
    //*********** A matrix control ************//
    //*****************************************//
    always @(posedge clk) begin
        if(rst_n)
            cnt_A_round <= 0;
        else if(in_valid)
            cnt_A_round <= 0;
        else if(k_divide_16 < 4) begin
            if(wait_4_cycle == 4 - k_divide_16)
                cnt_A_round <= cnt_A_round + 1;
        end
        else if(cnt_row == k_divide_16-1)
            cnt_A_round <= cnt_A_round + 1;
    end

    assign A_index = {cnt_A_round, cnt_row};

    always @(posedge clk) begin
        A_data_out_0_buf  <= A_data_out_0;
        A_data_out_1_buf  <= A_data_out_1;
        A_data_out_2_buf  <= A_data_out_2;
        A_data_out_3_buf  <= A_data_out_3;
        A_data_out_4_buf  <= A_data_out_4;
        A_data_out_5_buf  <= A_data_out_5;
        A_data_out_6_buf  <= A_data_out_6;
        A_data_out_7_buf  <= A_data_out_7;
        A_data_out_8_buf  <= A_data_out_8;
        A_data_out_9_buf  <= A_data_out_9;
        A_data_out_10_buf <= A_data_out_10;
        A_data_out_11_buf <= (K_reg == 27 && cnt_row == 1) ? 0 : A_data_out_11;
        A_data_out_12_buf <= (K_reg == 27 && cnt_row == 1) ? 0 : A_data_out_12;
        A_data_out_13_buf <= (K_reg == 27 && cnt_row == 1) ? 0 : A_data_out_13;
        A_data_out_14_buf <= (K_reg == 27 && cnt_row == 1) ? 0 : A_data_out_14;
        A_data_out_15_buf <= (K_reg == 27 && cnt_row == 1) ? 0 : A_data_out_15;
    end

    always @(posedge clk) begin
        if(rst_n) begin
            for(i=0; i<16; i=i+1) begin
                A_data[i][0] <= 0;
                A_data[i][1] <= 0;
                A_data[i][2] <= 0;
                A_data[i][3] <= 0;
            end
        end
        else begin
            {A_data[0][0], A_data[0][1], A_data[0][2], A_data[0][3]} <= A_data_out_0_buf;
            {A_data[1][0], A_data[1][1], A_data[1][2], A_data[1][3]} <= A_data_out_1_buf;
            {A_data[2][0], A_data[2][1], A_data[2][2], A_data[2][3]} <= A_data_out_2_buf;
            {A_data[3][0], A_data[3][1], A_data[3][2], A_data[3][3]} <= A_data_out_3_buf;
            {A_data[4][0], A_data[4][1], A_data[4][2], A_data[4][3]} <= A_data_out_4_buf;
            {A_data[5][0], A_data[5][1], A_data[5][2], A_data[5][3]} <= A_data_out_5_buf;
            {A_data[6][0], A_data[6][1], A_data[6][2], A_data[6][3]} <= A_data_out_6_buf;
            {A_data[7][0], A_data[7][1], A_data[7][2], A_data[7][3]} <= A_data_out_7_buf;
            {A_data[8][0], A_data[8][1], A_data[8][2], A_data[8][3]} <= A_data_out_8_buf;
            {A_data[9][0], A_data[9][1], A_data[9][2], A_data[9][3]} <= A_data_out_9_buf;
            {A_data[10][0], A_data[10][1], A_data[10][2], A_data[10][3]} <= A_data_out_10_buf;
            {A_data[11][0], A_data[11][1], A_data[11][2], A_data[11][3]} <= A_data_out_11_buf;
            {A_data[12][0], A_data[12][1], A_data[12][2], A_data[12][3]} <= A_data_out_12_buf;
            {A_data[13][0], A_data[13][1], A_data[13][2], A_data[13][3]} <= A_data_out_13_buf;
            {A_data[14][0], A_data[14][1], A_data[14][2], A_data[14][3]} <= A_data_out_14_buf;
            {A_data[15][0], A_data[15][1], A_data[15][2], A_data[15][3]} <= A_data_out_15_buf;
        end
    end

    //*****************************************//
    //*********** B matrix control ************//
    //*****************************************//
    always @(posedge clk) begin
        if(rst_n)
            cnt_B_round <= 0;
        else if(in_valid)
            cnt_B_round <= 0;
        else if(k_divide_16 < 4) begin
            if(wait_4_cycle == 4 - k_divide_16 && cnt_A_round == 3)
                cnt_B_round <= cnt_B_round + 1;
        end
        else if(cnt_row == k_divide_16-1 && cnt_A_round == 3)
            cnt_B_round <= cnt_B_round + 1;
    end

    assign B_index = {cnt_B_round, cnt_row};

    always @(posedge clk) begin
        B_data_out_0_buf  <= B_data_out_0;
        B_data_out_1_buf  <= B_data_out_1;
        B_data_out_2_buf  <= B_data_out_2;
        B_data_out_3_buf  <= B_data_out_3;
        B_data_out_4_buf  <= B_data_out_4;
        B_data_out_5_buf  <= B_data_out_5;
        B_data_out_6_buf  <= B_data_out_6;
        B_data_out_7_buf  <= B_data_out_7;
        B_data_out_8_buf  <= B_data_out_8;
        B_data_out_9_buf  <= B_data_out_9;
        B_data_out_10_buf <= B_data_out_10;
        B_data_out_11_buf <= (K_reg == 27 && cnt_row == 1) ? 0 : B_data_out_11;
        B_data_out_12_buf <= (K_reg == 27 && cnt_row == 1) ? 0 : B_data_out_12;
        B_data_out_13_buf <= (K_reg == 27 && cnt_row == 1) ? 0 : B_data_out_13;
        B_data_out_14_buf <= (K_reg == 27 && cnt_row == 1) ? 0 : B_data_out_14;
        B_data_out_15_buf <= (K_reg == 27 && cnt_row == 1) ? 0 : B_data_out_15;
    end

    always @(posedge clk) begin
        if(rst_n) begin
            for(i=0; i<16; i=i+1) begin
                B_data[i][0] <= 0;
                B_data[i][1] <= 0;
                B_data[i][2] <= 0;
                B_data[i][3] <= 0;
            end
        end
        else begin
            {B_data[0][0], B_data[0][1], B_data[0][2], B_data[0][3]} <= B_data_out_0_buf;
            {B_data[1][0], B_data[1][1], B_data[1][2], B_data[1][3]} <= B_data_out_1_buf;
            {B_data[2][0], B_data[2][1], B_data[2][2], B_data[2][3]} <= B_data_out_2_buf;
            {B_data[3][0], B_data[3][1], B_data[3][2], B_data[3][3]} <= B_data_out_3_buf;
            {B_data[4][0], B_data[4][1], B_data[4][2], B_data[4][3]} <= B_data_out_4_buf;
            {B_data[5][0], B_data[5][1], B_data[5][2], B_data[5][3]} <= B_data_out_5_buf;
            {B_data[6][0], B_data[6][1], B_data[6][2], B_data[6][3]} <= B_data_out_6_buf;
            {B_data[7][0], B_data[7][1], B_data[7][2], B_data[7][3]} <= B_data_out_7_buf;
            {B_data[8][0], B_data[8][1], B_data[8][2], B_data[8][3]} <= B_data_out_8_buf;
            {B_data[9][0], B_data[9][1], B_data[9][2], B_data[9][3]} <= B_data_out_9_buf;
            {B_data[10][0], B_data[10][1], B_data[10][2], B_data[10][3]} <= B_data_out_10_buf;
            {B_data[11][0], B_data[11][1], B_data[11][2], B_data[11][3]} <= B_data_out_11_buf;
            {B_data[12][0], B_data[12][1], B_data[12][2], B_data[12][3]} <= B_data_out_12_buf;
            {B_data[13][0], B_data[13][1], B_data[13][2], B_data[13][3]} <= B_data_out_13_buf;
            {B_data[14][0], B_data[14][1], B_data[14][2], B_data[14][3]} <= B_data_out_14_buf;
            {B_data[15][0], B_data[15][1], B_data[15][2], B_data[15][3]} <= B_data_out_15_buf;
        end
    end

    //*****************************************//
    //********** calculate A times B **********//
    //*****************************************//
    always @(posedge clk) begin
        if(rst_n) begin
            for(i=0; i<16; i=i+1) begin
                for(j=0; j<4; j=j+1) begin
                    for(k=0; k<4; k=k+1) begin
                        A_times_B[i][j][k] <= 0;
                    end
                end
            end
        end
        else begin
            for(i=0; i<16; i=i+1) begin
                for(j=0; j<4; j=j+1) begin
                    for(k=0; k<4; k=k+1) begin
                        A_times_B[i][j][k] <= A_data[i][j] * (B_data[i][k]+input_offset);
                        // A_times_B[i][j][k] <= A_data[i][j] * B_data[i][k];
                    end
                end
            end
        end
    end

    //*****************************************//
    //*********** C matrix control ************//
    //*****************************************//
    always @(posedge clk) begin
        for(i=0; i<4; i=i+1) begin
            for(j=0; j<4; j=j+1) begin
                Add_A_times_B[i][j] <= A_times_B[0][i][j] + A_times_B[1][i][j] + A_times_B[2][i][j] + A_times_B[3][i][j]
                                     + A_times_B[4][i][j] + A_times_B[5][i][j] + A_times_B[6][i][j] + A_times_B[7][i][j]
                                     + A_times_B[8][i][j] + A_times_B[9][i][j] + A_times_B[10][i][j] + A_times_B[11][i][j]
                                     + A_times_B[12][i][j] + A_times_B[13][i][j] + A_times_B[14][i][j] + A_times_B[15][i][j];
            end
        end
    end

    always @(posedge clk) begin
        if(rst_n) begin
            for(i=0; i<4; i=i+1) begin
                for(j=0; j<4; j=j+1) begin
                    C_data_buf[i][j] <= 0;
                end
            end
        end
        else if(cnt_first_reset == 5) begin
            for(i=0; i<4; i=i+1) begin
                for(j=0; j<4; j=j+1) begin
                    C_data_buf[i][j] <= Add_A_times_B[i][j];
                end
            end
        end
        else if(wait_4_cycle == 5-(k_divide_16 < 4 ? k_divide_16 : 0)) begin
            for(i=0; i<4; i=i+1) begin
                for(j=0; j<4; j=j+1) begin
                    C_data_buf[i][j] <= Add_A_times_B[i][j];
                end
            end
        end
        else begin
            for(i=0; i<4; i=i+1) begin
                for(j=0; j<4; j=j+1) begin
                    C_data_buf[i][j] <= C_data_buf[i][j] + Add_A_times_B[i][j];
                end
            end
        end
    end
    
    always @(posedge clk) begin
        if(rst_n) begin
            for(i=0; i<4; i=i+1) begin
                for(j=0; j<4; j=j+1) begin
                    C_data_store[i][j] <= 0;
                end
            end
        end
        else if(wait_4_cycle == 5) begin
            for(i=0; i<4; i=i+1) begin
                for(j=0; j<4; j=j+1) begin
                    C_data_store[i][j] <= C_data_buf[i][j];
                end
            end
        end
    end

    always @(posedge clk) begin
        if(rst_n)
            cnt_out_cycles <= 0;
        else if(output_C_flag)
            cnt_out_cycles <= cnt_out_cycles + 1;
    end

    assign C_data_in = {C_data_store[cnt_out_cycles][0],  C_data_store[cnt_out_cycles][1],  C_data_store[cnt_out_cycles][2],  C_data_store[cnt_out_cycles][3]};

    always @(posedge clk) begin
        if(rst_n)
            n_C_index <= 0;
        else begin
            if(in_valid)
                n_C_index <= 0;
            else if(output_C_flag)
                n_C_index <= n_C_index + 1;
        end
    end
    assign C_index = n_C_index;

    always @(posedge clk) begin
        if(rst_n)
            output_C_flag <= 0;
        else if(wait_4_cycle == 5 && busy && n_C_index!=63)
            output_C_flag <= 1;
        else if(cnt_out_cycles == 3)
            output_C_flag <= 0;
    end
    assign C_wr_en = output_C_flag;

    // out valid
    always @(posedge clk) begin
        if(rst_n)
            busy <= 0;
        else if(in_valid)
            busy <= 1;
        else if(n_C_index == 63)
            busy <= 0;
    end

endmodule



module global_buffer_bram #(parameter ADDR_BITS=8, parameter DATA_BITS=8)(
    input                      clk,
    input                      rst_n,
    input                      ram_en,
    input                      wr_en,
    input      [ADDR_BITS-1:0] index,
    input      [DATA_BITS-1:0] data_in,
    output reg [DATA_BITS-1:0] data_out
    );

    parameter DEPTH = 2**ADDR_BITS;

    reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];

    always @ (negedge clk) begin
        if (ram_en) begin
            if(wr_en) begin
                gbuff[index] <= data_in;
            end
            else begin
                data_out <= gbuff[index];
            end
        end
    end

endmodule

    