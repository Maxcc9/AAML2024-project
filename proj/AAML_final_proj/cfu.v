// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



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

  parameter [1:0] EXP_in = 2'b00;
  parameter [1:0] RECIP_in = 2'b01;

  wire [31:0] exp_index;
  parameter [31:0] exp_table_size = 64;  // 2^6
  parameter [31:0] exp_table_offset_bit = 2;  // 因為輸入絕對值範圍為 0~16，所以offset bit設為 6 - 4 = 2

  assign exp_index = (~raw_input + 1) >> (frac_bits - exp_table_offset_bit);
  
  wire [31:0] recip_index;
  parameter [31:0] recip_table_size = 32; // 2^5
  parameter [31:0] recip_table_offset_bit = 5; // 因為輸入絕對值為 0~1，所以offset bit設為 5 - 0 = 5
  
  assign recip_index = raw_input >> (frac_bits - recip_table_offset_bit);

   // Only not ready for a command when we have a response.
  assign cmd_ready = ~rsp_valid;

  always @(posedge clk) begin
    if (reset) begin
      rsp_payload_outputs_0 <= 0;
      rsp_valid <= 0;
    end else if (rsp_valid) begin
      rsp_valid <= ~rsp_ready;
    end else if (cmd_valid) begin
      rsp_valid <= 1;
      case (cmd_payload_function_id[9:3])
        EXP_in: begin
          rsp_payload_outputs_0 <= EXP_LOOKUP_TABLE[(exp_index < exp_table_size) ? exp_index : exp_table_size - 1];
        end
        RECIP_in: begin
          rsp_payload_outputs_0 <= RECIPROCAL_LOOKUP_TABLE[(recip_index < recip_table_size) ? recip_index : recip_table_size - 1];
        end
        default: begin
        end
      endcase
    end
  end

endmodule
