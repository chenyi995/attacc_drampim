import argparse
import math
import copy
import numpy as np

model = "gpt-3-175B"

dhead = 128
max_L = 2048
data_size = 16 # FP 16

n_attacc = 8
max_n_hbm = 8
n_hbm = 5
n_channel = 16
n_pch = 2
n_rank = 2
n_bank = 4
n_bg = 4
n_row = pow(2, 14)
n_col = pow(2, 5)
prefetch_size = 32 # byte
n_mac = 16


# Granularity size
HBM_GS = {}
HBM_GS['col']     = prefetch_size
HBM_GS['row']     = n_col * HBM_GS['col']
HBM_GS['ba']      = n_row * HBM_GS['row'] 
HBM_GS['bg']      = n_bank * HBM_GS['ba'] 
HBM_GS['rank']     = n_bg * HBM_GS['bg'] 
HBM_GS['pch']     = n_rank * HBM_GS['rank'] 
HBM_GS['ch']      = n_pch * HBM_GS['pch']
HBM_GS['hbm']     = n_channel * HBM_GS['ch']
HBM_GS['attacc']  = max_n_hbm * HBM_GS['hbm']


## --------------------------------------  HBM memory space -----------------------------------------##
## ------|  legacy CH  |  pCH  |  rank  | BG | BA |  row index  |  column index  |  access granularity  |------ ##
## bits  |     4       |   1   |   1   | 2  | 2  |     14      |        5       |          5           |       ##

## ----------------------------  Commands -------------------------------##
## MACAB: 8tCK (tCCDLx 2)
##  WRGB: 4tCK (write to SRAM not DRAM)
##  MVSB: 4tCK
##  MVGB: 4tCK
##  SFM: 16tCK (for L = 256)

cmd_score_wrgb   = []
cmd_score_mac    = []
cmd_score_mvsb   = []
cmd_sfm          = []
cmd_context_mvgb  = []
cmd_context_mac  = []
cmd_context_mvsb = []

valid_channels = []

rope_enabled = False
rope_num_agent = n_attacc
rope_mac_cmd = "PIM_MAC_PB"


def set_rope_config(enabled, num_agent, diff_rate):
  global rope_enabled, rope_num_agent, rope_diff_rate
  rope_enabled = enabled
  rope_num_agent = max(1, num_agent)
  rope_diff_rate = min(1.0, max(0.0, diff_rate))


def rope_agents_per_row():
  # Master K consumes one column; every packed agent consumes cos/sin.
  return max(1, int((n_col - 1) / 2))


def rope_agent_group_count():
  return math.ceil(rope_num_agent / rope_agents_per_row())


def rope_agent_group_size(agent_group_idx):
  first_agent = agent_group_idx * rope_agents_per_row()
  return max(1, min(rope_agents_per_row(), rope_num_agent - first_agent))


def rope_cols_per_chunk(agent_group_idx):
  return 1 + 2 * rope_agent_group_size(agent_group_idx)


def rope_row_tiles_per_group(agent_group_idx):
  vector_chunks = math.ceil(dhead / n_mac)
  chunks_per_row = max(1, int(n_col / rope_cols_per_chunk(agent_group_idx)))
  return math.ceil(vector_chunks / chunks_per_row)


def rope_rows_per_token():
  return sum(rope_row_tiles_per_group(group_idx)
             for group_idx in range(rope_agent_group_count()))


def rope_group_row_offset(agent_group_idx):
  return sum(rope_row_tiles_per_group(group_idx)
             for group_idx in range(agent_group_idx))


def ropim_row_base(head_idx, token_idx, agent_group_idx, row_tile, n_head_per_hbm):
  # Shared-K RoPIM row layout. A row holds the master K section plus a bounded
  # group of agents' Sk; large agent counts replicate K across row groups.
  kv_group_idx = int(head_idx / rope_num_agent)
  lch = kv_group_idx % n_channel
  local_group = int(kv_group_idx / n_channel)
  rows_per_token = rope_rows_per_token()
  group_row_base = ((local_group * max_L + token_idx) * rows_per_token +
                    rope_group_row_offset(agent_group_idx))
  row_idx = group_row_base + row_tile
  bank_idx = kv_group_idx % n_bank
  bg_idx = int(kv_group_idx / n_bank) % n_bg
  rank_idx = int(kv_group_idx / (n_bank * n_bg)) % n_rank
  return (lch * HBM_GS['ch'] + rank_idx * HBM_GS['rank'] +
          bg_idx * HBM_GS['bg'] + bank_idx * HBM_GS['ba'] +
          row_idx * HBM_GS['row'])


def rope_diff_hash(block_idx, agent_idx):
  value = ((block_idx + 1) * 0x9E3779B1) ^ ((agent_idx + 1) * 0x85EBCA77)
  value ^= value >> 16
  value = (value * 0xC2B2AE3D) & 0xffffffff
  value ^= value >> 16
  return value


def rope_has_diff(block_idx, agent_idx):
  return rope_diff_hash(block_idx, agent_idx) < int(rope_diff_rate * (1 << 32))


def rope_diff_rank(block_idx, agent_idx):
  return sum(1 for idx in range(agent_idx) if rope_has_diff(block_idx, idx))


def rope_diff_count(block_idx):
  return sum(1 for idx in range(rope_num_agent) if rope_has_diff(block_idx, idx))


def ropim_diff_row_base(head_idx, block_idx, coordinate, overflow_idx):
  # Slice-major, dense-agent-minor diff slab; row zero is bitmap metadata.
  kv_group_idx = int(head_idx / rope_num_agent)
  lch = kv_group_idx % n_channel
  bank_idx = kv_group_idx % n_bank
  bg_idx = int(kv_group_idx / n_bank) % n_bg
  rank_idx = int(kv_group_idx / (n_bank * n_bg)) % n_rank
  rows_per_coordinate = max(1, math.ceil(rope_diff_count(block_idx) / n_col))
  logical_row = 1 + coordinate * rows_per_coordinate + overflow_idx
  row_idx = int(n_row / 2) + logical_row % int(n_row / 2)
  return (lch * HBM_GS["ch"] + rank_idx * HBM_GS["rank"] +
          bg_idx * HBM_GS["bg"] + bank_idx * HBM_GS["ba"] +
          row_idx * HBM_GS["row"])


def generate_rope_trace(n_head_per_hbm, L):
  total_cmd = []
  vector_chunks = math.ceil(dhead / n_mac)
  agents_per_row = rope_agents_per_row()
  token_idx = max(0, L - 1)
  block_idx = int(token_idx / rope_token_block)
  token_in_block = token_idx % rope_token_block
  loaded_bitmaps = set()
  # n_head_per_hbm already includes the head-agent product assigned to this HBM.
  # Generate one RoPE pre-pass for every served head-agent pair, not only for
  # the agent table count.
  for logical_idx in range(n_head_per_hbm):
    head_idx = logical_idx
    agent_idx = logical_idx % rope_num_agent
    agent_group_idx = int(agent_idx / agents_per_row)
    group_agent_idx = agent_idx % agents_per_row
    cols_per_chunk = rope_cols_per_chunk(agent_group_idx)
    chunks_per_row = max(1, int(n_col / cols_per_chunk))
    row_tiles = math.ceil(vector_chunks / chunks_per_row)
    for row_tile in range(row_tiles):
      row_base = ropim_row_base(head_idx, token_idx, agent_group_idx, row_tile, n_head_per_hbm)
      start_chunk = row_tile * chunks_per_row
      end_chunk = min(vector_chunks, start_chunk + chunks_per_row)
      for chunk_idx in range(start_chunk, end_chunk):
        local_col = chunk_idx - start_chunk
        chunk_col = local_col * cols_per_chunk
        k_addr = row_base + chunk_col * HBM_GS['col']
        sk_col = chunk_col + 1 + 2 * group_agent_idx
        cos_addr = row_base + sk_col * HBM_GS['col']
        sin_addr = row_base + (sk_col + 1) * HBM_GS['col']
        # Q_rotate is already produced on the GPU and Q never enters DRAM. V is
        # not rotated in this model, so PIM only applies RoPE to K.
        # Approximate the two RoPE branches with one MAC for cos and one for sin.
        for addr in [cos_addr, sin_addr]:
          hex_addr = hex(addr)[2:]
          total_cmd.append("{} 0x{:0>8}".format(rope_mac_cmd, hex_addr))
      hex_addr = hex(row_base)[2:]
      total_cmd.append("PIM_MV_SB 0x{:0>8}".format(hex_addr))
      total_cmd.append("PIM_BARRIER 0x{:0>8}".format(hex_addr))

    if rope_has_diff(block_idx, agent_idx):
      kv_group_idx = int(head_idx / rope_num_agent)
      bitmap_key = (kv_group_idx, block_idx)
      dense_rank = rope_diff_rank(block_idx, agent_idx)
      overflow_idx = int(dense_rank / n_col)
      if bitmap_key not in loaded_bitmaps:
        metadata_addr = (ropim_diff_row_base(head_idx, block_idx, 0, 0) -
                         HBM_GS["row"])
        total_cmd.append("PIM_MV_SB 0x{:0>8}".format(metadata_addr))
        loaded_bitmaps.add(bitmap_key)
      for chunk_idx in range(vector_chunks):
        coordinate = token_in_block * vector_chunks + chunk_idx
        diff_addr = (ropim_diff_row_base(
            head_idx, block_idx, coordinate, overflow_idx) +
            (dense_rank % n_col) * HBM_GS["col"])
        # Sk is staged from dense pass; correction has two RoPE branches.
        for _ in range(2):
          total_cmd.append("{} 0x{:0>8}".format(rope_mac_cmd, diff_addr))
        # Model the element-wise accumulation:
        # K_rot_agent = K_rot_master + K_rot_diff before attention score.
        total_cmd.append("{} 0x{:0>8}".format(rope_mac_cmd, diff_addr))
        row_base = diff_addr - (dense_rank % n_col) * HBM_GS["col"]
        total_cmd.append("PIM_MV_SB 0x{:0>8}".format(row_base))
        total_cmd.append("PIM_BARRIER 0x{:0>8}".format(row_base))
  return total_cmd

def cmd_list_reset():
  cmd_score_wrgb   = []
  cmd_score_mac    = []
  cmd_score_mvsb   = []
  cmd_sfm          = []
  cmd_context_mvgb = []
  cmd_context_mac  = []
  cmd_context_mvsb = []

  valid_channel = []

def Attention(L, key_addr, val_addr, itr, valid_channel = n_channel):
  cmd_score_wrgb.append([])
  cmd_score_mac.append([])
  cmd_score_mvsb.append([])
  cmd_sfm.append([])
  cmd_context_mvgb.append([])
  cmd_context_mac.append([])
  cmd_context_mvsb.append([])

  valid_channels.append(valid_channel);

  def score_cpvec(addr_offset, L):
    ## (pCH) C R (MAC)
    ## write input vector to gemv buffer
    # number of partition = (R parallel units)

    for col_idx in range(math.ceil(dhead / n_mac)):
      for lch in range(math.ceil(valid_channel)):
        # GEMV buffer address, col granularity = 1
        addr = addr_offset + lch * HBM_GS['ch'] + col_idx
        hex_addr = hex(addr)[2:]
        cmd_score_wrgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))

  def score_mac(addr_offset, L):
    ## (pCH) C R (MAC)
    # MAC and move output vector to softmax buffer
    ## Vector (1 x k) x Matrix (k x n) multiplication
    ## GEMV unit = adder tree mode
    for n_idx in range(math.ceil(L / n_pch)):
      cmd_score_mac[itr].append([])
      for k_idx in range(math.ceil(dhead / n_mac)):
        idx = k_idx + n_idx * math.ceil(dhead / n_mac) 

        bg_idx = idx % (n_bg * n_rank) 
        num_bg_indices = int(idx / (n_bg * n_rank))

        bank_idx = num_bg_indices % (n_bank)
        num_bank_indices = int(num_bg_indices / n_bank)

        col_idx = num_bank_indices % (int(HBM_GS['row'] / HBM_GS['col'])) 
        row_idx = int(num_bank_indices / (int(HBM_GS['row'] / HBM_GS['col'])))

        # All bank command (legacy channel)
        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + lch * HBM_GS['ch'] + bg_idx * HBM_GS['bg'] + \
                 bank_idx * HBM_GS['ba'] + row_idx * HBM_GS['row'] + col_idx * HBM_GS['col']
          hex_addr = hex(addr)[2:]
          cmd_score_mac[itr][-1].append("PIM_MAC_PB 0x{0:0>8}".format(hex_addr))
         ## parallelization

      ## MVSB command (Move to Softmax buffer) 
      ## A output element is generated for every n_idx
      if n_idx % 16 == 15 or n_idx == math.ceil(L / n_pch) - 1:
        cmd_score_mvsb[itr].append([])
        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + lch * HBM_GS['ch']
          hex_addr = hex(addr)[2:]
          cmd_score_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

  def context_cpvec(addr_offset, L):
    ## (pCH) R C (MAC)
    ## write input vector to gemv buffer
    # number of columns of partition = L / (R parallel units)
    for col_idx in range(math.ceil(L / (n_pch * n_mac))):
      for lch in range(math.ceil(valid_channel)):
        # GEMV buffer address, col granularity = 1
        addr = addr_offset + lch * HBM_GS['ch'] + col_idx
        hex_addr = hex(addr)[2:]
        cmd_context_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

  def context_mac(addr_offset, L):
    ## (pCH) R C (MAC)
    # MAC and move output vector to softmax buffer
    ## Vector (1xk) x Matrix (k x n ) multiplication
    ## GEMV unit = mac mode
    for n_idx in range(math.ceil(dhead / (n_mac))):
      cmd_context_mac[itr].append([])
      for k_idx in range(math.ceil(L / (n_pch))):
        idx = k_idx + n_idx * math.ceil(L / (n_pch))
        bg_idx = idx % (n_bg * n_rank) 
        num_bg_indices = int(idx / (n_bg * n_rank))

        bank_idx = num_bg_indices % (n_bank)
        num_bank_indices = int(num_bg_indices / n_bank)

        col_idx = num_bank_indices % (int(HBM_GS['row'] / HBM_GS['col'])) 
        row_idx = int(num_bank_indices / (int(HBM_GS['row'] / HBM_GS['col'])))

        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + lch * HBM_GS['ch'] + bg_idx * HBM_GS['bg'] + \
                 bank_idx * HBM_GS['ba'] + row_idx * HBM_GS['row'] + col_idx * HBM_GS['col']
          hex_addr = hex(addr)[2:]
          cmd_context_mac[itr][-1].append("PIM_MAC_PB 0x{0:0>8}".format(hex_addr))

      ## parallelization. Generate 16 elements per n_idx
      cmd_context_mvsb[itr].append([])
      for lch in range(math.ceil(valid_channel)):
        addr = addr_offset + lch * HBM_GS['ch']
        hex_addr = hex(addr)[2:]
        cmd_context_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

  def softmax(L):
    for lch in range(math.ceil(valid_channel)):
      addr = lch * HBM_GS['ch'] 
      hex_addr = hex(addr)[2:]
      cmd_sfm[itr].append("PIM_SFM 0x{0:0>8}".format(hex_addr))

  score_cpvec(key_addr, L)

  score_mac(key_addr, L)

  softmax(L)

  context_cpvec(val_addr, L)

  context_mac(val_addr, L)


def run_attention(dhead, n_head_per_hbm, L, trace_file_name):
  partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * n_bank))
  head_offset = partition_size
  v_offset = pow(2, 23) 
  

  cmd_list_reset()
  ##-- Generate Commands --##
  num_itr = math.ceil(n_head_per_hbm / (n_channel))
  for itr in range(num_itr):
    remainder = 0
    if (n_head_per_hbm / ((itr+1) * n_channel) < 1):
      remainder = n_head_per_hbm % n_channel
    key_addr = itr * partition_size 
    val_addr = key_addr + v_offset
    if remainder == 0:
      Attention(L, key_addr, val_addr, itr)
    else:
      Attention(L, key_addr, val_addr, itr, remainder)


  rope_preamble = generate_rope_trace(n_head_per_hbm, L) if rope_enabled else []

  ##-- Ovelapping Commands --##
  barrier = []
  for lch in range(n_channel):
    addr = lch * HBM_GS['ch']
    hex_addr = hex(addr)[2:]
    barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

  total_cmd = rope_preamble
  for i in range(0, num_itr -1, 2):

    # Head0: Score
      ## WRGB
    total_cmd += cmd_score_wrgb[i]
      ## dummy MAC
    if i == 0:
      for j in range(valid_channels[i]):
        total_cmd.append(cmd_score_mac[i][0][j])
      ## BARRIER
    total_cmd += barrier

    length = math.ceil(L/n_pch/16)
    for j in range(0, length+1):
      ## MAC (Head0)
      if not j == length:
        stride = 16;
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_mac[i]):
            break;
          total_cmd += cmd_score_mac[i][j*stride+k]
      ## MVSB (Head0)
      if not j == 0:
        total_cmd += cmd_score_mvsb[i][j-1]
      ## WRGB (Head1)
      if not j == length:
        stride = int(math.ceil(dhead/n_mac)*math.ceil(valid_channels[i+1])/length);
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_wrgb[i+1]):
            break;
          total_cmd.append(cmd_score_wrgb[i+1][j*stride + k])
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # Head0: SoftMax, Head1: Score
    length = math.ceil(L/n_pch/16)
    for j in range(0, length+1):
      ## MAC (Head1)
      if not j == length:
        stride = 16;
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_mac[i+1]):
            break;
          total_cmd += cmd_score_mac[i+1][j*stride+k]
      ## MVSB (Head1)
      if not j == 0:
        total_cmd += cmd_score_mvsb[i+1][j-1]
      ## SFM (Head0)
      if j == 0:
        total_cmd += cmd_sfm[i]
      ## MVGB (Head0)
      if not j == length:
        if j >= math.floor(length/2):
          stride = int(math.ceil(L/(n_pch*n_mac))*math.ceil(valid_channels[i])/math.ceil(length/2));
          for k in range(stride):
            if ((j-math.floor(length/2))*stride + k) >= len(cmd_context_mvgb[i]):
              break;
            total_cmd.append(cmd_context_mvgb[i][(j-math.floor(length/2))*stride + k])
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # Head0: Context, Head1: Softmax
    length = math.ceil(dhead/n_mac)
    for j in range(0, length+1):
      ## MAC (Head0)
      if not j == length:
        total_cmd += cmd_context_mac[i][j]
      ## MVSB (Head0)
      if not j == 0:
        total_cmd += cmd_context_mvsb[i][j-1]
      ## SFM (Head1)
      if j == 0:
        total_cmd += cmd_sfm[i+1]
      ## MVGB (Head1)
      if not j == length:
        if j >= math.floor(length/2):
          stride = int(math.ceil(L/(n_pch*n_mac))*math.ceil(valid_channels[i+1])/math.ceil(length/2));
          for k in range(stride):
            if ((j-math.floor(length/2))*stride + k) >= len(cmd_context_mvgb[i+1]):
              break;
            total_cmd.append(cmd_context_mvgb[i+1][(j-math.floor(length/2))*stride + k])
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # Head1: Context
    length = math.ceil(dhead/n_mac)
    for j in range(0, length+1):
      ## MAC (Head0)
      if not j == length:
        total_cmd += cmd_context_mac[i][j]
      ## MVSB (Head0)
      if not j == 0:
        total_cmd += cmd_context_mvsb[i][j-1]
      ## BARRIER
      if not j == length:
        total_cmd += barrier


  if num_itr % 2 != 0:
    i = num_itr - 1

    # Score
      ## WRGB
    total_cmd += cmd_score_wrgb[i]
      ## BARRIER
    total_cmd += barrier

    length = math.ceil(L/n_pch/16)
    for j in range(0, length+1):
      ## MAC
      if not j == length:
        stride = 16;
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_mac[i]):
            break;
          total_cmd += cmd_score_mac[i][j*stride+k]
      ## MVSB
      if not j == 0:
        total_cmd += cmd_score_mvsb[i][j-1]
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # SoftMax
    ## SFM (Head0)
    total_cmd += cmd_sfm[i]
    ## MVGB (Head0)
    total_cmd += cmd_context_mvgb[i]
    ## BARRIER
    total_cmd += barrier

    # Context
    length = math.ceil(dhead/n_mac)
    for j in range(0, length+1):
      ## MAC
      if not j == length:
        total_cmd += cmd_context_mac[i][j]
      ## MVSB
      if not j == 0:
        total_cmd += cmd_context_mvsb[i][j-1]
      ## BARRIER
      if not j == length:
        total_cmd += barrier


  trace_file = open(trace_file_name, 'w')
  for cmd in total_cmd:
    trace_file.write(cmd + "\n")

  trace_file.close()

def main():
  global dhead, max_L, data_size, n_mac


  parser = argparse.ArgumentParser(description="Output path and operation infos",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
  parser.add_argument("-dh", "--dhead", type=int, default=128, 
                      help="dhead, default= 128")
  parser.add_argument("-nh", "--nhead", type=int, default=64,
                      help="Number of heads, default=64")
  parser.add_argument("-l", "--seqlen", type=int, default=2048,
                      help="Sequence length L, default= 2048")
  parser.add_argument("-maxl", "--maxlen", type=int, default=4096, 
                      help="maximum L, default= 4096")
  parser.add_argument("-db", "--dbyte", type=int, default=2, 
                      help="data type (B), default= 2")
  parser.add_argument("--rope", action="store_true",
                      help="enable RoPIM-style K/Sk row-buffer-local RoPE pre-pass")
  parser.add_argument("--num-agent", type=int, default=n_attacc,
                      help="number of GPU-generated Sk vectors/PIM agents")
  parser.add_argument("--diff-rate", type=float, default=0.1,
                      help="per-agent Bernoulli KV-block diff probability")
  parser.add_argument("-o", "--output", type=str, default="attacc_buffer.trace", 
                      help="output path")

  args = parser.parse_args()

  dhead = args.dhead
  max_L = args.maxlen
  L = args.seqlen
  n_head_per_hbm = args.nhead 

  data_size = args.dbyte
  n_mac = int(HBM_GS['col'] / data_size)
  set_rope_config(args.rope, args.num_agent, args.diff_rate)

  print("------   Make a trace of buffer-level AttAcc   ------")

  args_dict = vars(args)
  print("All Arguments:")
  for key, value in args_dict.items():
      print(f"     {key}: {value}")
  print("---------------------------------------------------")
  run_attention(dhead, n_head_per_hbm, L, args.output)



if __name__ == "__main__":
  main()
