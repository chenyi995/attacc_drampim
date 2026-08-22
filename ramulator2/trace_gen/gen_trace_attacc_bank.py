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
n_channel_total = 16
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

# First channel of the KV pool a TLB-resolved run belongs to (``--pool-base``).
# When set, head ``lch`` of a block placed in channel ``blk`` is addressed at
# pool_base + ((blk - pool_base + lch) % n_channel): heads wrap inside the
# pool instead of running past its last channel into a foreign pool.  ``None``
# keeps the original AttAcc striping (block channel + lch).
pool_base = None

def ch_delta(addr_offset, lch):
  """Byte delta from ``addr_offset`` to the same offset in head lch's channel."""
  if pool_base is None:
    return lch * HBM_GS['ch']
  blk = (addr_offset // HBM_GS['ch']) % n_channel_total
  target = pool_base + ((blk - pool_base + lch) % n_channel)
  return (target - blk) * HBM_GS['ch']

def barrier_channel_addr(addr_offset, lch):
  if pool_base is None:
    return (addr_offset // HBM_GS['ch']) * HBM_GS['ch'] + lch * HBM_GS['ch']
  return (pool_base + lch) * HBM_GS['ch']

def cmd_list_reset():
  cmd_score_wrgb   = []
  cmd_score_mac    = []
  cmd_score_mvsb   = []
  cmd_sfm          = []
  cmd_context_mvgb = []
  cmd_context_mac  = []
  cmd_context_mvsb = []

  valid_channel = []

def Attention(L, key_addr, val_addr, itr, valid_channel=None):
  # ``n_channel`` is set from the command line in main().  Do not capture its
  # import-time value (16) as a Python default argument, otherwise a
  # TLB-resolved one-channel run still emits commands to sixteen channels.
  if valid_channel is None:
    valid_channel = n_channel
  cmd_score_wrgb.append([])
  cmd_score_mac.append([])
  cmd_score_mvsb.append([])
  cmd_sfm.append([])
  cmd_context_mvgb.append([])
  cmd_context_mac.append([])
  cmd_context_mvsb.append([])

  valid_channels.append(valid_channel);

  def score_cpvec(addr_offset, L):
    ## (pCH) C, C, R, R (MAC)
    ## write input vector to gemv buffer
    # number of partition = (R parallel units)

    # Data broadcasting for pch, rank, bg, and ba
    for ba_idx in range(n_bank): # number of partitions
      for col_idx in range(math.ceil(dhead / n_bank / n_mac)):
        for lch in range(math.ceil(valid_channel)):
          # GEMV buffer address, col granularity = 1
          addr = addr_offset + ch_delta(addr_offset, lch) + ba_idx * HBM_GS['ba'] + col_idx
          hex_addr = hex(addr)[2:]
          cmd_score_wrgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))

  def score_mac(addr_offset, L):
    ## (pCH) C, C, R, R (MAC)
    # MAC and move output vector to softmax buffer
    ## Vector (1 x k) x Matrix (k x n) multiplication
    ## GEMV unit = adder tree mode
    for n_idx in range(math.ceil(L / n_pch / n_rank / n_bg)):# 16 
      cmd_score_mac[itr].append([])
      for k_idx in range(math.ceil(dhead / n_bank / n_mac)): # 2
        idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac) 

        # All bank command (legacy channel)
        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + ch_delta(addr_offset, lch) + idx * HBM_GS['col']
          hex_addr = hex(addr)[2:]
          cmd_score_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))
         ## parallelization

      ## MVSB command (Move to Softmax buffer) 
      ## A output element is generated for every n_idx
      if n_idx % 16 == 15 or n_idx == math.ceil(L / n_pch / n_rank / n_bg) - 1:
        cmd_score_mvsb[itr].append([])
        for bg_idx in range(n_bg):   
          for rank in range(n_rank):
            for lch in range(math.ceil(valid_channel)):
              bank_addr = addr_offset + ch_delta(addr_offset, lch) + rank * HBM_GS['rank'] + \
                          bg_idx * HBM_GS['bg']
              hex_addr = hex(bank_addr)[2:]
              cmd_score_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

  ## (pCH) R, R, C, C (MAC)
  def context_cpvec(addr_offset, L):
    ## write input vector to gemv buffer
    ## number of partition = (BG and BA banks)

    # Data broadcasting for bg and ba
    for rank in range(n_rank):
      for bg_idx in range(n_bg):
        for col_idx in range(math.ceil(L / (n_pch * n_rank * n_bg * n_mac))):
          # number of columns of partition = L / (R parallel units)
            for lch in range(math.ceil(valid_channel)):
              # GEMV buffer address, col granularity = 1
              addr = addr_offset + ch_delta(addr_offset, lch) + rank * HBM_GS['rank'] + \
                     bg_idx * HBM_GS['bg'] + col_idx
              hex_addr = hex(addr)[2:]
              cmd_context_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

  def context_mac(addr_offset, L):
    # MAC and move output vector to softmax buffer
    ## Vector (1xk) x Matrix (k x n ) multiplication
    ## GEMV unit = mac mode
    for n_idx in range(math.ceil(dhead / (n_bank * n_mac))):
      cmd_context_mac[itr].append([])
      for k_idx in range(math.ceil(L / (n_pch * n_rank * n_bg))):
        idx = k_idx + n_idx * math.ceil(L / (n_pch * n_rank * n_bg))
        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + ch_delta(addr_offset, lch) + idx * HBM_GS['col']
          hex_addr = hex(addr)[2:]
          cmd_context_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

      ## parallelization. Generate 16 elements per n_idx
      cmd_context_mvsb[itr].append([])
      for ba_idx in range(n_bank):
        for rank in range(n_rank):
          for lch in range(math.ceil(valid_channel)):
            bank_addr = addr_offset + ch_delta(addr_offset, lch) + rank * HBM_GS['rank'] + \
                        ba_idx * HBM_GS['ba']
            hex_addr = hex(bank_addr)[2:]
            cmd_context_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

  def softmax(L, addr_offset):
    channel_base = (addr_offset // HBM_GS['ch']) * HBM_GS['ch']
    for lch in range(math.ceil(valid_channel)):
      addr = channel_base + ch_delta(addr_offset, lch)
      hex_addr = hex(addr)[2:]
      cmd_sfm[itr].append("PIM_SFM 0x{0:0>8}".format(hex_addr))

  score_cpvec(key_addr, L)

  score_mac(key_addr, L)

  softmax(L, key_addr)

  context_cpvec(val_addr, L)

  context_mac(val_addr, L)


# n_head and n_req = n_req per a HBM 
def _shared_query_attention_commands(n_head_per_hbm, L, key_base, value_base,
                                     shared_queries):
  """Build a shared-KV multi-query command stream.

  The normal trace generator models head pipelining.  Reusing it by inflating
  ``nhead`` would incorrectly turn queries into physical heads.  For a true
  shared-KV batch, keep the head count unchanged and, for each K/V row
  command, issue the per-query operations while that row is still selected.
  Q, score/softmax state, and PV results remain private to each query; only
  the DRAM row selection and K/V address are common.

  This is deliberately used only for a batch larger than one.  Batch one
  continues through the legacy command interleaving below, preserving the
  original AttAcc trace exactly.
  """
  partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * n_bank))
  num_itr = math.ceil(n_head_per_hbm / n_channel)
  cmd_list_reset()
  for itr in range(num_itr):
    remainder = 0
    if n_head_per_hbm / ((itr + 1) * n_channel) < 1:
      remainder = n_head_per_hbm % n_channel
    valid_channel = n_channel if remainder == 0 else remainder
    offset = 0  # The TLB base already names the one shared resident segment.
    key_addr = key_base + offset
    val_addr = value_base + offset
    Attention(L, key_addr, val_addr, itr, valid_channel)

  barrier = []
  for lch in range(n_channel):
    barrier.append("PIM_BARRIER 0x{0:0>8}".format(
        hex(barrier_channel_addr(key_base, lch))[2:]))

  total_cmd = []
  for itr in range(num_itr):
    # Score phase.  Each query writes its own Q vector.  The MACs are then
    # interleaved at a K-row granularity, so repeating Q does not create a
    # second logical K/V placement or a second head.
    for _ in range(shared_queries):
      total_cmd += cmd_score_wrgb[itr]
    total_cmd += barrier
    score_mvsb_index = 0
    for score_index, score_cmds in enumerate(cmd_score_mac[itr]):
      for _ in range(shared_queries):
        total_cmd += score_cmds
        if (score_index % 16 == 15 or
                score_index == len(cmd_score_mac[itr]) - 1):
          total_cmd += cmd_score_mvsb[itr][score_mvsb_index]
      if score_index % 16 == 15 or score_index == len(cmd_score_mac[itr]) - 1:
        score_mvsb_index += 1
    # Score MAC/MVSB commands are ordered by the trace itself.  The barrier
    # is only needed before consuming the completed score state, not after
    # every K-row command (which would hide the row-buffer benefit of a
    # shared-KV batch).
    total_cmd += barrier

    # Softmax is query-private.  It follows that query's score stream before
    # any V/PV operation consumes the result.
    for _ in range(shared_queries):
      total_cmd += cmd_sfm[itr]
    total_cmd += barrier

    # Context/PV phase mirrors the score ordering over the shared V rows.
    for _ in range(shared_queries):
      total_cmd += cmd_context_mvgb[itr]
    total_cmd += barrier
    for context_index, context_cmds in enumerate(cmd_context_mac[itr]):
      for _ in range(shared_queries):
        total_cmd += context_cmds
        total_cmd += cmd_context_mvsb[itr][context_index]
    total_cmd += barrier
  return total_cmd


def run_attention(dhead, n_head_per_hbm, L, trace_file_name, key_base=None,
                  value_base=None, shared_kv=False, shared_queries=1,
                  mq_command=False, phase="full"):
  if shared_queries < 1:
    raise ValueError("shared_queries must be positive")
  if shared_queries > 1:
    if not shared_kv:
      raise ValueError("shared_queries requires --shared-kv")
  if mq_command and shared_queries < 1:
    raise ValueError("--mq requires a positive --shared-queries")
  if phase not in ("full", "score", "context"):
    raise ValueError("--phase must be full, score, or context")

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
    # CacheBlend supplies TLB-resolved physical byte addresses.  The old
    # synthetic placement remains the default for every legacy invocation.
    # ``shared_kv`` represents several independent Q batches that all scan
    # the same resident K/V segment.  The default remains the legacy layout
    # where each head group owns a distinct K/V partition.
    offset = 0 if shared_kv else itr * partition_size
    key_addr = (key_base if key_base is not None else 0) + offset
    val_addr = ((value_base if value_base is not None else key_addr + v_offset) +
                (offset if value_base is not None else 0))
    if remainder == 0:
      Attention(L, key_addr, val_addr, itr)
    else:
      Attention(L, key_addr, val_addr, itr, remainder)


  ##-- Ovelapping Commands --##
  barrier = []
  for lch in range(n_channel):
    addr = barrier_channel_addr(key_base if key_base is not None else 0, lch)
    hex_addr = hex(addr)[2:]
    barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

  total_cmd = []
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

    length = math.ceil(L/n_pch/n_rank/n_bg/16)
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
        stride = int(n_bank*math.ceil(dhead /n_bank /n_mac)*math.ceil(valid_channels[i+1])/length);
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_wrgb[i+1]):
            break;
          total_cmd.append(cmd_score_wrgb[i+1][j*stride + k])
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # Head0: SoftMax, Head1: Score
    length = math.ceil(L/n_pch/n_rank/n_bg/16)
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
          stride = int(n_rank*n_bg*math.ceil(L/(n_pch*n_rank*n_bg*n_mac))*math.ceil(valid_channels[i])/math.ceil(length/2));
          for k in range(stride):
            if ((j-math.floor(length/2))*stride + k) >= len(cmd_context_mvgb[i]):
              break;
            total_cmd.append(cmd_context_mvgb[i][(j-math.floor(length/2))*stride + k])
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # Head0: Context, Head1: Softmax
    length = math.ceil(dhead/n_bank/n_mac)
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
          stride = int(n_rank*n_bg*math.ceil(L/(n_pch*n_rank*n_bg*n_mac))*math.ceil(valid_channels[i+1])/math.ceil(length/2));
          for k in range(stride):
            if ((j-math.floor(length/2))*stride + k) >= len(cmd_context_mvgb[i+1]):
              break;
            total_cmd.append(cmd_context_mvgb[i+1][(j-math.floor(length/2))*stride + k])
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # Head1: Context
    length = math.ceil(dhead/n_bank/n_mac)
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

    length = math.ceil(L/n_pch/n_rank/n_bg/16)
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
    length = math.ceil(dhead/n_bank/n_mac)
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


  if phase != "full":
    # Asymmetric MQ sweeps (PLAN_mq_command.md, extended 2026-08-21): the
    # score phase runs once with n_q resident queries, the context phase runs
    # ceil(n_q / n_c) passes with n_c resident probability vectors.  Each
    # phase is its own Ramulator run with its own --shared-queries, so the
    # stream is sliced here at the first MVGB: WRGB + score MACs/MVSB + SFM
    # belong to the score phase, MVGB + context MACs/MVSB to the context
    # phase.  The two-head pipeline interleaves phases across heads, so the
    # slice is defined only for a single head-iteration stream.
    if math.ceil(n_head_per_hbm / n_channel) > 1:
      raise ValueError("--phase score/context requires nhead <= channels")
    boundary = next((index for index, cmd in enumerate(total_cmd)
                     if cmd.startswith("PIM_MV_GB")), len(total_cmd))
    total_cmd = (total_cmd[:boundary] if phase == "score"
                 else total_cmd[boundary:])

  if shared_queries > 1:
    # Preserve the original two-head pipeline exactly, but expand every
    # query-private PIM operation at the point where its K/V row is live.
    # A barrier is shared by the whole batch: it denotes a true phase
    # dependency, not one dependency per Q.
    #
    # Two batch-command schemes:
    # * replicate (default): every non-barrier command is issued once per Q.
    #   B queries against one column = B MAC_ABs, each re-reading the column.
    # * mq (--mq): the MQ-MAC semantics -- ONE MAC_AB reads the column once
    #   and the bank PE multiplies it against every resident Q internally.
    #   Only the genuinely query-private data movements stay per Q: WR_GB
    #   (load each Q), MV_SB (each Q's partial scores), SFM (each Q's
    #   softmax), MV_GB (each Q's probabilities).  The n-fold MAC time and
    #   the power stretch are modeled by the host-side nCCDAB override.
    expanded_cmd = []
    for cmd in total_cmd:
      if cmd.startswith("PIM_BARRIER"):
        expanded_cmd.append(cmd)
      elif mq_command and cmd.startswith("PIM_MAC_AB"):
        expanded_cmd.append(cmd)
      else:
        expanded_cmd.extend([cmd] * shared_queries)
    total_cmd = expanded_cmd

  with open(trace_file_name, 'w') as trace_file:
    for cmd in total_cmd:
      trace_file.write(cmd + "\n")

def main():
  global dhead, max_L, data_size, n_mac, n_channel, pool_base


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
  parser.add_argument("-o", "--output", type=str, default="attacc_bank.trace", 
                      help="output path")
  parser.add_argument("--key-addr", type=lambda value: int(value, 0), default=None,
                      help="physical byte address of the first K vector")
  parser.add_argument("--value-addr", type=lambda value: int(value, 0), default=None,
                      help="physical byte address of the first V vector")
  parser.add_argument("--shared-kv", action="store_true",
                      help="reuse one K/V physical segment for every query batch")
  parser.add_argument("--shared-queries", type=int, default=1,
                      help="number of private Q streams sharing that resident K/V segment")
  parser.add_argument("--phase", choices=("full", "score", "context"),
                      default="full",
                      help="emit only one attention phase (asymmetric MQ "
                           "sweeps run score and context as separate jobs "
                           "with their own --shared-queries)")
  parser.add_argument("--mq", action="store_true",
                      help="MQ-MAC batch command: one MAC_AB per column serves every "
                           "resident Q (PE-internal n-fold multiply); only WR_GB/MV_SB/"
                           "SFM/MV_GB stay per query.  Default replicates every command "
                           "per query.")
  parser.add_argument("--channels", type=int, default=16,
                      help="number of contiguous physical channels assigned to this KV class")
  parser.add_argument("--pool-base", type=int, default=None,
                      help="first channel of that KV class's pool; heads then wrap inside "
                           "[pool-base, pool-base + channels) instead of striping past it")

  args = parser.parse_args()

  dhead = args.dhead
  max_L = args.maxlen
  L = args.seqlen
  n_head_per_hbm = args.nhead 

  data_size = args.dbyte
  n_mac = int(HBM_GS['col'] / data_size)
  if not 1 <= args.channels <= 16:
    raise ValueError("channels must be in [1, 16]")
  n_channel = args.channels
  if args.pool_base is not None:
    if not 0 <= args.pool_base <= 16 - args.channels:
      raise ValueError("pool-base must keep the pool inside the 16 channels")
    pool_base = args.pool_base

  print("------   Make a trace of bank-level AttAcc   ------")

  args_dict = vars(args)
  print("All Arguments:")
  for key, value in args_dict.items():
      print(f"     {key}: {value}")
  print("---------------------------------------------------")
  run_attention(dhead, n_head_per_hbm, L, args.output, args.key_addr,
                args.value_addr, args.shared_kv, args.shared_queries,
                mq_command=args.mq, phase=args.phase)



if __name__ == "__main__":
  main()
