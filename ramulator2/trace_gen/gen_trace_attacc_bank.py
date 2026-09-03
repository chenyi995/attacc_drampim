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
# head->HBM remap (ruling chenyi9 2026-08-27): one head owns one HBM; the
# channels a run spans carry the head's OWN token stripes (L divides across
# channels) instead of replicating the same columns per channel as extra
# heads.  n_head_per_hbm then means "heads resident on this HBM" and only
# shrinks each head's channel share; head parallelism across HBMs is a
# multiplier outside the trace.
head_hbm_stripe = False
n_head_per_hbm = 1  # overwritten in main() from --nhead; used by the stripe rule
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
## MACAB cadence is governed by the simulator preset nCCDAB (6 PC / 4 NPC),
## or the per-run yaml override for MQ (energy-clamped: 8 tCK at n=8).
## The old "8tCK (tCCDLx2)" annotation was upstream AttAcc's, not enforced.
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

def Attention(L, key_addr, val_addr, itr, valid_channel=None, extents=None):
  """Emit one scan's command stream.

  ``extents`` (chenyi9 2026-09-03) is the list of PHYSICAL K/V extents this
  scan actually reads, as ``(key_addr, value_addr, rows)``.  A reuse scan
  touches several of them -- a reused block's cached chunk, and the handful
  of recomputed rows that patch it -- and they are NOT contiguous.  Emitting
  all of them into ONE trace is what lets Ramulator decide the activations:
  extents that share a DRAM row are merged by its row buffer, extents that do
  not each pay their own ACT.  This restores upstream AttAcc's accounting,
  where one scan was one simulation and Ramulator counted every ACT; the
  per-extent split introduced with the CacheBlend port (0aced82) had to sum
  independent simulations instead, which can neither merge nor charge
  correctly across extents.

  ``extents=None`` keeps the single-extent behaviour byte for byte: the
  stream is exactly what ``(L, key_addr, val_addr)`` produced before.
  """
  # ``n_channel`` is set from the command line in main().  Do not capture its
  # import-time value (16) as a Python default argument, otherwise a
  # TLB-resolved one-channel run still emits commands to sixteen channels.
  if valid_channel is None:
    valid_channel = n_channel
  if not extents:
    extents = [(key_addr, val_addr, L)]
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

  def score_mac(extent_list):
    ## (pCH) C, C, R, R (MAC)
    # MAC and move output vector to softmax buffer
    ## Vector (1 x k) x Matrix (k x n) multiplication
    ## GEMV unit = adder tree mode
    # Each extent walks its OWN base address; the step counter is global so a
    # softmax-buffer flush still lands every 16 steps of the concatenated
    # stream rather than at every extent boundary.
    total_steps = sum(math.ceil(e_len / n_pch / n_rank / n_bg)
                      for _, _, e_len in extent_list)
    step = 0
    for addr_offset, _e_val, e_len in extent_list:
     for n_idx in range(math.ceil(e_len / n_pch / n_rank / n_bg)):# 16 
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
      step += 1
      if step % 16 == 0 or step == total_steps:
        cmd_score_mvsb[itr].append([])
        for bg_idx in range(n_bg):   
          for rank in range(n_rank):
            for lch in range(math.ceil(valid_channel)):
              bank_addr = addr_offset + ch_delta(addr_offset, lch) + rank * HBM_GS['rank'] + \
                          bg_idx * HBM_GS['bg']
              hex_addr = hex(bank_addr)[2:]
              cmd_score_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))
     # end of this extent

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

  def context_mac(extent_list):
    # MAC and move output vector to softmax buffer
    ## Vector (1xk) x Matrix (k x n ) multiplication
    ## GEMV unit = mac mode
    # n_idx stays the OUTER loop so the assembly below can keep indexing
    # ``cmd_context_mac[itr][j]`` by n_idx; every extent contributes its own
    # addresses inside that bucket.
    addr_offset = extent_list[0][1]
    for n_idx in range(math.ceil(dhead / (n_bank * n_mac))):
      cmd_context_mac[itr].append([])
      for _e_key, e_val, e_len in extent_list:
        for k_idx in range(math.ceil(e_len / (n_pch * n_rank * n_bg))):
          idx = k_idx + n_idx * math.ceil(e_len / (n_pch * n_rank * n_bg))
          for lch in range(math.ceil(valid_channel)):
            addr = e_val + ch_delta(e_val, lch) + idx * HBM_GS['col']
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

  # Query write-in and softmax are per SCAN, not per extent: the Q vector goes
  # to the GEMV buffer once and the softmax runs once over the whole context.
  # The two MAC phases are the ones that walk memory, so they take the extent
  # list.  ``L`` is the scan's total row count.
  score_cpvec(key_addr, L)

  score_mac(extents)

  softmax(L, key_addr)

  context_cpvec(val_addr, L)

  context_mac(extents)


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
  if head_hbm_stripe:
    L = math.ceil(L / max(1, n_channel // max(1, n_head_per_hbm)))
  partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * n_bank))
  num_itr = 1 if head_hbm_stripe else math.ceil(n_head_per_hbm / n_channel)
  cmd_list_reset()
  for itr in range(num_itr):
    remainder = 0
    if not head_hbm_stripe and n_head_per_hbm / ((itr + 1) * n_channel) < 1:
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
                  mq_command=False, phase="full", kv_extents=None):
  """``kv_extents``: the scan's real physical extents, [(key, value, rows)].

  One trace per CHANNEL carrying every extent that channel holds, so
  Ramulator's row buffer decides the activations (chenyi9 ruling 2026-09-03).
  ``None`` keeps the legacy single-extent path byte for byte.
  """
  if kv_extents:
    L = sum(rows for _, _, rows in kv_extents)
    key_base = kv_extents[0][0]
    value_base = kv_extents[0][1]
  if shared_queries < 1:
    raise ValueError("shared_queries must be positive")
  if shared_queries > 1:
    if not shared_kv:
      raise ValueError("shared_queries requires --shared-kv")
  if mq_command and shared_queries < 1:
    raise ValueError("--mq requires a positive --shared-queries")
  if phase not in ("full", "score", "context"):
    raise ValueError("--phase must be full, score, or context")

  if head_hbm_stripe and not kv_extents:
    # head->HBM remap (chenyi9 2026-08-27): the run's channels hold this
    # head's OWN token stripes, so every per-channel command count below
    # (generation and assembly alike) is derived from the striped length.
    # Explicit extents are ALREADY that channel's real rows, so they must not
    # be striped a second time.
    stripe_width = max(1, n_channel // max(1, n_head_per_hbm))
    L = math.ceil(L / stripe_width)

  partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * n_bank))
  head_offset = partition_size
  v_offset = pow(2, 23) 
  

  cmd_list_reset()
  ##-- Generate Commands --##
  num_itr = 1 if head_hbm_stripe else math.ceil(n_head_per_hbm / (n_channel))
  for itr in range(num_itr):
    remainder = 0
    if not head_hbm_stripe and (n_head_per_hbm / ((itr+1) * n_channel) < 1):
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
    # With explicit extents the caller has already resolved every address, so
    # the per-head partition offset must not shift them.
    itr_extents = kv_extents if kv_extents else None
    if remainder == 0:
      Attention(L, key_addr, val_addr, itr, extents=itr_extents)
    else:
      Attention(L, key_addr, val_addr, itr, remainder, extents=itr_extents)


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

    # With several extents the concatenated MAC stream is longer than
    # ceil(L/16) steps (each extent rounds up on its own), so drive the
    # loop from the real step count.  Identical to the old expression
    # for a single extent.
    length = max(math.ceil(L/n_pch/n_rank/n_bg/16),
                 math.ceil(len(cmd_score_mac[i]) / 16))
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
    # With several extents the concatenated MAC stream is longer than
    # ceil(L/16) steps (each extent rounds up on its own), so drive the
    # loop from the real step count.  Identical to the old expression
    # for a single extent.
    length = max(math.ceil(L/n_pch/n_rank/n_bg/16),
                 math.ceil(len(cmd_score_mac[i+1]) / 16))
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

    # With several extents the concatenated MAC stream is longer than
    # ceil(L/16) steps (each extent rounds up on its own), so drive the
    # loop from the real step count.  Identical to the old expression
    # for a single extent.
    length = max(math.ceil(L/n_pch/n_rank/n_bg/16),
                 math.ceil(len(cmd_score_mac[i]) / 16))
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
    # Per-phase MQ runs (PLAN_mq_command.md; streaming-P revision 2026-08-24):
    # the score phase runs once with n_q resident queries; the context phase
    # ALSO runs once with the same n_q -- probability vectors are NOT
    # resident.  A P entry has (almost) no per-bank reuse (one scalar per V
    # column per output pass), so each query's P simply streams through the
    # double-buffered GEMV-buffer halves via MV_GB; its bound is the
    # movement-bus bandwidth (32 B per nBL tCK per pCH, the stack-level
    # 1024-bit @ 5.2 Gbps TSV path) plus the MVSB<->MVGB direction
    # turnaround (nRTW/nWTRL), not the buffer capacity.  Each phase is its
    # own Ramulator run with its own --shared-queries, so the stream is
    # sliced here at the first MVGB: WRGB + score MACs/MVSB + SFM belong to
    # the score phase, MVGB + context MACs/MVSB to the context phase.  The
    # two-head pipeline interleaves phases across heads, so the slice is
    # defined only for a single head-iteration stream.
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
  global dhead, max_L, data_size, n_mac, n_channel, pool_base, n_head_per_hbm


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
  parser.add_argument("--head-hbm-stripe", action="store_true",
                      help="head->HBM remap: channels carry the head's own "
                           "token stripes (L splits across the run's "
                           "channels); nhead = heads resident on this HBM")
  parser.add_argument("--kv-extents-file", type=str, default=None,
                      help="file with the scan's real physical extents, one "
                           "'<key_addr> <value_addr> <rows>' per line (decimal "
                           "or 0x...).  One trace then carries EVERY extent of "
                           "one channel, so Ramulator's row buffer decides the "
                           "activations -- extents sharing a DRAM row merge, "
                           "extents that do not each pay an ACT.  Overrides "
                           "--key-addr/--value-addr/-l.")
  parser.add_argument("--pool-base", type=int, default=None,
                      help="first channel of that KV class's pool; heads then wrap inside "
                           "[pool-base, pool-base + channels) instead of striping past it")

  args = parser.parse_args()

  dhead = args.dhead
  max_L = args.maxlen
  L = args.seqlen
  n_head_per_hbm = args.nhead
  global head_hbm_stripe
  head_hbm_stripe = bool(getattr(args, "head_hbm_stripe", False)) 

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
  kv_extents = None
  if args.kv_extents_file:
    kv_extents = []
    with open(args.kv_extents_file) as handle:
      for line in handle:
        line = line.split("#", 1)[0].strip()
        if not line:
          continue
        key_field, value_field, rows_field = line.split()
        kv_extents.append((int(key_field, 0), int(value_field, 0),
                           int(rows_field, 0)))
    if not kv_extents:
      raise ValueError("--kv-extents-file is empty")

  run_attention(dhead, n_head_per_hbm, L, args.output, args.key_addr,
                args.value_addr, args.shared_kv, args.shared_queries,
                mq_command=args.mq, phase=args.phase, kv_extents=kv_extents)



if __name__ == "__main__":
  main()
