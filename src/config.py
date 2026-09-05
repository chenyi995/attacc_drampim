from src.type import *

SCALING_FACTOR = {}
SCALING_FACTOR['MAX_COMPUTE_UTIL'] = 0.8
SCALING_FACTOR['MAX_OFF_MEM_BW_UTIL'] = 0.85

# ---------------------------------------------------------------------------
# One HBM3 stack, shared by the GPU's near memory and the AttAcc far memory.
#
# Organisation and timing are the Ramulator presets the PIM side is simulated
# with (``HBM3_8Gb_2R`` / ``HBM3_5.2Gbps`` in ramulator2/src/dram/impl/
# HBM3-PIM.cpp), so the GPU's HBM in the refined model is *the same device*
# as the PIM's: 16 channels x 2 pseudo-channels x 2 ranks x 4 BG x 4 banks,
# 1 KiB rows of 32 x 32 B columns, 5.2 Gbps -> 670.4 GB/s per stack.  The
# A100a GPU has five of them (80 GB, 3352 GB/s), exactly the AttAcc stack
# count, which is also why the legacy ``OFF_MEM_BW_PER_DEVICE`` was 3352.
# ---------------------------------------------------------------------------
HBM3_STACK = {
    'BYTES_PER_S': 670.4 * 1000 * 1000 * 1000,
    'CAPACITY_BYTES': 16 * 1024 * 1024 * 1024,
    'CHANNELS': 16,
    'PSEUDO_CHANNELS': 2,
    'RANKS': 2,
    'BANK_GROUPS': 4,
    'BANKS_PER_GROUP': 4,
    'ROW_BYTES': 1024,
    'COLUMN_BYTES': 32,
    'tCK_PS': 1300,
    # cycles (HBM3_5.2Gbps preset)
    'nBL': 2, 'nCL': 19, 'nRCD': 19, 'nRP': 19, 'nRAS': 45, 'nRC': 63,
    'nCCDS': 2, 'nCCDL': 4, 'nRRDS': 2, 'nRRDL': 4, 'nFAW': 39,
    'nRTW': 3, 'nWTRL': 11, 'nRFC': 260, 'nREFI': 5070,
}


def hbm3_stream_efficiency(spec=HBM3_STACK):
    """Fraction of peak a bank-interleaved streaming access pattern sustains.

    A GPU memory controller keeps >= tRC / (row data time) banks in flight per
    channel, so row activation is hidden behind the data bus and the only
    losses are refresh (tRFC / tREFI) and the read<->write bus turnaround,
    charged once per 1 KiB row (32 columns x nBL cycles of data against
    (nRTW + nWTRL) / 2 cycles of bubble).  With the HBM3 presets this is
    0.949 x 0.901 = 0.855 -- the legacy model's flat 0.85 derived from the
    device timing rather than assumed.
    """
    refresh = 1.0 - spec['nRFC'] / spec['nREFI']
    row_data_cycles = (spec['ROW_BYTES'] // spec['COLUMN_BYTES']) * spec['nBL']
    turnaround = (spec['nRTW'] + spec['nWTRL']) / 2.0
    bus = row_data_cycles / (row_data_cycles + turnaround)
    return refresh * bus


GPU_MODELS = ('legacy', 'refined', 'flash')

# ENERGY_TABLE: pJ per byte
# Cache info: https://core.ac.uk/download/pdf/232142915.pdf
ENERGY_TABLE = {
    'GPU': {},
    'CPU': {},
    'PIM': {
        PIMType.BA: {},
        PIMType.BG: {},
        PIMType.BUFFER: {}
    }
}
ENERGY_TABLE['GPU']['reg'] = 0.0675
#4-way cache, ref: https://arxiv.org/pdf/1509.02308v1.pdf
ENERGY_TABLE['GPU'][ 'l1'] = 0.16 * 8  
ENERGY_TABLE['GPU']['l2'] = 0.3 * 8
ENERGY_TABLE['GPU']['alu'] = 0.32
ENERGY_TABLE['GPU']['mem'] = (0.11 + 0.44 + 1.01 + 1.23 + 0.5 + 0.3) * 8
# ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10067395
ENERGY_TABLE['GPU'][ 'comm'] = 1.3 * 8  

## TODO: Add energy of CPU (pJ per byte)
ENERGY_TABLE['CPU']['reg'] = 0
ENERGY_TABLE['CPU']['l1'] = 0
ENERGY_TABLE['CPU']['l2'] = 0
ENERGY_TABLE['CPU']['alu'] = 0
ENERGY_TABLE['CPU']['mem'] = 0
ENERGY_TABLE['CPU']['comm'] = 0

## 2017 MICRO FGDRAM
## https://www.cs.utexas.edu/users/skeckler/pubs/MICRO_2017_Fine_Grained_DRAM.pdf
## Cell (ACT/PRE) energy: 0.11pJ/b,
## Cell (RD/WRT) energy: 0.44pJ/b,

## RD/WR Energy (column decoder to BG MUX): 1.01 pJ/b
## RD/WR Energy (BG Mux to GIO Mux): 1.23 pJ/b
## TSV energy : 0.5 pJ/b
## Silicon interposer IO energy : 0.3 pJ/b

## energy_table = [energy between DRAM cell and PE, energy between PE and buffer die

ENERGY_TABLE['PIM'][PIMType.BA]['mem'] = (0.11 +
                                          0.44) * 8  #, (1.01 + 1.23 + 0.5) * 8]
ENERGY_TABLE['PIM'][PIMType.BG]['mem'] = (0.11 + 0.44 +
                                          1.01) * 8  #, (1.23 + 0.5) * 8]
ENERGY_TABLE['PIM'][PIMType.BUFFER]['mem'] = (0.11 + 0.44 + 1.01 + 1.23 +
                                              0.5) * 8  #, 0]

ENERGY_TABLE['PIM'][PIMType.BA]['sram'] = 0.0034
ENERGY_TABLE['PIM'][PIMType.BG]['sram'] = 0.0034
ENERGY_TABLE['PIM'][PIMType.BUFFER]['sram'] = 0.0034

ENERGY_TABLE['PIM'][PIMType.BA]['alu'] = 0.32
ENERGY_TABLE['PIM'][PIMType.BG]['alu'] = 0.32
ENERGY_TABLE['PIM'][PIMType.BUFFER]['alu'] = 0.32

ENERGY_TABLE['PIM'][PIMType.BA]['io'] = [0.3, 0.5, 1.23, 1.01]
ENERGY_TABLE['PIM'][PIMType.BG]['io'] = [0.3, 0.5, 1.23, 1.01]
ENERGY_TABLE['PIM'][PIMType.BUFFER]['io'] = [0.3, 0.5, 1.23, 1.01]

# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10067395
ENERGY_TABLE['PIM'][PIMType.BA]['comm'] = 10.4
ENERGY_TABLE['PIM'][PIMType.BG]['comm'] = 10.4
ENERGY_TABLE['PIM'][PIMType.BUFFER]['comm'] = 10.4


def make_xpu_config(gpu_type: GPUType,
                    num_gpu=None,
                    flops=None,
                    mem_cap=None,
                    mem_bw=None,
                    power_constraint=True,
                    gpu_model='legacy',
                    pim_link_bw=None,
                    attn_splitk=False):
    assert gpu_model in GPU_MODELS, "gpu_model must be one of {}".format(GPU_MODELS)
    config = {'GPU': {}, 'CPU': {}}
    config['GPU']["GPUTYPE"] = gpu_type
    config['GPU']["NUM_DEVICE"] = 8 if num_gpu is None else num_gpu
    # ``legacy``: the original AttAcc xPU model (flat 0.8 compute
    # utilisation on every GEMM).  ``refined`` (user decision 2026-08-20):
    # the projection GEMMs and the attention score/context matmuls are
    # priced with the cuBLAS efficiency measured for their size instead of
    # the flat 0.8 -- nothing else about them changes (same tiling, same
    # per-layer SM occupancy, same memory model, S = QK^T still materialised),
    # so attention can only get slower, never faster; and every GPU<->AttAcc
    # transfer pays the NVLink latency plus the far HBM3's streaming time.
    # Softmax, activation, norm and all-reduce keep the legacy GPU model.
    # ``flash`` (user decision 2026-08-21) = ``refined`` plus the attention
    # priced as a fused FlashAttention-2 kernel: one CTA per (head, request,
    # 128-row Q block), efficiency vs key length from the FA-2 A100 plots,
    # S = QK^T never leaves the SM (softmax fused, no HBM traffic), decode
    # (m = 1) split over keys (flash-decoding).
    # The GPU's near HBM is the same HBM3 stack as the AttAcc's (5 x 670.4
    # GB/s); its derived streaming efficiency (0.855) matches the legacy 0.85
    # constant, which is therefore kept.
    config['GPU']["GPU_MODEL"] = gpu_model
    # fused attention: rows of Q held per CTA (FlashAttention-2, d = 128) and
    # the tensor-core MMA row granularity a short Q block is padded to.
    config['GPU']["ATTN_Q_BLOCK"] = 128
    config['GPU']["ATTN_MMA_ROWS"] = 16
    # keys per CTA when a decode-shaped attention (m == 1) is split over the
    # key dimension (flash-decoding) to fill the SMs.
    config['GPU']["ATTN_DECODE_SPLIT"] = 256
    # flash only: also let a short-Q prefill attention (m > 1) split its key
    # range across CTAs (flash-attn's ``num_splits`` heuristic) when the
    # (head, request, Q-block) CTAs alone cannot fill the SMs; the split is
    # chosen to maximise FA efficiency(keys/s) x SM occupancy.  Off by
    # default so the 2026-08-21 flash matrix stays reproducible.
    config['GPU']["ATTN_SPLITK"] = bool(attn_splitk)
    config['GPU']["HBM_SPEC"] = HBM3_STACK
    config['GPU']["NUM_HBM_STACKS"] = 5
    config['GPU']["HBM_STREAM_EFF"] = hbm3_stream_efficiency()
    # latency of one GPU<->AttAcc NVLink transfer: the intercept of the A100
    # all-reduce fit used by ``get_nvlink_time`` (6.06 us).
    config['GPU']["NVLINK_LATENCY_S"] = 6060e-9
    # Bandwidth of the GPU <-> AttAcc link used by X2G transfers (K/V, Q,
    # context); the GPU <-> GPU all-reduce keeps INTERFACE_BW.  None = same
    # NVLink generation as the GPU fabric (the original AttAcc assumption).
    config['GPU']["PIM_LINK_BW"] = pim_link_bw

    if gpu_type == GPUType.A100a:
        # Ref: DGX-A100 whitepaper
        config['GPU']["NUM_CORE"] = 108
        config['GPU']["FLOPS_PER_DEVICE"] = 312 * 1000 * 1000 * 1000 * 1000 \
                                            if flops is None else flops
        config['GPU']["MEM_CAPACITY_PER_DEVICE"] = 80 * 1024 * 1024 * 1024 \
                                                    if mem_cap is None else mem_cap

        # 5 x HBM3 stacks at 670.4 GB/s == the AttAcc memory device
        config['GPU']["OFF_MEM_BW_PER_DEVICE"] = 5 * HBM3_STACK['BYTES_PER_S'] \
                                                  if mem_bw is None else mem_bw
        config['GPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        #config['GPU']["L2_MEM_BW_PER_DEVICE"] = 3.8 * 1000 * 1000 * 1000 * 1000
        config['GPU']["L1_CAP_PER_CORE"] = 192 * 1024
        config['GPU']["L2_CAP_PER_DEVICE"] = 40 * 1024 * 1024
        config['GPU']["INTERFACE_BW"] = 600 * 1000 * 1000 * 1000
        config['GPU']["ENERGY_TABLE"] = ENERGY_TABLE['GPU']

        config['CPU']["NUM_DEVICE"] = 2
        config['CPU']["NUM_CORE"] = 64
        config['CPU']["FLOPS_PER_DEVICE"] = 4 * 1000 * 1000 * 1000 * 1000
        config['CPU']["MEM_CAPACITY_PER_DEVICE"] = 1024 * 1024 * 1024 * 1024
        config['CPU']["OFF_MEM_BW_PER_DEVICE"] = 200 * 1000 * 1000 * 1000
        config['CPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        # TODO: Modify it
        config['CPU']["L1_CAP_PER_CORE"] = 96 * 1024
        config['CPU']["L2_CAP_PER_DEVICE"] = 256 * 1024 * 1024
        config['CPU']["INTERFACE_BW"] = 4 * 64 * 1000 * 1000 * 1000
        config['CPU']["ENERGY_TABLE"] = ENERGY_TABLE['CPU']

    elif gpu_type == GPUType.H100:
        # Ref: DGX-H100 whitepaper
        config['GPU']["NUM_CORE"] = 132
        config['GPU']["FLOPS_PER_DEVICE"] = 989.4 * 1000 * 1000 * 1000 * 1000 \
                                            if flops is None else flops
        config['GPU']["MEM_CAPACITY_PER_DEVICE"] = 80 * 1024 * 1024 * 1024 \
                                                   if mem_cap is None else mem_cap
        config['GPU']["OFF_MEM_BW_PER_DEVICE"] = 3352 * 1000 * 1000 * 1000 \
                                                 if mem_bw is None else mem_bw
        config['GPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        # 5.5TB/s, https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
        #config['GPU']["L2_MEM_BW_PER_DEVICE"] = 5.5 * 1000 * 1000 * 1000 * 1000
        config['GPU']["L1_CAP_PER_CORE"] = 256 * 1024
        config['GPU']["L2_CAP_PER_DEVICE"] = 50 * 1024 * 1024
        # NVLINK: 900GB/s (Read 450GB/s Write 450GB/s)
        config['GPU']["INTERFACE_BW"] = 900 * 1000 * 1000 * 1000
        config['GPU']["ENERGY_TABLE"] = ENERGY_TABLE['GPU']

        # H100 DGX CPU configuration sapphire-rapids
        # https://www.servethehome.com/4th-gen-intel-xeon-scalable-sapphire-rapids-leaps-forward/7/
        config['CPU']["NUM_DEVICE"] = 2
        config['CPU']["NUM_CORE"] = 56
        # 4TFLOPS per CPU (half precision)
        config['CPU']["FLOPS_PER_DEVICE"] = 4 * 1000 * 1000 * 1000 * 1000
        # (2TB, dual processors)
        config['CPU']["MEM_CAPACITY_PER_DEVICE"] = 1024 * 1024 * 1024 * 1024
        # channels x dpc x 4400 MT/s  https://www.intel.com/content/www/us/en/products/sku/231746/intel-xeon-platinum-8480-processor-105m-cache-2-00-ghz/specifications.html
        config['CPU']["OFF_MEM_BW_PER_DEVICE"] = 8 * 2 * 4400 * (
            64 / 8) * 1000 * 1000
        config['CPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
        # 5.5TB/s, https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
        config['CPU']["L2_MEM_BW_PER_DEVICE"] = 5.5 * 1000 * 1000 * 1000 * 1000
        # TODO: Modify it
        config['CPU']["L1_CAP_PER_CORE"] = 48 * 1024
        config['CPU']["L2_CAP_PER_DEVICE"] = 2 * 1024 * 1024
        config['CPU']["INTERFACE_BW"] = 4 * 128 * 1000 * 1000 * 1000
        config['CPU']["ENERGY_TABLE"] = ENERGY_TABLE['CPU']

    return config


# Rank x BG x BA / 2 (tCCD)
BW_SCALE = {
    False: {
        PIMType.BA: 2 * 4 * 4 / 2,
        PIMType.BG: 2 * 4,
        PIMType.BUFFER: 1
    },
    True: {
        PIMType.BA: 9,
        PIMType.BG: 3,
        PIMType.BUFFER: 1
    }
}


def make_pim_config(pim_type: PIMType,
                    interface_type: InterfaceType,
                    opb=1,
                    num_attacc=8,
                    num_hbm=5,
                    bw_scale=None,
                    power_constraint=True):
    config = {}
    config["PIM_TYPE"] = pim_type
    config["POWER_CONSTRAINT"] = power_constraint
    config["ENERGY_TABLE"] = ENERGY_TABLE['PIM'][pim_type]

    internal_bandwidth_scale =  BW_SCALE[power_constraint][pim_type] \
                                if bw_scale is None else bw_scale
    config["NUM_ATTACC"] = num_attacc
    config["NUM_HBM"] = num_hbm
    config["MEM_CAPACITY_PER_HBM"] = 16 * 1024 * 1024 * 1024
    config[
        "MEM_BW_PER_HBM"] = 670.4 * 1000 * 1000 * 1000 * internal_bandwidth_scale
    config["FLOPS_PER_HBM"] = config["MEM_BW_PER_HBM"] * opb
    config["SOFTMAX_MEM_BW"] = 670.4 * 1000 * 1000 * 1000 * num_hbm
    config["SOFTMAX_FLOPS"] = config["SOFTMAX_MEM_BW"]

    if interface_type == InterfaceType.NVLINK3:
        config["INTERFACE_BW"] = 600 * 1000 * 1000 * 1000
    elif interface_type == InterfaceType.NVLINK4:
        config["INTERFACE_BW"] = 900 * 1000 * 1000 * 1000
    elif interface_type == InterfaceType.PCIE4:
        config["INTERFACE_BW"] = 64 * 1000 * 1000 * 1000
    elif interface_type == InterfaceType.PCIE5:
        config["INTERFACE_BW"] = 128 * 1000 * 1000 * 1000
    else:
        assert 0, "Invalid interface type"

    return config


def make_model_config(name, dtype):
    # ---- model_table entry = [n_layers, d_model, n_heads, d_head, ff_scale,
    # gqa_group].  SOURCES for the experiment1 models (every geometry number is
    # from a published architecture table; chenyi9 2026-08-29):
    #   LLaMA-1  (Touvron et al. 2023, arXiv:2302.13971, Table 2):
    #       7B  = 32 layers / dim 4096 / 32 heads
    #       33B = 60 layers / dim 6656 / 52 heads
    #       65B = 80 layers / dim 8192 / 64 heads
    #   Llama-3 (Grattafiori et al. 2024, arXiv:2407.21783, Sec. 3.2 / Table 3):
    #       8B  = 32 layers / dim 4096 / 32 query heads / 8 KV heads (GQA group 4)
    #   GPT-3   (Brown et al. 2020, arXiv:2005.14165, Table 2.1):
    #       13B  = 40 layers / n_heads 40 / d_head 128 (Table lists d_model 5140,
    #              which is NOT n_heads x d_head = 5120; we use the self-consistent
    #              5120 so d_head stays 128)
    #       175B = 96 layers / d_model 12288 / 96 heads / d_head 128
    # d_head is 128 for all of these per the same tables.  ``ff_scale``/``gqa``
    # follow each family (LLaMA SwiGLU 8/3, GPT 4x); the MT-*/OPT rows below are
    # legacy AttAcc entries, not used by experiment1.
    model_table = {}
    # Fast physical-DAG regression model.  It preserves the 128-wide head
    # geometry used by the HBM-PIM trace generator while keeping only four
    # decoder layers and a 1024-wide hidden state.
    model_table['CACHEBLEND-TINY'] = [4, 1024, 8, 128, 4, 1]
    model_table['GPT-175B'] = [96, 12288, 96, 128, 4, 1]   # GPT-3 Table 2.1
    model_table['GPT-89B'] = [48, 12288, 96, 128, 4, 1]
    model_table['GPT-13B'] = [40, 5120, 40, 128, 4, 1]     # GPT-3 Table 2.1 (5120=40x128)
    model_table['LLAMA-7B'] = [32, 4096, 32, 128, 8 / 3, 1]  # LLaMA-1 Table 2
    # Llama-3-8B (arXiv:2407.21783): 32 Q heads sharing 8 KV heads (group 4),
    # FFN 14336; the GQA sibling of LLAMA-7B (ruling chenyi9 2026-08-27).
    model_table['LLAMA3-8B'] = [32, 4096, 32, 128, 3.5, 4]
    # LLaMA-1-33B (arXiv:2302.13971 Table 2): 60 layers, dim 6656, 52 heads
    # (no GQA); the second medium model beside GPT-13B (chenyi9 2026-08-29).
    model_table['LLAMA-33B'] = [60, 6656, 52, 128, 8 / 3, 1]
    model_table['LLAMA-65B'] = [80, 8192, 64, 128, 8 / 3, 1]  # LLaMA-1 Table 2
    model_table['MT-76B'] = [60, 10240, 40, 128, 4, 1]
    model_table['MT-146B'] = [80, 12288, 80, 128, 4, 1]
    model_table['MT-310B'] = [96, 16384, 128, 128, 4, 1]
    model_table['MT-530B'] = [105, 20480, 128, 160, 4, 1]
    model_table['MT-1008B'] = [128, 25600, 160, 160, 4, 1]
    model_table['OPT-66B'] = [64, 9216, 72, 128, 4, 1]

    ndec, hdim, nheads, dhead, ff_scale, gqa_size = model_table[name]
    config = {
        'name': name,
        'ndec': ndec,
        'hdim': hdim,
        'num_heads': nheads,
        'dhead': dhead,
        'ff_scale': ff_scale,
        'gqa_size': gqa_size,
        'dtype': dtype
    }
    return config
