from src.type import *
from src.model import *
import math
from src.ramulator_wrapper import *
from src.gemm_table import gemm_efficiency, attention_efficiency


class xPU:

    def __init__(self, name: DeviceType, config, scaling_factor):
        self.name = name
        self.gpu_type = None
        if self.name == DeviceType.GPU:
            self.gpu_type = config['GPUTYPE']

        self.num_xpu = config['NUM_DEVICE']
        self.num_core = config['NUM_CORE']
        self.peak_flops = config['FLOPS_PER_DEVICE']
        self.peak_memory_bandwidth = config['OFF_MEM_BW_PER_DEVICE']
        self.peak_l2_bandwidth = config['L2_MEM_BW_PER_DEVICE']
        self.l1_cache_size = config['L1_CAP_PER_CORE']
        self.l2_cache_size = config['L2_CAP_PER_DEVICE']
        self.max_interface_bandwidth = config['INTERFACE_BW']
        self.pim_link_bandwidth = (config.get('PIM_LINK_BW') or
                                   self.max_interface_bandwidth)
        self.aggregate_memory_capacity = config[
            'MEM_CAPACITY_PER_DEVICE'] * self.num_xpu

        self.max_compute_util = scaling_factor['MAX_COMPUTE_UTIL']
        self.max_memory_util = scaling_factor['MAX_OFF_MEM_BW_UTIL']
        self.energy_table = config['ENERGY_TABLE']

        # Refined GPU model (see config.make_xpu_config): only the GPU has
        # one; the CPU xPU keeps the legacy formulas.
        self.gpu_model = config.get('GPU_MODEL', 'legacy')
        self.hbm_stream_eff = config.get('HBM_STREAM_EFF', self.max_memory_util)
        self.nvlink_latency = config.get('NVLINK_LATENCY_S', 0.0)
        self.attn_q_block = config.get('ATTN_Q_BLOCK', 128)
        self.attn_mma_rows = config.get('ATTN_MMA_ROWS', 16)
        self.attn_decode_split = config.get('ATTN_DECODE_SPLIT', 256)
        self.attn_splitk = config.get('ATTN_SPLITK', False)
        hbm_spec = config.get('HBM_SPEC')
        # Streaming rate of the far (AttAcc) HBM when the GPU pulls K/V over
        # the link: same stack count and device as the near memory.
        self.far_hbm_bandwidth = (
            config.get('NUM_HBM_STACKS', 5) * hbm_spec['BYTES_PER_S'] *
            self.hbm_stream_eff if hbm_spec else float('inf'))

        self.table_tiles = {}

    @property
    def refined(self):
        """cuBLAS-table GEMMs + NVLink-latency transfers (refined and flash)."""
        return self.name == DeviceType.GPU and self.gpu_model in ('refined', 'flash')

    @property
    def flash(self):
        """Attention priced as a fused FlashAttention-2 kernel."""
        return self.name == DeviceType.GPU and self.gpu_model == 'flash'

    # ------------------------------------------------------------------
    # FlashAttention (``flash``) helpers
    # ------------------------------------------------------------------
    def _wave_util(self, blocks):
        """SM occupancy of ``blocks`` thread blocks with wave quantisation."""
        blocks = max(int(blocks), 1)
        return blocks / (math.ceil(blocks / self.num_core) * self.num_core)

    def _attn_key_length(self, layer: Layer):
        m, n, k, numOp, dbyte = layer.get_infos()
        if 'score' in layer.name:
            return n  # S = Q K^T: n keys, k = head dim
        if 'context' in layer.name:
            return k  # O = P V: k keys, n = head dim
        return max(n, k)

    def _attn_shape(self, layer: Layer):
        """(thread blocks, padded query rows, key length) of a fused attention.

        Prefill: one CTA per (head, request, ``ATTN_Q_BLOCK`` query rows) --
        the FlashAttention-2 decomposition.  Decode (m == 1): flash-decoding
        splits the key dimension into ``ATTN_DECODE_SPLIT``-key CTAs instead.
        The MMA pads a short Q block to ``ATTN_MMA_ROWS`` rows.
        """
        m, n, k, numOp, dbyte = layer.get_infos()
        keys = self._attn_key_length(layer)
        if m <= 1:
            blocks = numOp * math.ceil(keys / self.attn_decode_split)
        else:
            blocks = numOp * math.ceil(m / self.attn_q_block)
            if self.attn_splitk and blocks < self.num_core:
                # choose the key split that maximises efficiency x occupancy
                best = (attention_efficiency(keys) * self._wave_util(blocks), 1)
                for split in (2, 4, 8, 16, 32, 64):
                    if keys / split < self.attn_decode_split:
                        break
                    score = (attention_efficiency(keys / split) *
                             self._wave_util(blocks * split))
                    if score > best[0]:
                        best = (score, split)
                blocks *= best[1]
                keys = keys / best[1]
        padded_m = math.ceil(m / self.attn_mma_rows) * self.attn_mma_rows
        return blocks, padded_m, keys

    def _flash_traffic(self, layer: Layer):
        """(off-chip, L2, L1, reg) bytes of a fused attention MATMUL / softmax.

        Q, K, V and O cross HBM once; the S = QK^T tile stays on the SM, so
        the softmax has no off-chip traffic.  K/V are re-read from L2 once
        per Q block.
        """
        m, n, k, numOp, dbyte = layer.get_infos()
        if layer.type == LayerType.SOFTMAX:
            on_chip = sum(layer.get_size())
            return [0, 0, 0], [0], [on_chip], [on_chip]
        q_blocks = math.ceil(max(m, 1) / self.attn_q_block)
        if 'score' in layer.name:
            off = [m * k, n * k, 0]
            l2 = [m * k, q_blocks * n * k, 0]
        else:
            off = [0, n * k, m * n]
            l2 = [0, q_blocks * n * k, m * n]
        off = [i * dbyte * numOp for i in off]
        l2 = [i * dbyte * numOp for i in l2]
        reg = [m * n * k, m * n * k, m * n * k]
        return off, l2, list(l2), reg

    def _flash_compute_time(self, layer: Layer):
        m, n, k, numOp, dbyte = layer.get_infos()
        if layer.type == LayerType.SOFTMAX:
            return 0.0  # fused; its cost is inside the attention efficiency
        peak = self.peak_flops * int(2 / dbyte)
        blocks, padded_m, keys = self._attn_shape(layer)
        eff = attention_efficiency(keys) * self._wave_util(blocks)
        return 2 * padded_m * n * k * numOp / (peak * eff)

    def _flash_mem_time(self, layer: Layer):
        off_data, l2_data, l1_data, reg_data = self._get_traffic(layer)
        layer.off_traffic = sum(off_data)
        if layer.type == LayerType.SOFTMAX:
            return 0, 0, 0, 0
        blocks, _, _ = self._attn_shape(layer)
        mem_bw = (self.peak_memory_bandwidth * self.max_memory_util *
                  self._wave_util(blocks))
        return sum(off_data) / mem_bw, sum(l2_data) / self.peak_l2_bandwidth, 0, 0

    def _get_traffic_for_tile(self, tm, tn, layer: Layer):
        m, n, k, numOp, dbyte = layer.get_infos()
        traffic = [math.ceil(n / tn) * m * k, math.ceil(m / tm) * n * k, m * n]
        traffic = [i * dbyte * numOp for i in traffic]
        return traffic

    def _get_optimal_tile(self, layer: Layer):
        m, n, k, numOp, dbyte = layer.get_infos()
        config = (m, n, k, numOp, dbyte)
        if config in self.table_tiles.keys():
            return self.table_tiles[config]

        trange = [8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512]

        # find L1 tile size
        l1_tm = 0
        l1_tn = 0
        l1_tk = 32
        opt_config = [0, 0]
        min_cost = float('inf')
        for l1_tm in trange:
            for l1_tn in trange:
                l1_tm = min(l1_tm, m)
                l1_tn = min(l1_tn, n)
                required_capacity = (
                    l1_tm + l1_tn) * l1_tk * dbyte + l1_tm * l1_tn * dbyte
                if required_capacity > self.l1_cache_size:
                    continue
                l2_access = sum(self._get_traffic_for_tile(l1_tm, l1_tn, layer))

                ## applying SM underutilization to cost function
                num_threadblock = numOp
                if layer.type == LayerType.FC:
                    num_threadblock = math.ceil(m / l1_tm) * math.ceil(
                        n / l1_tn) * numOp

                tmp = math.ceil(num_threadblock / self.num_core) * self.num_core
                core_utilization = num_threadblock / tmp
                cost = l2_access * pow((1 / core_utilization), 2)
                if cost < min_cost:
                    min_cost = cost
                    opt_config = [l1_tm, l1_tn]

        l1_tm, l1_tn = opt_config

        # find L2 tile size
        ## experimentally found L2 tile_k size
        l2_tk = k / 64

        min_access = float('inf')
        opt_config = [0, 0]
        for l2_tm in [l1_tm * i for i in range(1, int(m / l1_tm) + 1)] + [m]:
            for l2_tn in [l1_tn * i for i in range(1,
                                                   int(n / l1_tn) + 1)] + [n]:
                l2_tm = min(l2_tm, m)
                l2_tn = min(l2_tn, n)
                required_capacity = (
                    l2_tm + l2_tn) * l2_tk * dbyte + l2_tm * l2_tn * dbyte
                if required_capacity > self.l2_cache_size:
                    if l2_tm != l1_tm or l2_tn != l1_tn:
                        continue

                access = math.ceil(m / l2_tm) * n * k * dbyte + \
                          math.ceil(n / l2_tn) * m * k * dbyte + m * n * dbyte

                if access < min_access:
                    min_access = access
                    opt_config = [l2_tm, l2_tn]

        l2_tm, l2_tn = opt_config
        out_tiles = [l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk]
        self.table_tiles[config] = out_tiles
        return out_tiles

    def _get_traffic(self, layer: Layer):
        # return tuple of 4 elements (off-mem, L2, L1, reg)
        m, n, k, numOp, dbyte = layer.get_infos()
        if self.flash and layer.type in (LayerType.MATMUL, LayerType.SOFTMAX):
            return self._flash_traffic(layer)
        if layer.type in [
                LayerType.SOFTMAX, LayerType.ACT, LayerType.NORM, LayerType.G2G,
                LayerType.X2G
        ]:
            data = layer.get_size()
            return data, data, data, data

        elif layer.type in [LayerType.FC, LayerType.MATMUL]:
            l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk = self._get_optimal_tile(
                layer)
            reg_tm, reg_tn, reg_tk = 16, 16, 32

            off_data = self._get_traffic_for_tile(l2_tm, l2_tn, layer)
            l2_data = self._get_traffic_for_tile(l1_tm, l1_tn, layer)
            l1_data = self._get_traffic_for_tile(reg_tm, reg_tn, layer)
            reg_data = [m * n * k, m * n * k, m * n * k]

            return off_data, l2_data, l1_data, reg_data

        else:
            assert 0, "Invalid layer type"

    def _compute_time(self, layer: Layer):
        if self.flash and layer.type in (LayerType.MATMUL, LayerType.SOFTMAX):
            return self._flash_compute_time(layer)
        l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk = self._get_optimal_tile(layer)
        m, n, k, numOp, dbyte = layer.get_infos()
        flops = self.peak_flops * self.max_compute_util
        if self.refined and layer.type in (LayerType.FC, LayerType.MATMUL):
            # Refined model: the flat MAX_COMPUTE_UTIL (0.8) is replaced by
            # the cuBLAS efficiency measured for this GEMM size (projections
            # and the attention score/context matmuls alike).  Everything
            # else -- tiling, the per-layer SM occupancy below, the memory
            # model -- is the legacy AttAcc formulation, so the only effect
            # is how much slower the same matrix computation runs at the
            # size-dependent intensity.
            flops = self.peak_flops * gemm_efficiency(m, n, k)
        if self.name == DeviceType.GPU:
            num_threadblock = numOp
            if layer.type == LayerType.FC:
                num_threadblock = math.ceil(m / l1_tm) * math.ceil(
                    n / l1_tn) * numOp

            tmp = math.ceil(num_threadblock / self.num_core) * self.num_core
            core_utilization = num_threadblock / tmp

            flops = flops * core_utilization

        ## e.g., peak flops of FP8  is twice that of FP16
        flops *= int(2 / dbyte)

        if flops == 0:
            import pdb
            pdb.set_trace()

        return layer.get_flops() / flops

    def _mem_time(self, layer: Layer):
        if self.flash and layer.type in (LayerType.MATMUL, LayerType.SOFTMAX):
            return self._flash_mem_time(layer)
        l1_tm, l1_tn, l1_tk, l2_tm, l2_tn, l2_tk = self._get_optimal_tile(layer)
        m, n, k, numOp, dbyte = layer.get_infos()

        off_data, l2_data, l1_data, reg_data = self._get_traffic(layer)
        layer.off_traffic = sum(off_data)

        mem_bw = self.peak_memory_bandwidth * self.max_memory_util
        if self.name == DeviceType.GPU:
            if layer.type == LayerType.ACT:
                exec_time = (
                    0.000000447 *
                    (1555 * 1000 * 1000 * 1000 / self.peak_memory_bandwidth) *
                    sum(off_data) + 8.29) / 1000 / 1000
                return exec_time, 0, 0, 0

            elif layer.type == LayerType.NORM:
                exec_time = (
                    0.0000016 *
                    (1555 * 1000 * 1000 * 1000 / self.peak_memory_bandwidth) *
                    sum(off_data) + 6.87) / 1000 / 1000
                return exec_time, 0, 0, 0

            else:
                num_threadblock = numOp
                if layer.type == LayerType.FC:
                    num_threadblock = math.ceil(m / l1_tm) * math.ceil(
                        n / l1_tn) * numOp

                tmp = math.ceil(num_threadblock / self.num_core) * self.num_core
                core_utilization = num_threadblock / tmp
                mem_bw = mem_bw * core_utilization

                return sum(off_data) / mem_bw, sum(
                    l2_data) / self.peak_l2_bandwidth, 0, 0
        else:
            return sum(off_data) / mem_bw, sum(
                l2_data) / self.peak_l2_bandwidth, 0, 0

    def _exec_time(self, layer: Layer):
        compute_time = self._compute_time(layer)
        mem_time = max(*self._mem_time(layer))
        max_time = 0
        if compute_time > mem_time:
            max_time = compute_time
            layer.bound = "compute"
        else:
            max_time = mem_time
            layer.bound = "memory"
        layer.time = max_time

        return max_time

    def _get_energy(self, layer: Layer):
        off_data, l2_data, l1_data, reg_data = self._get_traffic(layer)
        if self.name == DeviceType.CPU:
            energy_per_acc = self.energy_table['mem']
            e_off = sum(off_data) * energy_per_acc
            e_flop = layer.get_flops() / 2 * self.energy_table['alu']
            energies = [e_off, 0, 0, 0, e_flop, 0]
        else:
            e_off = sum(off_data) * self.energy_table['mem']
            e_l2 = sum(l2_data) * self.energy_table['l2']
            e_l1 = sum(l1_data) * self.energy_table['l1']
            e_reg = sum(reg_data) * self.energy_table['reg']
            e_flop = layer.get_flops() / 2 * self.energy_table['alu']
            energies = [e_off, e_l2, e_l1, e_reg, e_flop, 0]
        energies = [i * self.num_xpu for i in energies]
        return energies

    def _io_time_energy(self, layer: Layer):
        m, n, k, numOp, dbyte = layer.get_infos()

        def get_nvlink_time(size):
            # interpolation of real data on A100
            # size unit: Byte
            if size == 0:
                return 1
            else:
                approx_ns_time = 6060 + 0.009 * size * (
                    (600 * 1000 * 1000 * 1000 / self.max_interface_bandwidth))
                approx_time = approx_ns_time / 1000 / 1000 / 1000
                return max(approx_time,
                           size / (self.max_interface_bandwidth / 2))

        if self.name == DeviceType.CPU:
            ## RX, TX --> 1/2x
            bw = self.max_interface_bandwidth / 2
            traffic = m * n * numOp * dbyte
            exec_time = traffic / bw
            # we ignore CPU energy
            energy = 0
        else:
            ## each GPU has partial sum of output.
            traffic = m * n * numOp * dbyte
            interface_bw = self.max_interface_bandwidth / 2
            if layer.type == LayerType.X2G:
                exec_time = traffic / (self.pim_link_bandwidth / 2)
                if self.refined:
                    # K/V (or Q / context) moving between the GPU and the
                    # AttAcc: one NVLink latency per transfer, and the far
                    # HBM3 has to stream the bytes as well as the link.
                    exec_time = self.nvlink_latency + max(
                        exec_time, traffic / self.far_hbm_bandwidth)
            else:
                ## allreduce
                exec_time = get_nvlink_time(
                    traffic / self.num_xpu) * (self.num_xpu - 1)

            # all reduce communication
            energy = self.num_xpu * traffic * self.energy_table['comm']
        return exec_time, [0, 0, 0, 0, 0, energy]

    def get_time_and_energy(self, layer: Layer):
        if layer.type in [LayerType.X2G, LayerType.G2G]:
            return self._io_time_energy(layer)
        else:
            return self._exec_time(layer), self._get_energy(layer)


class PIM:

    def __init__(self, config, scaling_factor, ramulator):
        self.name = DeviceType.PIM
        self.num_attacc = config['NUM_ATTACC']
        self.num_hbm = config['NUM_HBM']
        self.pim_type = config['PIM_TYPE']
        self.peak_memory_bandwidth = config['MEM_BW_PER_HBM'] * self.num_hbm
        self.softmax_peak_flops = config['SOFTMAX_FLOPS']
        self.softmax_peak_bandwidth = config['SOFTMAX_MEM_BW']
        self.max_interface_bandwidth = config['INTERFACE_BW']
        self.aggregate_memory_capacity = config[
            'MEM_CAPACITY_PER_HBM'] * self.num_attacc * self.num_hbm
        self.energy_table = config['ENERGY_TABLE']
        self.io_energy_table = self.energy_table['io']
        self.power_constraint = config['POWER_CONSTRAINT']
        self.ramulator = ramulator

    def _get_traffic(self, layer: Layer):
        # return tuple of 4 elements (off-mem, L2, L1, reg)
        m, n, k, numOp, dbyte = layer.get_infos()
        if layer.type in [LayerType.MATMUL, LayerType.FC, LayerType.SOFTMAX]:
            data = layer.get_size()
            return data, [0], [0], [0]

        else:
            assert 0, "In get_traffic function, PIM could not support this layer"

    def _io_time_energy(self, layer: Layer):
        m, n, k, numOp, dbyte = layer.get_infos()
        interface_bw = self.max_interface_bandwidth / 2
        traffic = m * n * numOp * dbyte
        exec_time = traffic / interface_bw

        energy = traffic * self.energy_table['comm'] * self.num_attacc

        return exec_time, [0, 0, 0, 0, 0, energy]

    def _compute_time(self, layer: Layer):
        flops = self.softmax_peak_flops
        flops *= int(2 / layer.dbyte)
        compute_time = layer.get_flops() / flops
        return compute_time

    def _mem_time(self, layer: Layer):
        mem_bw = self.softmax_peak_bandwidth
        mem_time = sum(layer.get_size()) / mem_bw
        return mem_time

    def _get_energy(self, layer: Layer):
        off_data = layer.get_size()
        e_off = sum(off_data) * self.energy_table['sram'] * self.num_attacc
        e_flop = layer.get_flops(
        ) / 2 * self.energy_table['alu'] * self.num_attacc

        return [e_off, 0, 0, 0, e_flop, 0]

    def get_time_and_energy(self, layer: Layer):
        if layer.type == LayerType.X2G:
            return self._io_time_energy(layer)

        elif layer.type == LayerType.MATMUL:
            ## operational granularity = the attention layer
            if 'score' in layer.name:
                m, n, k, numOp, dbyte = layer.get_infos()
                time, traffic = self.ramulator.output(
                    self.pim_type, layer, self.power_constraint)
                io_energy = 0
                for i in range(len(self.io_energy_table)):
                    io_energy += traffic[i] * self.io_energy_table[i]

                energy_per_access = self.energy_table['mem']
                cell_energy = traffic[-1] * energy_per_access
                dram_energy = cell_energy + io_energy
                cal_energy = layer.get_flops() / 2 * self.energy_table['alu']

                energies = [dram_energy, 0, 0, 0, cal_energy, 0]
                energies = [i * self.num_attacc for i in energies]

                return time, energies
            else:
                return 0, [0, 0, 0, 0, 0, 0]

        elif layer.type == LayerType.SOFTMAX:
            # Execution time
            compute_time = self._compute_time(layer)
            mem_time = self._mem_time(layer)

            if compute_time > mem_time:
                layer.bound = 'compute'
            else:
                layer.bound = 'memory'
            exec_time = max(compute_time, mem_time)
            layer.time = exec_time

            energy = self._get_energy(layer)

            return exec_time, energy

        else:
            assert 0, "PIM does not support this layer."

    def get_time_and_energy_runs(self, layer: Layer):
        """Return one timing result per address-resolved CacheBlend extent."""
        if layer.type != LayerType.MATMUL or "score" not in layer.name:
            return [self.get_time_and_energy(layer)]
        measured = self.ramulator.output_runs(self.pim_type, layer,
                                              self.power_constraint)
        # One result per CHANNEL when the layer carries extent groups; the
        # energy split follows each channel's real row count.
        extent_groups = getattr(layer, "pim_kv_extent_groups", None)
        if extent_groups:
            run_lengths = [sum(rows for _, _, rows in extents)
                           for _, _, extents in extent_groups]
        else:
            run_lengths = [run[2] for run in getattr(layer, "pim_kv_runs", ())]
        total_rows = sum(run_lengths)
        results = []
        for index, (time, traffic) in enumerate(measured):
            io_energy = sum(traffic[index] * self.io_energy_table[index]
                            for index in range(len(self.io_energy_table)))
            dram_energy = traffic[-1] * self.energy_table['mem'] + io_energy
            fraction = (run_lengths[index] / total_rows
                        if total_rows else 1 / len(measured))
            cal_energy = (layer.get_flops() * fraction / 2 *
                          self.energy_table['alu'])
            energy = [dram_energy, 0, 0, 0, cal_energy, 0]
            results.append((time, [value * self.num_attacc for value in energy]))
        return results
