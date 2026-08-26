import pandas as pd
import subprocess
import math
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from src.config import *
from src.model import *
from src.type import *


# --- MQ-MAC batch command timing (PLAN_mq_command.md §2.1; 2026-08-23 rev) --
# One MAC_AB reads a column once and the bank PE multiplies it against every
# resident Q internally.  Design principle (chenyi9, 2026-08-23): the DRAM
# command cadence is NEVER stretched by compute -- the column stream runs at
# the preset nCCDAB and the PE computes in the slack, raising its clock as
# needed.  Precedent: Samsung's FIMDRAM executes PIM ops strictly off the
# command stream ("each memory command increments the CRF PC") and its
# official PIMSimulator carries no PIM-specific timing at all (standard
# tCCDL=4).  The nCCDAB presets (6 PC / 4 NPC) are DRAM-read-side
# constraints with no compute term (the earlier equal-power stretch of
# compute energy into the interval was removed as contrary to both the
# design intent and the precedent).  PE power is accounted SEPARATELY
# against the stack budget: see mq_pe_power_w / MQ_POWER_BUDGET_W.
_MQ_TCK_NS = 0.769                     # command-clock period used everywhere
_MQ_NCCDAB_PC = 6                      # HBM3_5.2Gbps preset, power-constrained
_MQ_NCCDAB_NPC = 4                     # HBM3_5.2Gbps_NPC preset
MQ_DEFAULT_PE_FREQ_GHZ = 0.666         # AttAcc's synthesized GEMV-unit clock
MQ_DEFAULT_GEMV_BUFFER_BYTES = 512     # AttAcc's 16 x 256-bit GEMV buffer
# One Q occupies 64 B per bank in the score phase (d_head=128, BF16, split
# over the 4 banks of a bank group), i.e. two 32-B buffer entries.
MQ_QUERY_SLICE_BYTES = 64
# Stack-level power budget: the red line of AttAcc Fig. 7(a), computed from
# the JEDEC JESD238 IDD7 all-bank interleave-read loop (hand-read from the
# figure, 2026-08-23).  Used only to CHECK the separately-accounted PE
# power increment (mq_pe_power_w); it does not enter the interval.
MQ_POWER_BUDGET_W = 116
# Cell-side microscopic energies (FGDRAM-based ENERGY_TABLE):
_MQ_E_COL_PJ = ENERGY_TABLE['PIM'][PIMType.BA]['mem'] * 32   # one 32-B read
_MQ_E_Q_PJ = (16 * ENERGY_TABLE['PIM'][PIMType.BA]['alu'] +  # 16-lane MAC
              32 * ENERGY_TABLE['PIM'][PIMType.BA]['sram'])  # buffer read


def mq_query_capacity(gemv_buffer_bytes=MQ_DEFAULT_GEMV_BUFFER_BYTES):
    """Resident-Q capacity of one bank's GEMV buffer (the sweep splits beyond).

    Q is the ONLY capacity-bound operand (ruling 2026-08-24): a Q slice is
    reused across every K column of the bank, so it must stay resident and
    n_q = S/64 -- the stock 512-B buffer holds 8, more queries need a larger
    buffer (SRAM + PE area/power).  The context-phase P vector has (almost)
    no per-bank reuse and therefore streams through the double-buffered
    halves via MV_GB; its bound is the movement-bus bandwidth and direction
    turnaround (priced physically in the trace), never this capacity.
    """
    return max(1, int(gemv_buffer_bytes) // MQ_QUERY_SLICE_BYTES)


def mq_interval_cycles(shared_queries, power_constraint,
                       pe_freq_ghz=MQ_DEFAULT_PE_FREQ_GHZ):
    """Effective nCCDAB (in command cycles) of one MQ-MAC command.

    interval = max(preset floor, PE-throughput term).  The preset floor
    (6 PC / 4 NPC) is a DRAM-read-side constraint and is NEVER stretched
    by compute; the PE term ceil(n/(f*tCK)) only matters when the PE is
    slower than the matching requirement f*(n) = n/(floor*tCK) -- e.g.
    AttAcc's stock 666 MHz PE running MQ unchanged.  Compute power is
    accounted separately (mq_pe_power_w vs MQ_POWER_BUDGET_W).
    """
    n = max(1, int(shared_queries))
    pe_cycles = math.ceil(n / (float(pe_freq_ghz) * _MQ_TCK_NS))
    floor = _MQ_NCCDAB_PC if power_constraint else _MQ_NCCDAB_NPC
    return max(floor, pe_cycles)


def mq_pe_power_w(shared_queries, interval_cycles, active_banks=16 * 64):
    """Stack-level PE dynamic-power increment of an MQ sweep, in watts.

    Per bank and per column read the PE spends n * _MQ_E_Q_PJ (n 16-lane
    MACs plus n 32-B buffer reads, cell-side FGDRAM scale); one command
    lands every interval_cycles.  Default active_banks = 16 banks/pCH x
    64 pCH (a whole 8-Hi stack scanning).  Compare against
    MQ_POWER_BUDGET_W minus the column-stream draw -- the check is
    reported, not enforced here.
    """
    n = max(1, int(shared_queries))
    interval_ns = max(1, int(interval_cycles)) * _MQ_TCK_NS
    return active_banks * n * _MQ_E_Q_PJ / (interval_ns * 1000.0)


class Ramulator:

    def __init__(self,
                 modelinfos,
                 ramulator_dir,
                 output_log='',
                 fast_mode=False,
                 num_hbm=5,
                 workers=1,
                 signature_cache=True):
        self.df = pd.DataFrame()
        self.ramulator_dir = ramulator_dir
        self.output_log = output_log
        if os.path.exists(output_log):
            self.df = pd.read_csv(output_log)
        self.tCK = 0.769  # ns
        self.num_hbm = num_hbm
        self.nhead = modelinfos['num_heads']
        self.dhead = modelinfos['dhead']
        self.fast_mode = fast_mode
        if workers < 1:
            raise ValueError("Ramulator workers must be at least one")
        # This controls host-side simulation parallelism only.  It is not a
        # hardware-channel count and therefore must not change cycle/energy
        # aggregation below.
        self.workers = workers
        self._trace_name_lock = Lock()
        self._trace_call_index = 0
        # Cache only TLB-resolved reuse runs.  The original no-reuse path
        # continues to use AttAcc's existing CSV cache unchanged.
        self.signature_cache_enabled = signature_cache
        self._signature_cache = {}
        self._signature_cache_lock = Lock()
        self._signature_cache_hits = 0
        self._signature_cache_misses = 0
        self._ramulator_invocations = 0

    @staticmethod
    def _address_mapping_signature(address):
        """HBM3-PIM fields that influence a fresh-trace command schedule.

        Ramulator's HBM3-PIM mapper has 32-B columns; 32 columns per row;
        then bank, bank-group, rank, pseudo-channel and channel fields.  A
        standalone run starts with no open row, and the model has no
        row-number-dependent timing.  We therefore intentionally omit only
        the absolute row index, while retaining every bank-selection field and
        the byte/column offset that determines row-boundary crossings.
        """
        if address is None:
            return None
        column_bytes = 32
        row_bytes = 32 * column_bytes
        bank_bytes = (1 << 14) * row_bytes
        bank_group_bytes = 4 * bank_bytes
        rank_bytes = 4 * bank_group_bytes
        pseudo_channel_bytes = 2 * rank_bytes
        channel_bytes = 2 * pseudo_channel_bytes
        return (
            (address // channel_bytes) % 16,
            (address // pseudo_channel_bytes) % 2,
            (address // rank_bytes) % 2,
            (address // bank_group_bytes) % 4,
            (address // bank_bytes) % 4,
            address % row_bytes,
        )

    def _run_signature(self, pim_type, run_length, num_ops_per_hbm, dbyte,
                       power_constraint, key_addr, value_addr, channel_count,
                       shared_kv, shared_queries, channel_base=None,
                       mq_command=False, nccdab_override=None):
        return (
            pim_type.name, run_length, num_ops_per_hbm, dbyte,
            bool(power_constraint), self.dhead, self.num_hbm, channel_count,
            bool(shared_kv), shared_queries, channel_base,
            bool(mq_command), nccdab_override,
            self._address_mapping_signature(key_addr),
            self._address_mapping_signature(value_addr),
        )

    def cache_report(self):
        """Host-side signature-cache counters for workload reports/tests."""
        with self._signature_cache_lock:
            return {
                "enabled": self.signature_cache_enabled,
                "hits": self._signature_cache_hits,
                "misses": self._signature_cache_misses,
                "ramulator_invocations": self._ramulator_invocations,
                "entries": len(self._signature_cache),
            }

    def _unique_trace_stem(self, base: str) -> str:
        """Return a process-local unique stem for generated Ramulator files.

        Multiple PIM layers often have the same shape.  With host workers
        enabled, a shape-derived name alone lets two independent calls race on
        the same ``.trace``/``.yaml`` path.  The suffix is bookkeeping only;
        it has no effect on the command stream or simulation result.
        """
        with self._trace_name_lock:
            index = self._trace_call_index
            self._trace_call_index += 1
        # The process id keeps concurrently running simulations (e.g. several
        # sbatch jobs sharing one checkout) from clobbering each other's
        # trace/YAML files.
        return "{}_p{}_call{}".format(base, os.getpid(), index)

    def make_yaml_file(self, yaml_file, file_name, power_constraint,
                       nccdab_override=None):
        trace_path = os.path.join(self.ramulator_dir, file_name + ".trace")
        line = ""
        line += "Frontend:\n"
        line += "  impl: PIMLoadStoreTrace\n"
        line += "  path: {}\n".format(trace_path)
        line += "  clock_ratio: 1\n"
        line += "\n"
        line += "  Translation:\n"
        line += "    impl: NoTranslation\n"
        line += "    max_addr: 2147483648\n"
        line += "              \n"
        line += "\n"
        line += "MemorySystem:\n"
        line += "  impl: PIMDRAM\n"
        line += "  clock_ratio: 1\n"
        line += "  DRAM:\n"
        line += "    impl: HBM3-PIM\n"
        line += "    org:\n"
        line += "      preset: HBM3_8Gb_2R\n"
        line += "      channel: 16\n"
        line += "    timing:\n"
        if power_constraint:
            line += "      preset: HBM3_5.2Gbps\n"
        else:
            line += "      preset: HBM3_5.2Gbps_NPC\n"
        if nccdab_override is not None:
            # MQ-MAC command: one MAC_AB carries every resident Q, so its
            # effective command-to-command interval is set per run (power
            # stretch and PE throughput, see mq_interval_cycles).  Ramulator2
            # natively overwrites preset timings with user-provided values.
            line += "      nCCDAB: {}\n".format(int(nccdab_override))
        line += "\n"
        line += "  Controller:\n"
        line += "    impl: HBM3-PIM\n"
        line += "    Scheduler:\n"
        line += "      impl: PIM\n"
        line += "    RefreshManager:\n"
        line += "      impl: AllBankHBM3\n"
        line += "      #impl: No\n"
        line += "    plugins:\n"
        line += "\n"
        line += "  AddrMapper:\n"
        line += "    impl: HBM3-PIM\n"
        with open(yaml_file, 'w') as f:
            f.write(line)

    def update_log_file(self, log):
        if self.df.empty:
            if os.path.exists(self.output_log):
                df = pd.read_csv(self.output_log)
            else:
                columns = [
                    'L', 'nhead', 'dhead', 'dbyte', 'pim_type',
                    'power_constraint', 'cycle', 'mac', 'softmax', 'mvgb',
                    'mvsb', 'wrgb'
                ]
                df = pd.DataFrame(columns=columns)
        else:
            df = self.df
        if len(df.columns) > 12:
            import pdb
            pdb.set_trace()
        new_df = pd.DataFrame(columns=df.columns)
        new_df.loc[0] = log
        df = pd.concat([df, new_df]).drop_duplicates()
        self.df = df
        self.df.to_csv(self.output_log, index=False)

    #def run_ramulator(self):
    def run_ramulator(self, pim_type: PIMType, l, num_ops_per_hbm, dbyte,
                      yaml_file, file_name, key_addr=None, value_addr=None,
                      channel_count=16, shared_kv=False, shared_queries=1,
                      channel_base=None, mq_command=False, phase="full"):
        pim_type_name = pim_type.name.lower(
        ) if not pim_type == PIMType.BA else "bank"
        trace_file = os.path.join(self.ramulator_dir, file_name + '.trace')

        trace_exc = os.path.join(
            self.ramulator_dir,
            "trace_gen/gen_trace_attacc_{}.py".format(pim_type_name))
        trace_args = "--dhead {} --nhead {} --seqlen {} --dbyte {} --output {}".format(
            self.dhead, num_ops_per_hbm, l, dbyte, trace_file)
        if key_addr is not None:
            trace_args += " --key-addr 0x{:x}".format(key_addr)
        if value_addr is not None:
            trace_args += " --value-addr 0x{:x}".format(value_addr)
        if channel_count != 16:
            trace_args += " --channels {}".format(channel_count)
        if channel_base is not None and channel_count != 16:
            # Heads stripe with wrap-around inside [base, base + channels).
            trace_args += " --pool-base {}".format(channel_base)
        if shared_kv:
            trace_args += " --shared-kv"
        if shared_queries != 1:
            trace_args += " --shared-queries {}".format(shared_queries)
        if mq_command:
            trace_args += " --mq"
        if phase != "full":
            trace_args += " --phase {}".format(phase)

        gen_trace_cmd = f"python3 {trace_exc} {trace_args}"

        # TraceGen prints a header per invocation.  It is not simulation data
        # and writing it tens of thousands of times dominates experiment log
        # I/O, so discard only stdout while retaining failures on stderr.
        generated = subprocess.run(gen_trace_cmd, shell=True,
                                   stdout=subprocess.DEVNULL)
        if generated.returncode:
            raise RuntimeError("Ramulator trace generation failed: {}".format(trace_exc))

        # run ramulator
        ramulator_file = os.path.join(self.ramulator_dir, "ramulator2")
        run_ramulator_cmd = f"{ramulator_file} -f {yaml_file}"
        try:
            result = subprocess.run(run_ramulator_cmd,
                                    stdout=subprocess.PIPE,
                                    text=True,
                                    shell=True,
                                    check=True)
            output_lines = result.stdout.strip().split('\n')
            output_list = [line.strip() for line in output_lines]
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            assert 0

        try:
            os.remove(trace_file)
        except FileNotFoundError:
            pass

        # parsing output
        n_cmds = {"mac": 0, "sfm": 0, "mvgb": 0, "mvsb": 0, "wrgb": 0}
        cycle = 0
        for line in output_list:
            if "mac" in line:
                n_cmds["mac"] += int(line.split()[-1])
            elif "softmax_requests" in line:
                n_cmds["sfm"] += int(line.split()[-1])
            elif "move_to_gemv_buffer" in line:
                n_cmds["mvgb"] += int(line.split()[-1])
            elif "move_to_softmax_buffer" in line:
                n_cmds["mvsb"] += int(line.split()[-1])
            elif "write_to_gemv_buffer" in line:
                n_cmds["wrgb"] += int(line.split()[-1])
            elif "memory_system_cycles" in line:
                cycle += int(line.split()[-1])

        out = [
            cycle, n_cmds["mac"], n_cmds["sfm"], n_cmds["mvgb"], n_cmds["mvsb"],
            n_cmds["wrgb"]
        ]
        with self._signature_cache_lock:
            self._ramulator_invocations += 1
        return out

    def run(self, pim_type: PIMType, layer: Layer, power_constraint=True,
            record_log=True, per_run=False):
        if os.path.exists(self.ramulator_dir):
            l = layer.n
            dhead = self.dhead
            dbyte = layer.dbyte
            num_ops_per_attacc = layer.numOp
            num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
            num_ops_group = 1
            if self.fast_mode:
                minimum_heads = 64
                num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
                num_ops_per_hbm = minimum_heads

            file_name = "attacc_l{}_nattn{}_dhead{}_dbyte{}_pc{}".format(
                l, num_ops_per_hbm, dhead, layer.dbyte, int(power_constraint))
            file_name = self._unique_trace_stem(file_name)
            # A CacheBlend scan may cover several independently allocated
            # reusable blocks.  Run the exact contiguous K/V ranges one at a
            # time rather than falsely extending the first range over the
            # intervening physical space.  Each run yields a local softmax
            # tuple; the CacheBlend DIE event merges those tuples.
            kv_runs = getattr(layer, "pim_kv_runs", None)
            if kv_runs is None:
                kv_runs = ((getattr(layer, "pim_key_addr", None),
                            getattr(layer, "pim_value_addr", None), l),)
            shared_kv = bool(getattr(layer, "pim_shared_kv", False))
            # A CacheBlend batch has a query dimension distinct from heads.
            # Do not encode it in ``numOp``: that would change head/channel
            # placement in the trace generator.  The explicit argument keeps
            # the resident K/V addresses shared while preserving per-Q work.
            shared_queries = int(getattr(layer, "pim_shared_queries", 1))
            if shared_queries < 1:
                raise ValueError("pim_shared_queries must be positive")
            if shared_queries > 1 and not shared_kv:
                raise ValueError("pim_shared_queries requires pim_shared_kv")
            # MQ-MAC batch command (PLAN_mq_command.md): one MAC_AB serves
            # every resident Q; its command interval carries the n-fold PE
            # time and the power stretch.  'replicate' keeps the legacy
            # one-command-per-(column, query) trace expansion.
            batch_command = getattr(layer, "pim_batch_command", "replicate")
            if batch_command not in ("replicate", "mq"):
                raise ValueError("pim_batch_command must be 'replicate' or 'mq'")
            mq_command = batch_command == "mq" and shared_queries > 1
            phase = getattr(layer, "pim_phase", "full")
            if phase not in ("full", "score", "context"):
                raise ValueError("pim_phase must be 'full', 'score' or 'context'")
            nccdab_override = None
            if mq_command:
                pe_freq_ghz = float(getattr(layer, "pim_pe_freq_ghz",
                                            MQ_DEFAULT_PE_FREQ_GHZ))
                nccdab_override = mq_interval_cycles(
                    shared_queries, power_constraint, pe_freq_ghz)
            # Results are cached only for the address-resolved reuse path.
            # The CacheBlend/EPIC wrapper restarts Ramulator for every run,
            # so an equal mapping signature is exactly the same independent
            # simulator input modulo its unmodelled absolute row number.
            use_signature_cache = (self.signature_cache_enabled and
                                   getattr(layer, "pim_kv_runs", None) is not None)
            cached_results = {}
            pending_by_signature = {}
            for index, run in enumerate(kv_runs):
                if len(run) == 3:
                    key_addr, value_addr, run_length = run
                    channel_count = 16
                    channel_base = None
                else:
                    key_addr, value_addr, run_length, channel_base, channel_count = run
                signature = self._run_signature(
                    pim_type, run_length, num_ops_per_hbm, layer.dbyte,
                    power_constraint, key_addr, value_addr, channel_count,
                    shared_kv, shared_queries, channel_base,
                    mq_command, nccdab_override) + (phase,)
                if use_signature_cache:
                    with self._signature_cache_lock:
                        cached = self._signature_cache.get(signature)
                        if cached is not None:
                            self._signature_cache_hits += 1
                    if cached is not None:
                        cached_results[index] = cached
                        continue
                # With caching disabled every physical run remains a separate
                # job.  A unique key avoids accidentally deduplicating it in
                # the grouping code below.
                pending_key = signature if use_signature_cache else ("uncached", index)
                pending_by_signature.setdefault(pending_key, []).append(
                    (index, run_length, key_addr, value_addr, channel_count,
                     channel_base))

            run_jobs = []
            for signature, equivalent_runs in pending_by_signature.items():
                (index, run_length, key_addr, value_addr, channel_count,
                 channel_base) = equivalent_runs[0]
                run_file = "{}_run{}".format(file_name, index)
                yaml_file = os.path.join(self.ramulator_dir, run_file + '.yaml')
                self.make_yaml_file(yaml_file, run_file, power_constraint,
                                    nccdab_override=nccdab_override)
                run_jobs.append((signature, equivalent_runs, run_length, yaml_file,
                                 run_file, key_addr, value_addr, channel_count,
                                 channel_base))
            if use_signature_cache:
                with self._signature_cache_lock:
                    self._signature_cache_misses += len(run_jobs)
                    self._signature_cache_hits += sum(
                        len(equivalent_runs) - 1
                        for _, equivalent_runs, *_ in run_jobs)

            def execute(job):
                (_, _, run_length, yaml_file, run_file, key_addr,
                 value_addr, channel_count, channel_base) = job
                return self.run_ramulator(
                    pim_type, run_length, num_ops_per_hbm, layer.dbyte,
                    yaml_file, run_file, key_addr, value_addr, channel_count,
                    shared_kv, shared_queries, channel_base,
                    mq_command=mq_command, phase=phase)

            # Each job gets an unshared trace/YAML filename and contributes a
            # separate physical TLB run.  They are independent host jobs, so
            # parallel execution changes only wall-clock simulation time;
            # aggregation deliberately remains the previous sum.
            try:
                if self.workers == 1 or len(run_jobs) <= 1:
                    executed = [execute(job) for job in run_jobs]
                else:
                    with ThreadPoolExecutor(max_workers=min(self.workers,
                                                             len(run_jobs))) as pool:
                        executed = list(pool.map(execute, run_jobs))
            finally:
                for _, _, _, yaml_file, _, _, _, _, _ in run_jobs:
                    try:
                        os.remove(yaml_file)
                    except FileNotFoundError:
                        pass
            for job, measured in zip(run_jobs, executed):
                signature, equivalent_runs, *_ = job
                frozen = tuple(measured)
                if use_signature_cache:
                    with self._signature_cache_lock:
                        self._signature_cache[signature] = frozen
                for index, *_ in equivalent_runs:
                    cached_results[index] = frozen
            results = [cached_results[index] for index in range(len(kv_runs))]

            def postprocess(result):
                """Convert one address-resolved Ramulator result to model units."""
                cycle, mac, sfm, mvgb, mvsb, wrgb = result
                si_io = wrgb * 32  # 256 bit
                tsv_io = (wrgb + mvsb + mvgb) * 32
                giomux_io = (wrgb + mvsb + mvgb) * 32
                bgmux_io = (wrgb + mvsb + mvgb) * 32
                mem_acc = mac * 32
                if pim_type == PIMType.BA:
                    mem_acc *= 2 * 2 * 4 * 4
                elif pim_type == PIMType.BG:
                    mem_acc *= 2 * 2 * 4
                traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
                traffic = [i * self.num_hbm for i in traffic]
                traffic = [i * num_ops_group for i in traffic]
                return self.tCK * cycle / 1000 / 1000 / 1000, traffic

            # CacheBlend needs the latency of every physical extent, not only
            # their sum: independent HBM channels become independent DAG
            # resources and complete in parallel before the DIE LSE merge.
            if per_run:
                return [postprocess(result) for result in results]
            result = [sum(values) for values in zip(*results)]

            ## update log file

            log = [
                l, num_ops_per_hbm, dhead, dbyte, pim_type.name,
                power_constraint
            ] + result
            if record_log:
                self.update_log_file(log)

            return postprocess(result)

        else:
            assert 0, "Need to install ramulator"

    def output(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        # CacheBlend placement is part of the timing experiment: do not serve
        # a bank/row-resolved request from the legacy shape-only cache.
        if (getattr(layer, "pim_key_addr", None) is not None or
                getattr(layer, "pim_kv_runs", None) is not None):
            return self.run(pim_type, layer, power_constraint, record_log=False)
        if self.df.empty:
            self.run(pim_type, layer, power_constraint)

        num_ops_per_attacc = layer.numOp
        num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
        num_ops_group = 1
        if self.fast_mode:
            minimum_heads = 64
            num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
            num_ops_per_hbm = minimum_heads

        l = layer.n
        dhead = layer.k
        dbyte = layer.dbyte
        row = self.df[(self.df['L'] == l) & (self.df['nhead'] == num_ops_per_hbm) & \
                      (self.df['dbyte'] == dbyte) & (self.df['dhead'] == dhead) & \
                      (self.df['power_constraint'] == power_constraint) &  \
                      (self.df['pim_type'] == pim_type.name)]
        if row.empty:
            return self.run(pim_type, layer, power_constraint)

        else:
            cycle = int(row.iloc[0]['cycle'])
            mac = int(row.iloc[0]['mac'])
            softmax = int(row.iloc[0]['softmax'])
            mvgb = int(row.iloc[0]['mvgb'])
            mvsb = int(row.iloc[0]['mvsb'])
            wrgb = int(row.iloc[0]['wrgb'])
            si_io = wrgb * 32  # 256 bit
            tsv_io = (wrgb + mvsb + mvgb) * 32
            giomux_io = (wrgb + mvsb + mvgb) * 32
            bgmux_io = (wrgb + mvsb + mvgb) * 32
            mem_acc = mac * 32
            if pim_type == PIMType.BA:
                # pCH * Rank * bank group * bank
                mem_acc *= 2 * 2 * 4 * 4
            elif pim_type == PIMType.BG:
                # pCH * Rank * bank group
                mem_acc *= 2 * 2 * 4
            else:
                mem_acc *= 2

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            exec_time *= num_ops_group
            return exec_time, traffic

    def output_runs(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        """Return one `(time, traffic)` pair per physical CacheBlend extent.

        This deliberately bypasses the legacy shape-only CSV cache.  A run's
        channel/bank address is part of its timing input.
        """
        if getattr(layer, "pim_kv_runs", None) is None:
            return [self.output(pim_type, layer, power_constraint)]
        return self.run(pim_type, layer, power_constraint,
                        record_log=False, per_run=True)
