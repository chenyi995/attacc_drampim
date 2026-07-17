import pandas as pd
import subprocess
import math
import os
import concurrent.futures
from src.config import *
from src.model import *
from src.type import *


class Ramulator:

    def __init__(self,
                 modelinfos,
                 ramulator_dir,
                 output_log='',
                 fast_mode=False,
                 num_hbm=5,
                 rope=True,
                 num_agent=1,
                 diff_rate=0.1,
                 token_block=32,
                 sim_cores=1,
                 force_run=False):
        self.df = pd.DataFrame()
        self.ramulator_dir = ramulator_dir
        self.output_log = output_log
        if os.path.exists(output_log):
            self.df = pd.read_csv(output_log)
            if 'rope' not in self.df.columns:
                self.df['rope'] = False
            if 'num_agent' not in self.df.columns:
                self.df['num_agent'] = 0
            if 'diff_rate' not in self.df.columns:
                self.df['diff_rate'] = 0.1
            if 'token_block' not in self.df.columns:
                self.df['token_block'] = 32
            if 'v_master_diff' not in self.df.columns:
                self.df['v_master_diff'] = False
        self.tCK = 0.769  # ns
        self.num_hbm = num_hbm
        self.nhead = modelinfos['num_heads']
        self.dhead = modelinfos['dhead']
        self.fast_mode = fast_mode
        self.rope = rope
        self.num_agent = num_agent
        self.diff_rate = min(1.0, max(0.0, diff_rate))
        self.token_block = max(1, int(token_block))
        self.sim_cores = max(1, int(sim_cores))
        self.force_run = force_run
        self.v_master_diff = True

    def _effective_num_ops_per_attacc(self, layer):
        num_ops = layer.numOp
        if getattr(layer, "stage", "") == "gen":
            num_ops *= max(1, self.num_agent)
        return num_ops

    def make_yaml_file(self, yaml_file, file_name, power_constraint):
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
                    'power_constraint', 'rope', 'num_agent', 'diff_rate',
                    'token_block', 'v_master_diff', 'cycle', 'mac', 'softmax', 'mvgb', 'mvsb',
                    'wrgb'
                ]
                df = pd.DataFrame(columns=columns)
        else:
            df = self.df
        if 'rope' not in df.columns:
            df['rope'] = False
        if 'num_agent' not in df.columns:
            df['num_agent'] = 0
        if 'diff_rate' not in df.columns:
            df['diff_rate'] = 0.1
        if 'token_block' not in df.columns:
            df['token_block'] = 32
        if 'v_master_diff' not in df.columns:
            df['v_master_diff'] = False
        df = df[[
            'L', 'nhead', 'dhead', 'dbyte', 'pim_type', 'power_constraint',
            'rope', 'num_agent', 'diff_rate', 'token_block', 'v_master_diff', 'cycle', 'mac',
            'softmax', 'mvgb', 'mvsb', 'wrgb'
        ]]
        new_df = pd.DataFrame(columns=df.columns)
        new_df.loc[0] = log
        df = pd.concat([df, new_df]).drop_duplicates()
        self.df = df
        self.df.to_csv(self.output_log, index=False)

    def _parse_ramulator_output(self, output_lines):
        n_cmds = {"mac": 0, "sfm": 0, "mvgb": 0, "mvsb": 0, "wrgb": 0}
        cycle = 0
        for line in output_lines:
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

        return [
            cycle, n_cmds["mac"], n_cmds["sfm"], n_cmds["mvgb"],
            n_cmds["mvsb"], n_cmds["wrgb"]
        ]

    def _run_ramulator_binary(self, yaml_file):
        ramulator_file = os.path.join(self.ramulator_dir, "ramulator2")
        result = subprocess.run([ramulator_file, "-f", yaml_file],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                check=True)
        return [line.strip() for line in result.stdout.strip().split('\n')]

    def _split_trace_file(self, trace_file, file_name):
        with open(trace_file) as f:
            lines = f.readlines()

        num_workers = min(self.sim_cores, len(lines))
        if num_workers <= 1:
            return []

        chunk_size = math.ceil(len(lines) / num_workers)
        chunks = []
        for worker_id in range(num_workers):
            chunk_lines = lines[worker_id * chunk_size:(worker_id + 1) * chunk_size]
            if not chunk_lines:
                continue
            chunk_name = "{}_simw{}".format(file_name, worker_id)
            chunk_trace = os.path.join(self.ramulator_dir, chunk_name + ".trace")
            with open(chunk_trace, 'w') as f:
                f.writelines(chunk_lines)
            chunks.append((chunk_name, chunk_trace, len(chunk_lines)))
        return chunks

    def _run_trace_chunk(self, chunk_name, power_constraint):
        chunk_yaml = os.path.join(self.ramulator_dir, chunk_name + ".yaml")
        self.make_yaml_file(chunk_yaml, chunk_name, power_constraint)
        try:
            output = self._run_ramulator_binary(chunk_yaml)
            return self._parse_ramulator_output(output)
        finally:
            for path in [chunk_yaml, os.path.join(self.ramulator_dir, chunk_name + ".trace")]:
                if os.path.exists(path):
                    os.remove(path)

    def _run_trace_parallel(self, trace_file, file_name, yaml_file, power_constraint):
        output = self._run_ramulator_binary(yaml_file)
        return self._parse_ramulator_output(output)

    #def run_ramulator(self):
    def _make_worker(self, output_log):
        worker = Ramulator.__new__(Ramulator)
        worker.__dict__ = self.__dict__.copy()
        worker.df = pd.DataFrame()
        worker.output_log = output_log
        worker.sim_cores = 1
        worker.force_run = True
        return worker

    def _cache_key_for_layer(self, pim_type, layer, power_constraint):
        num_ops_per_attacc = self._effective_num_ops_per_attacc(layer)
        num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
        if self.fast_mode:
            minimum_heads = 64
            num_ops_per_hbm = minimum_heads
        return (layer.n, num_ops_per_hbm, layer.k, layer.dbyte, pim_type.name,
                power_constraint, self.rope, self.num_agent, self.diff_rate,
                self.token_block, self.v_master_diff)

    def _has_cached_result(self, key):
        if self.df.empty:
            return False
        l, nhead, dhead, dbyte, pim_name, power_constraint, rope, num_agent, diff_rate, token_block, v_master_diff = key
        row = self.df[(self.df['L'] == l) & (self.df['nhead'] == nhead) &
                      (self.df['dbyte'] == dbyte) &
                      (self.df['dhead'] == dhead) &
                      (self.df['power_constraint'] == power_constraint) &
                      (self.df['pim_type'] == pim_name) &
                      (self.df['rope'] == rope) &
                      (self.df['num_agent'] == num_agent) &
                      (self.df['diff_rate'] == diff_rate) &
                      (self.df['token_block'] == token_block) &
                      (self.df['v_master_diff'] == v_master_diff)]
        return not row.empty

    def precompute(self, pim_type, layers, power_constraint=True):
        if self.sim_cores <= 1:
            return

        unique_layers = {}
        for layer in layers:
            key = self._cache_key_for_layer(pim_type, layer, power_constraint)
            if self.force_run or not self._has_cached_result(key):
                unique_layers[key] = layer

        if not unique_layers:
            self.force_run = False
            return

        num_workers = min(self.sim_cores, len(unique_layers))
        print("Precomputing {} exact Ramulator cases with {} CPU workers".format(
            len(unique_layers), num_workers))

        def run_one(item):
            idx, layer = item
            log_path = os.path.join(self.ramulator_dir,
                                    ".ramulator_precompute_{}_{}.csv".format(
                                        os.getpid(), idx))
            worker = self._make_worker(log_path)
            try:
                worker.run(pim_type, layer, power_constraint)
                df = pd.read_csv(log_path)
                return df.iloc[-1].tolist()
            finally:
                if os.path.exists(log_path):
                    os.remove(log_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(run_one, item)
                for item in enumerate(unique_layers.values())
            ]
            for future in concurrent.futures.as_completed(futures):
                self.update_log_file(future.result())

        self.force_run = False

    def run_ramulator(self, pim_type: PIMType, l, num_ops_per_hbm, dbyte,
                      yaml_file, file_name, power_constraint):
        pim_type_name = pim_type.name.lower(
        ) if not pim_type == PIMType.BA else "bank"
        trace_file = os.path.join(self.ramulator_dir, file_name + '.trace')

        trace_exc = os.path.join(
            self.ramulator_dir,
            "trace_gen/gen_trace_attacc_{}.py".format(pim_type_name))
        trace_cmd = [
            "python3", trace_exc, "--dhead", str(self.dhead), "--nhead",
            str(num_ops_per_hbm), "--seqlen", str(l), "--dbyte", str(dbyte),
            "--output", trace_file
        ]
        if self.rope:
            trace_cmd += [
                "--rope", "--num-agent", str(self.num_agent),
                "--diff-rate", str(self.diff_rate),
                "--token-block", str(self.token_block)
            ]

        try:
            subprocess.run(trace_cmd, check=True)
            result = self._run_trace_parallel(trace_file, file_name, yaml_file,
                                              power_constraint)
        finally:
            if os.path.exists(trace_file):
                os.remove(trace_file)

        return result

    def run(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        if os.path.exists(self.ramulator_dir):
            l = layer.n
            dhead = self.dhead
            dbyte = layer.dbyte
            num_ops_per_attacc = self._effective_num_ops_per_attacc(layer)
            num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
            num_ops_group = 1
            if self.fast_mode:
                minimum_heads = 64
                num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
                num_ops_per_hbm = minimum_heads

            file_name = "attacc_l{}_nattn{}_dhead{}_dbyte{}_pc{}".format(
                l, num_ops_per_hbm, dhead, layer.dbyte, int(power_constraint))
            if self.rope:
                file_name += "_rope_agent{}_pb{}_blk{}_kvmd{}".format(
                    self.num_agent, str(self.diff_rate).replace(".", "p"),
                    self.token_block, self.v_master_diff)
            yaml_file = os.path.join(self.ramulator_dir, file_name + '.yaml')
            self.make_yaml_file(yaml_file, file_name, power_constraint)

            result = self.run_ramulator(pim_type, l, num_ops_per_hbm,
                                        layer.dbyte, yaml_file, file_name,
                                        power_constraint)

            # remove trace
            rm_yaml_cmd = f"rm {yaml_file}"
            try:
                os.system(rm_yaml_cmd)
            except Exception as e:
                print(f"Error: {e}")

            # post processing
            # 32: read granularity
            cycle, mac, sfm, mvgb, mvsb, wrgb = result
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
                mem_acc *= 1

            ## update log file

            log = [
                l, num_ops_per_hbm, dhead, dbyte, pim_type.name,
                power_constraint, self.rope, self.num_agent, self.diff_rate,
                self.token_block, self.v_master_diff
            ] + result
            self.update_log_file(log)

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            return exec_time, traffic

        else:
            assert 0, "Need to install ramulator"

    def output(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        if self.df.empty or self.force_run:
            return self.run(pim_type, layer, power_constraint)

        num_ops_per_attacc = self._effective_num_ops_per_attacc(layer)
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
                      (self.df['power_constraint'] == power_constraint) & \
                      (self.df['pim_type'] == pim_type.name) & \
                      (self.df['rope'] == self.rope) & \
                      (self.df['num_agent'] == self.num_agent) & \
                      (self.df['diff_rate'] == self.diff_rate) & \
                      (self.df['token_block'] == self.token_block) & \
                      (self.df['v_master_diff'] == self.v_master_diff)]
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
