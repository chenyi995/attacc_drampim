> 历史记录：保留当时的技术证据和评价，当前待审事项统一看 [CURRENT_ISSUES.md](../CURRENT_ISSUES.md)。

# 原始 AttAcc 来源摘录

由脚本直接读取 git 对象 `c600051`；行号属于原始文件，不属于当前 HEAD。只作来源对照，未执行这些历史代码。

## src/system.py

原文件 L110–L167：

```text
110:         def _pipeline(layers, level=False):
111:             qkv_time, prj_time, score_time, context_time, x2g_time, softmax_time = 0, 0, 0, 0, 0, 0
112:             for layer in layers:
113:                 if layer.name in ["qkv"]:
114:                     qkv_time += layer.exec_time
115:                 elif layer.name in ["proj"]:
116:                     prj_time += layer.exec_time
117:                 elif layer.name in ["comm_x2g"]:
118:                     x2g_time += layer.exec_time
119:                 elif layer.name in ["score"]:
120:                     score_time += layer.exec_time
121:                 elif layer.name in ["context"]:
122:                     context_time += layer.exec_time
123:                 elif layer.name in ["softmax"]:
124:                     softmax_time += layer.exec_time
125: 
126:             minimum_ratio = 1 / (self.model.num_heads / self.GPU.num_xpu)
127:             if level == False:
128:                 #softmax_time = 0
129:                 attn_time = score_time + context_time + softmax_time
130:                 if attn_time > x2g_time:
131:                     x2g_time *= minimum_ratio
132:                 else:
133:                     x2g_time -= attn_time * (1 - minimum_ratio)
134: 
135:             else:
136:                 #softmax_time = 0
137:                 fc_time = qkv_time + prj_time
138:                 attn_time = score_time + context_time + softmax_time
139:                 if attn_time > fc_time:
140:                     qkv_time *= minimum_ratio
141:                     prj_time *= minimum_ratio
142: 
143:                     if attn_time > x2g_time:
144:                         x2g_time *= minimum_ratio
145:                     else:
146:                         x2g_time -= attn_time * (1 - minimum_ratio)
147:                 else:
148:                     if fc_time > x2g_time:
149:                         x2g_time *= minimum_ratio
150:                         qkv_time -= attn_time * (1 - minimum_ratio) * (3 / 4)
151:                         prj_time -= attn_time * (1 - minimum_ratio) * (1 / 4)
152:                     else:
153:                         x2g_time -= attn_time * (1 - minimum_ratio)
154:                         qkv_time *= minimum_ratio
155:                         prj_time *= minimum_ratio
156:             softmax_time = 0
157: 
158:             for layer in layers:
159:                 if layer.name in ["qkv"]:
160:                     layer.exec_time = qkv_time
161:                 elif layer.name in ["proj"]:
162:                     layer.exec_time = prj_time
163:                 elif layer.name in ["comm_x2g"]:
164:                     # for 2 comm_x2g layers
165:                     layer.exec_time = x2g_time / 2
166:                 elif layer.name in ["softmax"]:
167:                     layer.exec_time = softmax_time
```

原文件 L220–L234：

```text
220:             s_decoder = self.model.sum_decoder
221:             g_decoder = self.model.gen_decoder
222: 
223:             ## Summarization stage
224:             for layer in s_decoder:
225:                 # Get execution time and energy
226:                 exec_time, energy = self.devices['GPU'].get_time_and_energy(
227:                     layer)
228: 
229:                 # Time to transfer KV matrices to memory (PCIe bandwidth)
230:                 if layer.type == LayerType.X2G:
231:                     exec_time += max(wrt_io_busy - time, 0)
232:                     wrt_io_busy = time + exec_time
233:                 layer.exec_time = exec_time
234:                 layer.energy = energy
```

原文件 L275–L286：

```text
275:                     unit_energy['g_l1'] += layer.energy[2]
276:                     unit_energy['g_reg'] += layer.energy[3]
277:                     unit_energy['g_alu'] += layer.energy[4]
278:                     unit_energy['g_comm'] += layer.energy[5]
279: 
280:                 # pipeline
281:                 if self.hetero_name == DeviceType.PIM:
282:                     _pipeline(decoder_block, pipe)
283:                     if parallel_ff:
284:                         _ff_parallel(decoder_block)
285: 
286:             s_perf = {
```

## src/devices.py

原文件 L225–L260：

```text
225:     def _io_time_energy(self, layer: Layer):
226:         m, n, k, numOp, dbyte = layer.get_infos()
227: 
228:         def get_nvlink_time(size):
229:             # interpolation of real data on A100
230:             # size unit: Byte
231:             if size == 0:
232:                 return 1
233:             else:
234:                 approx_ns_time = 6060 + 0.009 * size * (
235:                     (600 * 1000 * 1000 * 1000 / self.max_interface_bandwidth))
236:                 approx_time = approx_ns_time / 1000 / 1000 / 1000
237:                 return max(approx_time,
238:                            size / (self.max_interface_bandwidth / 2))
239: 
240:         if self.name == DeviceType.CPU:
241:             ## RX, TX --> 1/2x
242:             bw = self.max_interface_bandwidth / 2
243:             traffic = m * n * numOp * dbyte
244:             exec_time = traffic / bw
245:             # we ignore CPU energy
246:             energy = 0
247:         else:
248:             ## each GPU has partial sum of output.
249:             traffic = m * n * numOp * dbyte
250:             interface_bw = self.max_interface_bandwidth / 2
251:             if layer.type == LayerType.X2G:
252:                 exec_time = traffic / interface_bw
253:             else:
254:                 ## allreduce
255:                 exec_time = get_nvlink_time(
256:                     traffic / self.num_xpu) * (self.num_xpu - 1)
257: 
258:             # all reduce communication
259:             energy = self.num_xpu * traffic * self.energy_table['comm']
260:         return exec_time, [0, 0, 0, 0, 0, energy]
```

原文件 L326–L352：

```text
326:     def get_time_and_energy(self, layer: Layer):
327:         if layer.type == LayerType.X2G:
328:             return self._io_time_energy(layer)
329: 
330:         elif layer.type == LayerType.MATMUL:
331:             ## operational granularity = the attention layer
332:             if 'score' in layer.name:
333:                 m, n, k, numOp, dbyte = layer.get_infos()
334:                 time, traffic = self.ramulator.output(
335:                     self.pim_type, layer, self.power_constraint)
336:                 io_energy = 0
337:                 for i in range(len(self.io_energy_table)):
338:                     io_energy += traffic[i] * self.io_energy_table[i]
339: 
340:                 energy_per_access = self.energy_table['mem']
341:                 cell_energy = traffic[-1] * energy_per_access
342:                 dram_energy = cell_energy + io_energy
343:                 cal_energy = layer.get_flops() / 2 * self.energy_table['alu']
344: 
345:                 energies = [dram_energy, 0, 0, 0, cal_energy, 0]
346:                 energies = [i * self.num_attacc for i in energies]
347: 
348:                 return time, energies
349:             else:
350:                 return 0, [0, 0, 0, 0, 0, 0]
351: 
352:         elif layer.type == LayerType.SOFTMAX:
```

## src/ramulator_wrapper.py

原文件 L157–L173：

```text
157:     def run(self, pim_type: PIMType, layer: Layer, power_constraint=True):
158:         if os.path.exists(self.ramulator_dir):
159:             l = layer.n
160:             dhead = self.dhead
161:             dbyte = layer.dbyte
162:             num_ops_per_attacc = layer.numOp
163:             num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
164:             num_ops_group = 1
165:             if self.fast_mode:
166:                 minimum_heads = 64
167:                 num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
168:                 num_ops_per_hbm = minimum_heads
169: 
170:             file_name = "attacc_l{}_nattn{}_dhead{}_dbyte{}_pc{}".format(
171:                 l, num_ops_per_hbm, dhead, layer.dbyte, int(power_constraint))
172:             yaml_file = os.path.join(self.ramulator_dir, file_name + '.yaml')
173:             self.make_yaml_file(yaml_file, file_name, power_constraint)
```

原文件 L202–L218：

```text
202:             ## update log file
203: 
204:             log = [
205:                 l, num_ops_per_hbm, dhead, dbyte, pim_type.name,
206:                 power_constraint
207:             ] + result
208:             self.update_log_file(log)
209: 
210:             ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
211:             traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
212:             traffic = [i * self.num_hbm for i in traffic]
213:             traffic = [i * num_ops_group for i in traffic]
214:             exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
215:             return exec_time, traffic
216: 
217:         else:
218:             assert 0, "Need to install ramulator"
```

原文件 L220–L241：

```text
220:     def output(self, pim_type: PIMType, layer: Layer, power_constraint=True):
221:         if self.df.empty:
222:             self.run(pim_type, layer, power_constraint)
223: 
224:         num_ops_per_attacc = layer.numOp
225:         num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
226:         num_ops_group = 1
227:         if self.fast_mode:
228:             minimum_heads = 64
229:             num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
230:             num_ops_per_hbm = minimum_heads
231: 
232:         l = layer.n
233:         dhead = layer.k
234:         dbyte = layer.dbyte
235:         row = self.df[(self.df['L'] == l) & (self.df['nhead'] == num_ops_per_hbm) & \
236:                       (self.df['dbyte'] == dbyte) & (self.df['dhead'] == dhead) & \
237:                       (self.df['power_constraint'] == power_constraint) &  \
238:                       (self.df['pim_type'] == pim_type.name)]
239:         if row.empty:
240:             return self.run(pim_type, layer, power_constraint)
241: 
```

原文件 L263–L269：

```text
263:             ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
264:             traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
265:             traffic = [i * self.num_hbm for i in traffic]
266:             traffic = [i * num_ops_group for i in traffic]
267:             exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
268:             exec_time *= num_ops_group
269:             return exec_time, traffic
```

## src/model.py

原文件 L85–L127：

```text
85: 
86:     def __init__(self, modelinfos, tensor_parallel=8):
87:         self.sum_decoder = []
88:         self.gen_decoder = []
89:         self.name = modelinfos['name']
90:         self.ndec = modelinfos['ndec']
91:         self.num_heads = modelinfos['num_heads']
92:         self.hdim = modelinfos['hdim']
93:         self.ff_scale = modelinfos['ff_scale']
94:         self.dtype = modelinfos['dtype']
95:         self.dhead = int(self.hdim / self.num_heads)
96:         self.tp = tensor_parallel
97: 
98:     def build(self, batch, lin, lout, attn_on_hetero=False):
99:         self.sum_decoder = []
100:         self.gen_decoder = []
101: 
102:         # Summarization
103:         self.sum_decoder.append(
104:             Layer('sum', 'qkv', LayerType.FC, True, self.dtype, batch * lin,
105:                   3 * int(self.hdim / self.tp), self.hdim, 1))
106:         if (attn_on_hetero):
107:             # send kv matrices
108:             self.sum_decoder.append(
109:                 Layer('sum', 'comm_x2g', LayerType.X2G, False, self.dtype,
110:                       batch * lin, 2 * int(self.hdim / self.tp), 1, 1))
111:         self.sum_decoder.append(
112:             Layer('sum', 'score', LayerType.MATMUL, False, self.dtype, lin,
113:                   lin, self.dhead,
114:                   int(self.num_heads / self.tp) * batch))
115:         self.sum_decoder.append(
116:             Layer('sum', 'softmax', LayerType.SOFTMAX, False, self.dtype, lin,
117:                   lin, 1,
118:                   int(self.num_heads / self.tp) * batch))
119:         self.sum_decoder.append(
120:             Layer('sum', 'context', LayerType.MATMUL, False, self.dtype, lin,
121:                   self.dhead, lin,
122:                   int(self.num_heads / self.tp) * batch))
123:         self.sum_decoder.append(
124:             Layer('sum', 'proj', LayerType.FC, True, self.dtype, batch * lin,
125:                   self.hdim, int(self.hdim / self.tp), 1))
126:         self.sum_decoder.append(
127:             Layer('sum', 'comm_g2g', LayerType.G2G, False, self.dtype, batch * lin,
```

## src/config.py

原文件 L1–L30：

```text
1: from src.type import *
2: 
3: SCALING_FACTOR = {}
4: SCALING_FACTOR['MAX_COMPUTE_UTIL'] = 0.8
5: SCALING_FACTOR['MAX_OFF_MEM_BW_UTIL'] = 0.85
6: 
7: # ENERGY_TABLE: pJ per byte
8: # Cache info: https://core.ac.uk/download/pdf/232142915.pdf
9: ENERGY_TABLE = {
10:     'GPU': {},
11:     'CPU': {},
12:     'PIM': {
13:         PIMType.BA: {},
14:         PIMType.BG: {},
15:         PIMType.BUFFER: {}
16:     }
17: }
18: ENERGY_TABLE['GPU']['reg'] = 0.0675
19: #4-way cache, ref: https://arxiv.org/pdf/1509.02308v1.pdf
20: ENERGY_TABLE['GPU'][ 'l1'] = 0.16 * 8  
21: ENERGY_TABLE['GPU']['l2'] = 0.3 * 8
22: ENERGY_TABLE['GPU']['alu'] = 0.32
23: ENERGY_TABLE['GPU']['mem'] = (0.11 + 0.44 + 1.01 + 1.23 + 0.5 + 0.3) * 8
24: # ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10067395
25: ENERGY_TABLE['GPU'][ 'comm'] = 1.3 * 8  
26: 
27: ## TODO: Add energy of CPU (pJ per byte)
28: ENERGY_TABLE['CPU']['reg'] = 0
29: ENERGY_TABLE['CPU']['l1'] = 0
30: ENERGY_TABLE['CPU']['l2'] = 0
```

原文件 L79–L97：

```text
79:     config['GPU']["GPUTYPE"] = gpu_type
80:     config['GPU']["NUM_DEVICE"] = 8 if num_gpu is None else num_gpu
81: 
82:     if gpu_type == GPUType.A100a:
83:         # Ref: DGX-A100 whitepaper
84:         config['GPU']["NUM_CORE"] = 108
85:         config['GPU']["FLOPS_PER_DEVICE"] = 312 * 1000 * 1000 * 1000 * 1000 \
86:                                             if flops is None else flops
87:         config['GPU']["MEM_CAPACITY_PER_DEVICE"] = 80 * 1024 * 1024 * 1024 \
88:                                                     if mem_cap is None else mem_cap
89: 
90:         config['GPU']["OFF_MEM_BW_PER_DEVICE"] = 3352 * 1000 * 1000 * 1000 \
91:                                                   if mem_bw is None else mem_bw
92:         config['GPU']["L2_MEM_BW_PER_DEVICE"] = float('inf')
93:         #config['GPU']["L2_MEM_BW_PER_DEVICE"] = 3.8 * 1000 * 1000 * 1000 * 1000
94:         config['GPU']["L1_CAP_PER_CORE"] = 192 * 1024
95:         config['GPU']["L2_CAP_PER_DEVICE"] = 40 * 1024 * 1024
96:         config['GPU']["INTERFACE_BW"] = 600 * 1000 * 1000 * 1000
97:         config['GPU']["ENERGY_TABLE"] = ENERGY_TABLE['GPU']
```

## pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py

原文件 L332–L354：

```text
332:       if not j == 0:
333:         total_cmd += cmd_score_mvsb[i][j-1]
334:       ## BARRIER
335:       if not j == length:
336:         total_cmd += barrier
337: 
338:     # SoftMax
339:     ## SFM (Head0)
340:     total_cmd += cmd_sfm[i]
341:     ## MVGB (Head0)
342:     total_cmd += cmd_context_mvgb[i]
343:     ## BARRIER
344:     total_cmd += barrier
345: 
346:     # Context
347:     length = math.ceil(dhead/n_bank/n_mac)
348:     for j in range(0, length+1):
349:       ## MAC
350:       if not j == length:
351:         total_cmd += cmd_context_mac[i][j]
352:       ## MVSB
353:       if not j == 0:
354:         total_cmd += cmd_context_mvsb[i][j-1]
```

## ALL-BANK 地址语义来源

- `pim_ramulator_src/hbm3_pim_linear_mappers.cpp`：当前文件与原始 git 对象逐字节相同：`True`。
- `pim_ramulator_src/patches/src_dram_lambdas_action.patch`：当前文件与原始 git 对象逐字节相同：`True`。
- `pim_ramulator_src/patches/src_dram_lambdas_preq.patch`：当前文件与原始 git 对象逐字节相同：`True`。
