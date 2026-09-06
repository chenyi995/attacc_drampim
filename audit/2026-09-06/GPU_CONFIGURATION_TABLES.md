| GPU | dense FP16 TFLOPS | GPU HBM TB/s | GPU↔GPU GB/s | GPU↔PIM 默认 GB/s | 默认有效显存 GiB | 型号配置显存 GiB |
|---|---:|---:|---:|---:|---:|---:|
| A100a | 312 | 3.352 | 600 | 600 | 80 | 80 |
| H100 | 989.4 | 3.352 | 900 | 600 | 80 | 80 |
| H200 | 989.4 | 4.8 | 900 | 600 | 80 | 141 |
| B200 | 2250 | 8 | 1800 | 600 | 80 | 180 |

| A档（所有GPU逐一检查） | prefill | decode | bank 命令 | PE GHz |
|---|---|---|---|---:|
| A1 | gpu | pim | replicate | 0.666 |
| A2 | gpu | gpu | replicate | 0.666 |
| A3b | gpu | pim | replicate | 0.666 |
| A4c | gpu | pim | replicate | 0.666 |
| A4e | gpu | pim | replicate | 0.666 |
| A5 | pim | pim | mq | 1.3004 |
| A6 | dynamic | pim | mq | 1.3004 |
