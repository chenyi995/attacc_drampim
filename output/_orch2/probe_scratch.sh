#!/usr/bin/env bash
# Probe the /localdata switch WITHOUT touching production common.sh.
# Reproduces make_ramdir exactly and checks every artefact it must produce.
set -uo pipefail
REPO=/home/cw636/chenyi/attacc_drampim
R=$(cat "$REPO/output/_orch2/CURRENT_ROOT")
POOL=$R/cachepool
H=$(hostname -s)
fail=0

scratch_root() {
  local base=/localdata/kvpim_${USER}
  if mkdir -p "$base" 2>/dev/null && [ -w "$base" ]; then echo "$base"; else echo /tmp; fi
}

root=$(scratch_root)
echo "[$H] scratch_root -> [$root]"
[ -d "$root" ] || { echo "[$H] FAIL: root 不存在"; exit 1; }

tag="probe_$$"; model=GPT-13B
rd=$root/kvpim_${USER}_${tag}_$$
mkdir -p "$rd" || { echo "[$H] FAIL: mkdir $rd"; exit 1; }
ln -sf "$REPO/ramulator2/ramulator2" "$rd/ramulator2"
ln -sf "$REPO/ramulator2/trace_gen"  "$rd/trace_gen"
cp "$REPO/ramulator.out" "$rd/ramulator.out" 2>/dev/null || :
cat "$POOL/${model}__"*.jsonl > "$rd/signature_cache.jsonl" 2>/dev/null || :

chk() { if [ "$2" = yes ]; then echo "[$H]   ok  $1"; else echo "[$H]   FAIL $1"; fail=1; fi; }
chk "返回值非空且是目录"      "$([ -n "$rd" ] && [ -d "$rd" ] && echo yes || echo no)"
chk "ramulator2 符号链接存在"  "$([ -L "$rd/ramulator2" ] && echo yes || echo no)"
chk "ramulator2 指向可执行文件" "$([ -x "$rd/ramulator2" ] && echo yes || echo no)"
chk "trace_gen 可达"           "$([ -e "$rd/trace_gen" ] && echo yes || echo no)"
chk "签名缓存非空"             "$([ -s "$rd/signature_cache.jsonl" ] && echo yes || echo no)"
echo "[$H]   缓存行数 $(wc -l < "$rd/signature_cache.jsonl" 2>/dev/null || echo 0)"
# 真正跑一次 ramulator,证明它能在这个目录里工作
if "$rd/ramulator2" --help >/dev/null 2>&1 || [ $? -le 2 ]; then chk "ramulator2 可执行" yes; else chk "ramulator2 可执行" no; fi
echo "[$H]   可用空间 $(df -BG --output=avail "$root" 2>/dev/null | tail -1 | tr -dc '0-9')G"
rm -rf "$rd"
[ "$fail" = 0 ] && echo "[$H] PROBE PASS" || echo "[$H] PROBE FAIL"
