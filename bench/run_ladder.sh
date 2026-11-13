#!/bin/bash
# The post's ladder (Section 6): same serving mix (512-token prompts, 1,024 requests, 256 slots), one change added per step.
# Runs each step on its own GPU in parallel (needs 5 GPUs; set GPUS to reuse fewer, e.g. GPUS="0 0 0 0 0" runs them one by one).
cd "$(dirname "$0")"; mkdir -p out
export B=1024 MNS=256 OL=1024 OLVAR=1 GMU=0.88 PL=512 GDN_REPLAY_L=8
read -r -a G <<< "${GPUS:-0 1 2 3 4}"
s() { g=$1; t=$2; shift 2; env CUDA_VISIBLE_DEVICES=$g "$@" python ladder.py out/ladder_$t.json > out/ladder_$t.log 2>&1; grep RESULT out/ladder_$t.log; }
if [ "$(printf '%s\n' "${G[@]}" | sort -u | wc -l)" -eq 1 ]; then par=""; else par="&"; fi
eval "s ${G[0]} s0_stock  GDN_REPLAY=0 PC=1 SSMDT=auto $par"
eval "s ${G[1]} s1_bf16   GDN_REPLAY=0 PC=1 SSMDT=bfloat16 $par"
eval "s ${G[2]} s2_nopc   GDN_REPLAY=0 PC=0 SSMDT=bfloat16 $par"
eval "s ${G[3]} s3_cg2048 GDN_REPLAY=0 PC=0 SSMDT=bfloat16 CGMAX=2048 $par"
eval "s ${G[4]} s4_replay GDN_REPLAY=1 PC=0 SSMDT=bfloat16 CGMAX=2048 $par"
wait
