#!/usr/bin/env bash
#SBATCH --job-name=m2_before
#SBATCH --partition=athena-mini
#SBATCH --nodelist=node1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=30 --mem=300G
#SBATCH --time=12:00:00 --open-mode=append
#SBATCH --output=/home/cw636/chenyi/attacc_drampim/output/_verify/before2.out
#SBATCH --error=/home/cw636/chenyi/attacc_drampim/output/_verify/before2.out
set -uo pipefail
# HEAD tree: regenerate ONLY the full-event hashes (the 27 events=none baseline
# reports from the first matrix are still on disk and are reused).
bash /home/cw636/chenyi/attacc_drampim/output/_verify/run_matrix2.sh before2 \
     /home/cw636/chenyi/attacc_drampim/output/_verify/pristine evfull
