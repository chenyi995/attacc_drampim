#!/usr/bin/env bash
#SBATCH --job-name=ver_after
#SBATCH --partition=athena
#SBATCH --nodelist=node1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=60 --mem=400G
#SBATCH --time=08:00:00 --open-mode=append
#SBATCH --output=/home/cw636/chenyi/attacc_drampim/output/_verify/after.out
#SBATCH --error=/home/cw636/chenyi/attacc_drampim/output/_verify/after.out
set -uo pipefail
bash /home/cw636/chenyi/attacc_drampim/output/_verify/run_matrix.sh after
