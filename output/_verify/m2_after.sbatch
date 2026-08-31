#!/usr/bin/env bash
#SBATCH --job-name=m2_after
#SBATCH --partition=athena
#SBATCH --nodelist=node3
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=52 --mem=400G
#SBATCH --time=12:00:00 --open-mode=append
#SBATCH --output=/home/cw636/chenyi/attacc_drampim/output/_verify/after2.out
#SBATCH --error=/home/cw636/chenyi/attacc_drampim/output/_verify/after2.out
set -uo pipefail
bash /home/cw636/chenyi/attacc_drampim/output/_verify/run_matrix2.sh after2 /home/cw636/chenyi/attacc_drampim
