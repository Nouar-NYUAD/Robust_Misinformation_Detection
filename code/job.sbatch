#!/bin/bash -l
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --constraint="80g|h100|h200"
#SBATCH -c 16
#SBATCH --mem=150G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/naa9497/misinfo/logs/%x_%j.out
# usage: sbatch -J name job.sbatch "python gen_attacks.py --dataset pubhealth" ["python ..."]
module load gcc/12.2.0
export CC=$(which gcc) CXX=$(which g++)
export HF_HOME=/scratch/naa9497/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH=/scratch/naa9497/conda-envs/misinfo/bin:$PATH
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
# Run from a private snapshot of the code taken at job start, so later edits to code/ can never reach a running job.
# The snapshot sits next to code/ (misinfo/.code_<jobid>), so ROOT (= parent of the code folder) is unchanged.
SRC=${CODE_SRC:-/scratch/naa9497/misinfo/code}
SNAP=$(dirname "$SRC")/.code_${SLURM_JOB_ID}
cp -r "$SRC" "$SNAP"
cd "$SNAP"
echo "code snapshot: $SNAP ($(cat *.py | md5sum | cut -c1-12))"
nvidia-smi -L
rc=0
for cmd in "$@"; do echo ">>> $cmd"; eval "$cmd" || { echo "FAILED: $cmd"; rc=1; }; done
exit $rc
