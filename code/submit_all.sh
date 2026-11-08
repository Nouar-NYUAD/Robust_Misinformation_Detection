#!/bin/bash
# Submit the full pipeline on Jubail. Run from /scratch/naa9497/misinfo/code (login shell).
#   ATT : test attacks (Qwen2.5-32B) + adaptive attacks + preservation check (Gemma-3-12B)
#   RET : retrieval indexes + train-split augmentations (Llama-8B) + AIR training + AIR index
#   CAN : cached canonicalization of every input (ours + generic claim normalization)       [after ATT]
#   CLF : classifiers, 3 seeds; XLM-R also with adversarial training; +canon predictions   [after ATT RET CAN]
#   DET : LLM / RAG baselines (raw and +canon) and ours (UNWRAP, UNWRAP+AIR)                 [after ATT RET CAN]
#   ABL : ablations                                                                       [after ATT RET CAN]
set -e
DS="pubhealth climatefever finfact"
q() { for d in $DS; do printf '"%s" ' "${1//@/$d}"; done; }
sub() { eval sbatch --parsable "$@"; }

ATT=$(sub -J attacks job.sbatch \
  $(q "python gen_attacks.py --dataset @ --model Qwen/Qwen2.5-32B-Instruct") \
  $(q "python adaptive_attacks.py --dataset @") \
  $(q "python validate_attacks.py --dataset @"))
RET=${RET:-$(sub -J retr job.sbatch \
  "\"python retrieval.py --dataset $DS --encoder me5 bgem3\"" \
  "\"python gen_train_aug.py\"" \
  "\"python train_air.py --out /scratch/naa9497/misinfo/models/bgem3-air\"" \
  "\"python retrieval.py --dataset $DS --encoder air\"")}
CAN=$(sub -J canon --dependency=afterok:$ATT job.sbatch \
  $(q "python canonicalize.py --dataset @ --kind ours") \
  $(q "python canonicalize.py --dataset @ --kind norm"))
DEP=afterok:$ATT:$RET:$CAN
CLF=$(sub -J clf --dependency=$DEP job.sbatch \
  $(for s in 13 14 15; do q "python clf.py --dataset @ --model FacebookAI/xlm-roberta-large --tag xlmr --seed $s"; done) \
  $(for s in 13 14 15; do q "python clf.py --dataset @ --model microsoft/deberta-v3-large --tag deberta --seed $s"; done) \
  $(for s in 13 14 15; do q "python clf.py --dataset @ --model FacebookAI/xlm-roberta-large --tag xlmr-advtrain --aug --seed $s"; done))
DET=$(sub -J detect --dependency=$DEP job.sbatch \
  $(q "python detect.py --dataset @ --method ours") \
  $(q "python detect.py --dataset @ --method ours --encoder air --tag air") \
  $(q "python detect.py --dataset @ --method zs") \
  $(q "python detect.py --dataset @ --method zs --llm Qwen/Qwen2.5-7B-Instruct --tag qwen") \
  $(q "python detect.py --dataset @ --method rag") \
  $(q "python detect.py --dataset @ --method minicheck"))
DET2=$(sub -J detect_canon --dependency=$DEP job.sbatch \
  $(q "python detect.py --dataset @ --method zs --canon_input ours --tag canon") \
  $(q "python detect.py --dataset @ --method zs --llm Qwen/Qwen2.5-7B-Instruct --canon_input ours --tag qwen+canon") \
  $(q "python detect.py --dataset @ --method rag --canon_input ours --tag canon") \
  $(q "python detect.py --dataset @ --method minicheck --canon_input ours --tag canon"))
ABL=$(sub -J ablate --dependency=$DEP job.sbatch \
  $(q "python detect.py --dataset @ --method ours --no_canon --tag nocanon") \
  $(q "python detect.py --dataset @ --method ours --canon_kind norm --tag normcanon") \
  $(q "python detect.py --dataset @ --method ours --no_sparse --tag nosparse") \
  $(q "python detect.py --dataset @ --method ours --no_rerank --tag norerank") \
  $(q "python detect.py --dataset @ --method ours --encoder me5 --no_sparse --tag me5"))
echo "attacks=$ATT retr=$RET canon=$CAN clf=$CLF detect=$DET detect_canon=$DET2 ablate=$ABL"
