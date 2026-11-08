#!/bin/bash
# Literature baselines (submitted after the baseline smoke test passed). Run from /scratch/naa9497/misinfo/code.
#   CANON : id of the running canonicalization job (classifiers also write +canon predictions)
#   ARG   : rationales (Llama-3.1-8B) then official ARG trainer, 3 seeds x 3 datasets
#   KPF   : Kotonya & Toni SciBERT+evidence (3 seeds) and ProgramFC, 3 datasets
#   CLF   : XLM-R clean and XLM-R adversarial training, 3 seeds (replaces the old clf job, which also had DeBERTa)
set -e
CANON=${CANON:?set CANON=<job id>}
DS="pubhealth climatefever finfact"
q() { for d in $DS; do printf '"%s" ' "${1//@/$d}"; done; }
sub() { eval sbatch --parsable "$@"; }
ARG=$(sub -J arg job.sbatch \
  $(q "python arg_run.py --dataset @ --stage rationales") \
  $(for s in 3759 3760 3761; do q "python arg_run.py --dataset @ --stage train --seed $s"; done))
KPF=$(sub -J kot_pfc job.sbatch \
  $(for s in 13 14 15; do q "python kotonya_run.py --dataset @ --seed $s"; done) \
  $(q "python programfc_run.py --dataset @"))
CLF=$(sub -J clf2 --dependency=afterok:$CANON job.sbatch \
  $(for s in 13 14 15; do q "python clf.py --dataset @ --model FacebookAI/xlm-roberta-large --tag xlmr --seed $s"; done) \
  $(for s in 13 14 15; do q "python clf.py --dataset @ --model FacebookAI/xlm-roberta-large --tag xlmr-advtrain --aug --seed $s"; done))
echo "arg=$ARG kot_pfc=$KPF clf2=$CLF"
