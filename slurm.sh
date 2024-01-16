#!/bin/bash

# Manpage: https://slurm.schedmd.com/sbatch.html

##################################
######### Configuration ##########
##################################
# Configure the job name
#SBATCH --job-name LLM

##################################
####### Resources Request ########
##################################

# Use GPU partition (gpu1 and gpu2) or other partition (e.g.: short)
# Find more usable partitions with 'sinfo -a'
#SBATCH --partition=gpu2

# Configure the number of nodes (in partition above)
# Do NOT use --ntasks-per-node > 1 if you are not using MPI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Configure the number of GPUs
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0

# (optional) list all gpus (check GPU available)
# nvidia-smi
echo `date`
# source /etc/profile.d/modules.sh
# module load cuda/11.2
# load conda environment
. ./env.sh

start=`date +%s`

export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

##################################
########## Pretraining ###########
##################################
# CUDA_VISIBLE_DEVICES=0 ./pretrain.sh bert-base-uncased data/pretrain/train.txt data/pretrain/test.txt 32 > stdlog/log_bert-base-uncased.txt 2>&1 &
# P1=$!
# CUDA_VISIBLE_DEVICES=1 ./pretrain.sh bert-base-cased data/pretrain/train.txt data/pretrain/test.txt 32 > stdlog/log_bert-base-cased.txt 2>&1 &
# P2=$!
# CUDA_VISIBLE_DEVICES=2 ./pretrain.sh bert-large-uncased data/pretrain/train.txt data/pretrain/test.txt 16 > stdlog/log_bert-large-uncased.txt 2>&1 &
# P3=$!
# CUDA_VISIBLE_DEVICES=3 ./pretrain.sh bert-large-cased data/pretrain/train.txt data/pretrain/test.txt 16 > stdlog/log_bert-large-cased.txt 2>&1 &
# P4=$!
# wait $P1 $P2 $P3 $P4


# CUDA_VISIBLE_DEVICES=0 ./pretrain.sh albert-base-v2 data/pretrain/train.txt data/pretrain/test.txt 32 > stdlog/log_albert-base-v2.txt 2>&1 &
# P1=$!
# CUDA_VISIBLE_DEVICES=1 ./pretrain.sh albert-large-v2 data/pretrain/train.txt data/pretrain/test.txt 16 > stdlog/log_albert-large-v2.txt 2>&1 &
# P2=$!
# CUDA_VISIBLE_DEVICES=2 ./pretrain.sh albert-xlarge-v2 data/pretrain/train.txt data/pretrain/test.txt 8 > stdlog/log_albert-xlarge-v2.txt 2>&1 &
# P3=$!
# CUDA_VISIBLE_DEVICES=3 ./pretrain.sh albert-xxlarge-v2 data/pretrain/train.txt data/pretrain/test.txt 4 > stdlog/log_albert-xxlarge-v2.txt 2>&1 &
# P4=$!
# wait $P1 $P2 $P3 $P4

# CUDA_VISIBLE_DEVICES=0 ./pretrain.sh roberta-base data/pretrain/train.txt data/pretrain/test.txt 32 > stdlog/log_roberta-base.txt 2>&1 &
# P1=$!
# CUDA_VISIBLE_DEVICES=1 ./pretrain.sh roberta-large data/pretrain/train.txt data/pretrain/test.txt 16 > stdlog/log_roberta-large.txt 2>&1 &
# P2=$!
# CUDA_VISIBLE_DEVICES=2 ./pretrain.sh microsoft/deberta-v3-base data/pretrain/train.txt data/pretrain/test.txt 16 > stdlog/log_deberta-v3-base.txt 2>&1 &
# P3=$!
# CUDA_VISIBLE_DEVICES=3 ./pretrain.sh microsoft/deberta-v3-large data/pretrain/train.txt data/pretrain/test.txt 8 > stdlog/log_deberta-v3-large.txt 2>&1 &
# P4=$!
# wait $P1 $P2 $P3 $P4

# CUDA_VISIBLE_DEVICES=0 ./pretrain.sh albert-xlarge-v2 data/pretrain/train.txt data/pretrain/test.txt 8 > stdlog/log_albert-xlarge-v2.txt 2>&1 &
# P1=$!
# CUDA_VISIBLE_DEVICES=1 ./pretrain.sh roberta-large data/pretrain/train.txt data/pretrain/test.txt 8 > stdlog/log_roberta-large.txt 2>&1 &
# P2=$!
# CUDA_VISIBLE_DEVICES=2 ./pretrain.sh microsoft/deberta-v3-base data/pretrain/train.txt data/pretrain/test.txt 8 > stdlog/log_deberta-v3-base.txt 2>&1 &
# P3=$!
# CUDA_VISIBLE_DEVICES=3 ./pretrain.sh microsoft/deberta-v3-large data/pretrain/train.txt data/pretrain/test.txt 4 > stdlog/log_deberta-v3-large.txt 2>&1 &
# P4=$!
# wait $P1 $P2 $P3 $P4

# CUDA_VISIBLE_DEVICES=0 ./pretrain.sh bert-large-uncased data/pretrain/train.txt data/pretrain/test.txt 8 > stdlog/log_bert-large-uncased.txt 2>&1 &
# P1=$!
# wait $P1

# CUDA_VISIBLE_DEVICES=0 ./pretrain_base.sh output/aeroBERTv2-microsoft/deberta-v3-large data/pretrain/train.txt data/pretrain/test.txt 4

##################################
######### Fine-tuning NER ########
##################################

#model still running: albert-xlarge-v2, roberta-large, microsoft/deberta-v3-base, microsoft/deberta-v3-large

# CUDA_VISIBLE_DEVICES=0 ./ner.sh albert-base-v2 data/ner/train.json data/ner/test.json 32 > stdlog/log_ner_albert-base-v2.txt 2>&1 &
# P1=$!
# CUDA_VISIBLE_DEVICES=1 ./ner.sh albert-large-v2 data/ner/train.json data/ner/test.json 16 > stdlog/log_ner_albert-large-v2.txt 2>&1 &
# P2=$!
# CUDA_VISIBLE_DEVICES=2 ./ner.sh albert-xxlarge-v2 data/ner/train.json data/ner/test.json 4 > stdlog/log_ner_albert-xxlarge-v2.txt 2>&1 & 
# P3=$!
# CUDA_VISIBLE_DEVICES=3 ./ner.sh roberta-base data/ner/train.json data/ner/test.json 32 > stdlog/log_ner_roberta-base.txt 2>&1 & 
# P4=$!
# wait $P1 $P2 $P3 $P4

# CUDA_VISIBLE_DEVICES=0 ./ner.sh bert-base-uncased data/ner/train.json data/ner/test.json 32 > stdlog/log_ner_bert-base-uncased.txt 2>&1 &
# P1=$!
# CUDA_VISIBLE_DEVICES=1 ./ner.sh bert-base-cased data/ner/train.json data/ner/test.json 32 > stdlog/log_ner_bert-base-cased.txt 2>&1 &
# P2=$!
# CUDA_VISIBLE_DEVICES=2 ./ner.sh bert-large-cased data/ner/train.json data/ner/test.json 16 > stdlog/log_ner_bert-large-cased.txt 2>&1 &
# P3=$!
# CUDA_VISIBLE_DEVICES=3 ./ner.sh bert-large-uncased data/ner/train.json data/ner/test.json 8 > stdlog/log_ner_bert-large-uncased.txt 2>&1 &
# P4=$!
# wait $P1 $P2 $P3 $P4

#not running
# CUDA_VISIBLE_DEVICES=0 ./ner.sh albert-xlarge-v2 data/ner/train.json data/ner/test.json 8 > stdlog/log_ner_albert-xlarge-v2.txt 2>&1 &
# P1=$!
# CUDA_VISIBLE_DEVICES=1 ./ner.sh roberta-large data/ner/train.json data/ner/test.json 8 > stdlog/log_ner_roberta-large.txt 2>&1 &
# P2=$!
# CUDA_VISIBLE_DEVICES=0 ./ner.sh microsoft/deberta-v3-base data/ner/train.json data/ner/test.json 8 > stdlog/log_ner_deberta-v3-base.txt 2>&1 &
# P1=$!
# CUDA_VISIBLE_DEVICES=1 ./ner.sh microsoft/deberta-v3-large data/ner/train.json data/ner/test.json 4 > stdlog/log_ner_deberta-v3-large.txt 2>&1 &
# P2=$!
#end not running

##################################
##### Fine-tuning Classifier #####
##################################

# CUDA_VISIBLE_DEVICES=0 ./classifier.sh albert-base-v2 data/classifier/train.json data/classifier/test.json 32 > stdlog/log_classifier_albert-base-v2.txt 2>&1 &
# P1=$!
# CUDA_VISIBLE_DEVICES=1 ./classifier.sh albert-large-v2 data/classifier/train.json data/classifier/test.json 16 > stdlog/log_classifier_albert-large-v2.txt 2>&1 &
# P2=$!
# CUDA_VISIBLE_DEVICES=2 ./classifier.sh albert-xxlarge-v2 data/classifier/train.json data/classifier/test.json 4 > stdlog/log_classifier_albert-xxlarge-v2.txt 2>&1 &
# P3=$!
# CUDA_VISIBLE_DEVICES=3 ./classifier.sh roberta-base data/classifier/train.json data/classifier/test.json 32 > stdlog/log_classifier_roberta-base.txt 2>&1 &
# P4=$!
# wait $P1 $P2 $P3 $P4

# CUDA_VISIBLE_DEVICES=0 ./classifier.sh bert-base-uncased data/classifier/train.json data/classifier/test.json 32 > stdlog/log_classifier_bert-base-uncased.txt 2>&1 &
# P1=$!
# CUDA_VISIBLE_DEVICES=1 ./classifier.sh bert-base-cased data/classifier/train.json data/classifier/test.json 32 > stdlog/log_classifier_bert-base-cased.txt 2>&1 &
# P2=$!
# CUDA_VISIBLE_DEVICES=2 ./classifier.sh bert-large-cased data/classifier/train.json data/classifier/test.json 16 > stdlog/log_classifier_bert-large-cased.txt 2>&1 &
# P3=$!
# CUDA_VISIBLE_DEVICES=3 ./classifier.sh bert-large-uncased data/classifier/train.json data/classifier/test.json 8 > stdlog/log_classifier_bert-large-uncased.txt 2>&1 &
# P4=$!
# wait $P1 $P2 $P3 $P4

#not running
# CUDA_VISIBLE_DEVICES=2 ./classifier.sh albert-xlarge-v2 data/classifier/train.json data/classifier/test.json 8 > stdlog/log_classifier_albert-xlarge-v2.txt 2>&1 &
# P3=$!
# CUDA_VISIBLE_DEVICES=3 ./classifier.sh roberta-large data/classifier/train.json data/classifier/test.json 8 > stdlog/log_classifier_roberta-large.txt 2>&1 &
# P4=$!
# CUDA_VISIBLE_DEVICES=2 ./classifier.sh microsoft/deberta-v3-base data/classifier/train.json data/classifier/test.json 8 > stdlog/log_classifier_deberta-v3-base.txt 2>&1 &
# P3=$!
# CUDA_VISIBLE_DEVICES=3 ./classifier.sh microsoft/deberta-v3-large data/classifier/train.json data/classifier/test.json 4 > stdlog/log_classifier_deberta-v3-large.txt 2>&1 &
# P4=$!
# wait $P1 $P2 $P3 $P4
#end not running

##################################
###########  Baseline ############
##################################

# CUDA_VISIBLE_DEVICES=0 ./pretrain_base.sh bert-base-uncased data/pretrain/train.txt data/pretrain/test.txt 32 > stdlog/log_base_bert-base-uncased.txt 2>&1 &
# P1=$!
# CUDA_VISIBLE_DEVICES=1 ./pretrain_base.sh bert-base-cased data/pretrain/train.txt data/pretrain/test.txt 32 > stdlog/log_base_bert-base-cased.txt 2>&1 &
# P2=$!
# CUDA_VISIBLE_DEVICES=2 ./pretrain_base.sh bert-large-uncased data/pretrain/train.txt data/pretrain/test.txt 16 > stdlog/log_base_bert-large-uncased.txt 2>&1 &
# P3=$!
# CUDA_VISIBLE_DEVICES=3 ./pretrain_base.sh bert-large-cased data/pretrain/train.txt data/pretrain/test.txt 16 > stdlog/log_base_bert-large-cased.txt 2>&1 &
# P4=$!
# wait $P1 $P2 $P3 $P4

# CUDA_VISIBLE_DEVICES=0 ./classifier_base.sh bert-base-cased data/classifier/train.json data/classifier/test.json 32 > stdlog/log_classifier_base_bert-base-cased.txt 2>&1 &
# P1=$!
# CUDA_VISIBLE_DEVICES=1 ./classifier_base.sh bert-large-cased data/classifier/train.json data/classifier/test.json 16 > stdlog/log_classifier_base_bert-large-cased.txt 2>&1 &
# P2=$!
# CUDA_VISIBLE_DEVICES=2 ./classifier_base.sh bert-base-uncased data/classifier/train.json data/classifier/test.json 32 > stdlog/log_classifier_base_bert-base-uncased.txt 2>&1 &
# P3=$!
# CUDA_VISIBLE_DEVICES=3 ./classifier_base.sh bert-large-uncased data/classifier/train.json data/classifier/test.json 16 > stdlog/log_classifier_base_bert-large-uncased.txt 2>&1 &
# P4=$!
# wait $P1 $P2 $P3 $P4

# CUDA_VISIBLE_DEVICES=0 ./ner_base.sh bert-base-cased data/ner/train.json data/ner/test.json 32 > stdlog/log_ner_base_bert-base-cased.txt 2>&1 &
# P1=$!
# CUDA_VISIBLE_DEVICES=1 ./ner_base.sh bert-large-cased data/ner/train.json data/ner/test.json 16 > stdlog/log_ner_base_bert-large-cased.txt 2>&1 &
# P2=$!
# CUDA_VISIBLE_DEVICES=2 ./ner_base.sh bert-base-uncased data/ner/train.json data/ner/test.json 32 > stdlog/log_ner_base_bert-base-uncased.txt 2>&1 &
# P3=$!
# CUDA_VISIBLE_DEVICES=3 ./ner_base.sh bert-large-uncased data/ner/train.json data/ner/test.json 16 > stdlog/log_ner_base_bert-large-uncased.txt 2>&1 &
# P4=$!
# wait $P1 $P2 $P3 $P4

end=`date +%s`

runtime=$((end-start))
echo "Runtime: $runtime"
