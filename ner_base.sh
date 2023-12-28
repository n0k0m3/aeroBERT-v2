#!/bin/bash

# Finetune base model on NER task
# get model path from first argument
# get train and test data from second and third argument
MODEL_NAME=$1
TRAIN_DATA=$2
TEST_DATA=$3
BATCH_SIZE=$4

python run_ner.py \
    --output_dir=output/ner/$MODEL_NAME \
    --model_name_or_path=$MODEL_NAME \
    --train_file=$TRAIN_DATA \
    --validation_file=$TEST_DATA \
    --test_file=$TEST_DATA \
    --do_train \
    --do_eval \
    --do_predict \
    --text_column_name="Word" \
    --label_column_name="Tag" \
    --label_all_tokens \
    --return_entity_level_metrics \
    --per_device_train_batch_size=$BATCH_SIZE \
    --per_device_eval_batch_size=$BATCH_SIZE \
    --num_train_epochs=20 \
    --logging_strategy=epoch \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --load_best_model_at_end \
    --metric_for_best_model=eval_overall_f1 \
    --greater_is_better=True \
    --save_total_limit=2