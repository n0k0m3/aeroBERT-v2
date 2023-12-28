#!/bin/bash

# Train aeroBERTv2 model
# get model name from first argument
# get train and test data from second and third argument
MODEL_NAME=$1
TRAIN_DATA=$2
TEST_DATA=$3
BATCH_SIZE=$4

python run_mlm.py \
    --output_dir=output/aeroBERTv2-$MODEL_NAME \
    --model_name_or_path=$MODEL_NAME \
    --do_train \
    --train_file=$TRAIN_DATA \
    --do_eval \
    --validation_file=$TEST_DATA \
    --per_device_train_batch_size=$BATCH_SIZE \
    --per_device_eval_batch_size=$BATCH_SIZE \
    --line_by_line \
    --gradient_accumulation_steps=128 \
    --learning_rate=1e-4 \
    --max_steps=12500 \
    --save_total_limit=2 \
    --load_best_model_at_end \
    --metric_for_best_model=eval_loss \
    --evaluation_strategy=steps \
    --logging_steps=500 \
    --save_steps=500