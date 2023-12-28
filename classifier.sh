#!/bin/bash

# Finetune aeroBERTv2 model on text classification task
# get model path from first argument
# get train and test data from second and third argument
MODEL_NAME=$1
TRAIN_DATA=$2
TEST_DATA=$3
BATCH_SIZE=$4

python run_classification.py \
    --output_dir=output/classifier/aeroBERTv2-$MODEL_NAME \
    --model_name_or_path=output/aeroBERTv2-$MODEL_NAME \
    --train_file=$TRAIN_DATA \
    --validation_file=$TEST_DATA \
    --test_file=$TEST_DATA \
    --do_train \
    --do_eval \
    --do_predict \
    --text_column_names="requirements" \
    --label_column_name="label" \
    --per_device_train_batch_size=$BATCH_SIZE \
    --per_device_eval_batch_size=$BATCH_SIZE \
    --num_train_epochs=20 \
    --logging_strategy=epoch \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --load_best_model_at_end \
    --metric_name=f1 \
    --metric_for_best_model=eval_f1 \
    --greater_is_better=True \
    --save_total_limit=2