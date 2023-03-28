#!/bin/bash
SEED=42
NUM_EPOCHS=5
BATCH_SIZE=8
GRADIENT_ACC_STEPS=1
BATCH_SIZE_PER_GPU=$(( $BATCH_SIZE*$GRADIENT_ACC_STEPS ))
LEARN_RATE=0.00001
WARMUP=0.06
WEIGHT_DECAY=0.1
MAX_SEQ_LENGTH=512

MODEL='flax-community/alberti-bert-base-multilingual-cased'
OUTPUT_DIR='./outputs/alberti_base-output'
LOGGING_DIR='./logs/alberti_base.log'
DIR_NAME='sts'_${BATCH_SIZE_PER_GPU}_${WEIGHT_DECAY}_${LEARN_RATE}_$(date +'%m-%d-%y_%H-%M')

python ../bsc_run_glue.py --model_name_or_path $MODEL --seed $SEED \
                                         --dataset_script_path ./sts_dataset.py \
                                         --task_name stsb --do_train --do_eval --do_predict \
                                         --num_train_epochs $NUM_EPOCHS --gradient_accumulation_steps $GRADIENT_ACC_STEPS --per_device_train_batch_size $BATCH_SIZE \
                                         --learning_rate $LEARN_RATE --max_seq_length $MAX_SEQ_LENGTH \
                                         --warmup_ratio $WARMUP --weight_decay $WEIGHT_DECAY \
                                         --output_dir $OUTPUT_DIR/$DIR_NAME --overwrite_output_dir \
                                         --logging_dir $LOGGING_DIR/$DIR_NAME --logging_strategy epoch \
                                         --overwrite_cache \
                                         --metric_for_best_model combined_score --save_strategy epoch --evaluation_strategy epoch --load_best_model_at_end
rm -r -f $OUTPUT_DIR/$DIR_NAME/checkpoint*
rm -r -f $OUTPUT_DIR/$DIR_NAME/pytorch_model.bin
                