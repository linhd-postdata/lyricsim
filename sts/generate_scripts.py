
bl = "\\"
dir_name = "DIR_NAME='sts'_${BATCH_SIZE_PER_GPU}_${WEIGHT_DECAY}_${LEARN_RATE}_$(date +'%m-%d-%y_%H-%M')"
models = [
    {'m':'stsb_xlm_r_multilingual'},
    {'m':'bertin_roberta_base'}, 
    {'m':'alberti_base'},
    {'m':'roberta_base_bne'}, 
    {'m':'roberta_large_bne', 'grad':2}, 
]

names = {
    'stsb_xlm_r_multilingual':'sentence-transformers/stsb-xlm-r-multilingual', 
    'bertin_roberta_base':'bertin-project/bertin-roberta-base-spanish', 
    'alberti_base':'flax-community/alberti-bert-base-multilingual-cased',
    'roberta_base_bne':'PlanTL-GOB-ES/roberta-base-bne',
    'roberta_large_bne':'PlanTL-GOB-ES/roberta-large-bne', 
   }

batches = [8, 16, 32]
weight_dec = ['0.1', '0.01']
learn_rate = ['0.00001','0.00002','0.00003', '0.00005']

for model_d in models:
    model = model_d['m']
    name = names[model]
    grad = model_d['grad'] if 'grad' in model_d else 1
    for batch in batches:
        if batch == 32:
            grad *=2
        batch_noGrad = int(batch / grad)
        for weight in weight_dec:
            for lr in learn_rate:
                with open(f"{model}_sts_batch{batch}_lr{lr}_decay{weight}.sh", "w") as f:
                    f.write(f"""#!/bin/bash
SEED=42
NUM_EPOCHS=5
BATCH_SIZE={batch_noGrad}
GRADIENT_ACC_STEPS={grad}
BATCH_SIZE_PER_GPU=$(( $BATCH_SIZE*$GRADIENT_ACC_STEPS ))
LEARN_RATE={lr}
WARMUP=0.06
WEIGHT_DECAY={weight}
MAX_SEQ_LENGTH=512

MODEL='{name}'
OUTPUT_DIR='./outputs/{model}-output'
LOGGING_DIR='./logs/{model}.log'
{dir_name}

python ../bsc_run_glue.py --model_name_or_path $MODEL --seed $SEED {bl}
                                         --dataset_script_path ./sts_dataset.py {bl}
                                         --task_name stsb --do_train --do_eval --do_predict {bl}
                                         --num_train_epochs $NUM_EPOCHS --gradient_accumulation_steps $GRADIENT_ACC_STEPS --per_device_train_batch_size $BATCH_SIZE {bl}
                                         --learning_rate $LEARN_RATE --max_seq_length $MAX_SEQ_LENGTH {bl}
                                         --warmup_ratio $WARMUP --weight_decay $WEIGHT_DECAY {bl}
                                         --output_dir $OUTPUT_DIR/$DIR_NAME --overwrite_output_dir {bl}
                                         --logging_dir $LOGGING_DIR/$DIR_NAME --logging_strategy epoch {bl}
                                         --overwrite_cache {bl}
                                         --metric_for_best_model combined_score --save_strategy epoch --evaluation_strategy epoch --load_best_model_at_end
rm -r -f $OUTPUT_DIR/$DIR_NAME/checkpoint*
rm -r -f $OUTPUT_DIR/$DIR_NAME/pytorch_model.bin
                """)