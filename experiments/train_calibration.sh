#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=01-00:00:00

#SBATCH -o /scratch-shared/dwu18/cache/logs/out.calibration.%j.o
#SBATCH -o /scratch-shared/dwu18/cache/logs/out.calibration.%j.e

# source activate llama_factory
source activate py38cuda11
# source activate calibration

export HF_HUB_CACHE=/gpfs/work4/0/gus20642/dwu18/cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

evaluate_lang_directions() {
    # Parameters
    local TEST_DATASET="$1"  # Test dataset name
    local BASE_SYS="$2"     # Base system directory

    # Define language directions (can customize or pass as parameter if needed)
    local LANG_DIRECTIONS=("en-zh" "en-de" "en-ru" "en-cs" "en-is" "zh-en" "de-en" "ru-en" "cs-en" "is-en")

    # Define base source and target directories
    local BASE_SRC="/home/dwu18/projects/value_finetuning/src/llama_recipes/customer_data/${TEST_DATASET}/test"
    local BASE_TGT="/home/dwu18/projects/value_finetuning/src/llama_recipes/customer_data/${TEST_DATASET}/test"

    # Loop through each language direction
    for LANG_DIR in "${LANG_DIRECTIONS[@]}"; do
        # Extract source and target language codes
        local SRC_LANG=$(echo $LANG_DIR | cut -d'-' -f1)
        local TGT_LANG=$(echo $LANG_DIR | cut -d'-' -f2)

        # Define the file paths
        local SRC_FILE="${BASE_SRC}/${LANG_DIR}/test.${LANG_DIR}.${SRC_LANG}"
        local TGT_FILE="${BASE_TGT}/${LANG_DIR}/test.${LANG_DIR}.${TGT_LANG}"
        local SYS_FILE="${BASE_SYS}/${LANG_DIR}/hyp.${LANG_DIR}.${TGT_LANG}"

        # Define the output score files
        local COMET_SCORE_FILE="./${BASE_SYS}/${LANG_DIR}/comet.score"
        local KIWI_SCORE_FILE="./${BASE_SYS}/${LANG_DIR}/kiwi.score"

        echo "Calculating COMET scores for ${LANG_DIR}..."

        # Run COMET scoring
        comet-score -s $SRC_FILE -t $SYS_FILE -r $TGT_FILE --model Unbabel/wmt22-comet-da     >> $COMET_SCORE_FILE
        comet-score -s $SRC_FILE -t $SYS_FILE -r $TGT_FILE --model Unbabel/wmt22-cometkiwi-da >> $KIWI_SCORE_FILE

        echo "Finished ${LANG_DIR}"
    done

    echo "All language directions processed!"
}



################## MAIN ##################

ALPHA=$1
BETA=$2
GAMA=$3
LR=$4
SUBSET=$5
LIST_SIEZ=$6

echo "ALPHA is set to: $ALPHA"
echo "BETA is set to: $BETA"
echo "GAMA is set to: $GAMA"
echo "LR is set to: $LR"
echo "Subset is set to: $SUBSET"
echo "List size is set to: $LIST_SIEZ"

# Train ALMA
# final_loss = alpha * chose_nll_acc_loss + beta * value_acc_loss + gama * cpo_acc_loss

# calibrate with ALMA-R dataset
SETTING=${ALPHA}-${BETA}-${GAMA}-${LR}
TEST_DATASET=wmt22_testset

python -m llama_recipes.calibration --use_peft --peft_method lora \
        --model_name haoranxu/ALMA-7B-Pretrain \
        --preload_peft_dir haoranxu/ALMA-7B-Pretrain-LoRA \
        --output_dir ./checkpoints/7B/calibration/${SUBSET}/${SETTING} \
        --dataset calibration \
        --subset_name ${SUBSET} \
        --batching_strategy padding \
        --num_epochs 2 \
        --lr $LR \
        --batch_size_training 32 \
        --val_batch_size 32 \
        --gradient_accumulation_steps 8 \
        --alpha $ALPHA \
        --beta $BETA \
	--gama $GAMA \
        --lang_pairs "en-zh,en-de,en-ru,en-cs,en-is,zh-en,de-en,ru-en,cs-en,is-en" \
        --listwise_loss \
        --list_size $LIST_SIEZ \
        --use_wandb

for EPOCH in 0 1; do
    BASE_SYS=results/calibration/${TEST_DATASET}/${SUBSET}/${SETTING}-beam5/${EPOCH}
    python inference_formal.py --model_name haoranxu/ALMA-7B-Pretrain \
            --preload_peft_dir haoranxu/ALMA-7B-Pretrain-LoRA \
            --peft_model ./checkpoints/7B/calibration/${SUBSET}/${SETTING}/${EPOCH} \
            --dataset ${TEST_DATASET} \
            --val_batch_size 8 \
            --do_sample False \
            --output_dir ${BASE_SYS} \
            --lang_pairs en-zh,en-de,en-ru,en-cs,en-is,zh-en,de-en,ru-en,cs-en,is-en \
            --beam_size 5

    evaluate_lang_directions ${TEST_DATASET} ${BASE_SYS}
done

:<<!
# Train ALMA with VPO
CUDA_LAUNCH_BLOCKING=1 torchrun --nnodes 1 --nproc_per_node 2 -m llama_recipes.calibration --enable_fsdp --use_peft --peft_method lora \
        --model_name haoranxu/ALMA-7B-Pretrain \
        --preload_peft_dir haoranxu/ALMA-7B-Pretrain-LoRA \
        --output_dir ./checkpoints_l/${SETTING} \
        --dataset haoranxu/ALMA-R-Preference \
        --batching_strategy padding \
        --num_epochs 3 \
        --lr $LR \
        --batch_size_training 16 \
        --gradient_accumulation_steps 8 \
        --xpo_hyper $XPO_HYPER \
        --alpha $ALPHA \
        --beta $BETA \
        --lang_pairs "en-zh,en-de,en-ru" \
        --listwise_loss \
        --use_wandb

# Evaluate EPOCH 0,1,2
for EPOCH in 0 1; do
    BASE_SYS=results/calibration/${TEST_DATASET}/${SETTING}-beam5/${EPOCH}
    
    python inference_formal.py --model_name haoranxu/ALMA-7B-Pretrain \
            --preload_peft_dir haoranxu/ALMA-7B-Pretrain-LoRA \
            --peft_model ./checkpoints/7B/${SETTING}/${EPOCH} \
            --dataset ${TEST_DATASET} \
            --val_batch_size 8 \
            --do_sample False \
            --output_dir ${BASE_SYS} \
            --lang_pairs en-zh,en-de,en-ru \
            --beam_size 5

    evaluate_lang_directions ${TEST_DATASET} ${BASE_SYS}
done
!
