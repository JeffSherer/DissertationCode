#!/bin/bash
#SBATCH --job-name=llava_med_training
#SBATCH --output=llava_med_output_%j.txt
#SBATCH --error=llava_med_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --mem=320G

# Load necessary modules (if needed)
# source /etc/profile.d/modules.sh
# module load python/3.10
# module load cuda/11.7

# Activate conda environment
conda activate llava-med

# Run the training script
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /path/to/llama-med-vicuna-7b \
    --data_path /path/to/llava_med_instruct_60k_inline_mention_post.jsonl \
    --image_folder /data/to/llava_med_instruct_images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir /path/to/checkpoint_llava_med_instruct_60k_inline_mention \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
